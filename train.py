import argparse
import math
import random
import time
import importlib
import os
from functools import wraps
import sys
import copy
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from models import q_resnet
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.nn.modules.utils import _pair
from utils.config import FLAGS
import utils.AverageMeter as AverageMeter
import utils.data as data
import random 
import wandb

# experiment setting
def get_experiment_setting():
    experiment_setting = 'ema_decay_{ema_decay}/fp_pretrained_{fp_pretrained}/bit_list_{bit_list}'.format(ema_decay=getattr(FLAGS, 'ema_decay', None), fp_pretrained=getattr(FLAGS, 'fp_pretrained_file', None) is not None,  bit_list='_'.join([str(i) for i in getattr(FLAGS, 'bits_list', [])]))
    if getattr(FLAGS, 'act_bits_list', False):
        experiment_setting = os.path.join(experiment_setting, 'act_bits_list_{}'.format('_'.join([str(i) for i in FLAGS.act_bits_list])))
    if getattr(FLAGS, 'double_side', False):
        experiment_setting = os.path.join(experiment_setting, 'double_side_True')
    if not getattr(FLAGS, 'rescale', False):
        experiment_setting = os.path.join(experiment_setting, 'rescale_False')
    if not getattr(FLAGS, 'calib_pact', False):
        experiment_setting = os.path.join(experiment_setting, 'calib_pact_False')
    experiment_setting = os.path.join(experiment_setting, 'kappa_{kappa}'.format(kappa=getattr(FLAGS, 'kappa', 1.0)))
    if getattr(FLAGS, 'target_bitops', False):
        experiment_setting = os.path.join(experiment_setting, 'target_bitops_{}'.format(getattr(FLAGS, 'target_bitops', False)))
    if getattr(FLAGS, 'target_size', False):
        experiment_setting = os.path.join(experiment_setting, 'target_size_{}'.format(getattr(FLAGS, 'target_size', False)))
    if getattr(FLAGS, 'init_bit', False):
        experiment_setting = os.path.join(experiment_setting, 'init_bit_{}'.format(getattr(FLAGS, 'init_bit', False)))
    if getattr(FLAGS, 'unbiased', False):
        experiment_setting = os.path.join(experiment_setting, f'unbiased_True')
    print('Experiment settings: {}'.format(experiment_setting))
    return experiment_setting


# seed
def set_random_seed(seed=None):
    """set random seed"""
    if seed is None:
        seed = getattr(FLAGS, 'random_seed', 0)
    print('seed for random sampling: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_model():
    # import lib을 통해 model_lib에 module자체를 불러옴
    if FLAGS.model == 'resnet':
        model = q_resnet.Model(FLAGS.depth)
    model = model
    return model

def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == 'sgd':
        # all depthwise convolution ( N, 1, x, x ) has no weight decay
        # weight decay only on normal conv and fc
        optimizer = torch.optim.SGD(model.parameters(), FLAGS.lr,
                momentum = getattr(FLAGS, 'momentum', 0.9),
                weight_decay = getattr(FLAGS, 'weight_decay', 0))
    return optimizer

def get_meters(phase):
    """util function for meters"""
    def get_single_meter(phase, suffix = ''):
        meters = {}
        meters['loss'] = AverageMeter.ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
             meters['top{}_error'.format(k)] = AverageMeter.ScalarMeter(
		    '{}_top{}_error/{}'.format(phase, k, suffix))
        return meters
    
    assert phase in ['train', 'val', 'test'], 'Invalid phase.'
    
    meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = AverageMeter.ScalarMeter('best_val')
    return meters

def get_scheduler(optimizer):
    if getattr(FLAGS, "lr_scheduler", "cos_annealing_iter"):
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 0)
    else:
        lr_scheduler = None
    return lr_scheduler

def run_one_epoch(
        epoch, loader, model, criterion, optimizer, meters, phase = 'train', scheduler = None):
    """ run one epoch for train/val/test"""
    t_start = time.time()
    assert phase in ['train', 'val', 'test', 'cal'], "phase not be in train/val/test/cal."
    
    train = phase == 'train'
    if train:
        model.train()
    else:
        model.eval()
    
    for batch_idx, (input, target) in enumerate(loader):
        input.to(next(model.parameters()).device)
        target.to(next(model.parameters()).device)
        print(input.is_cuda, target.is_cuda)
        if phase == 'cal':
            if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                break
        target = target.cuda(non_blocking = True)
        if train:
            optimizer.zero_grad()
            loss = forward_and_loss(model, criterion, input, target, meters)
            # BITOPS Loss
            if epoch >= FLAGS.warmup_epochs and not getattr(FLAGS, 'hard_assignment', False):
                if getattr(FLAGS, 'wegith_only', False):
                    loss += getattr(FLAGS, 'kappa', 1.0) * get_model_size_loss(model)
                else:
                    loss += getattr(FLAGS, 'kappa', 1.0) * get_comp_cost_loss(model)
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            # if ema # 일단 not use
          
    val_top1 = None
    results = flush_scalar_meters(meters)
    print('{:.1f}s\t{}\t{}/{}: '.format(
        time.time() - t_start, phase, epoch, FLAGS.num_epochs) +
            ', '.join('{}: {}'.format(k, v) for k,v in results.items()))
    val_top1 = results['top1_error']
    return val_top1

def train_val_test():
    """ train and val"""
    torch.backends.cudnn.benchmark = True
    set_random_seed()
    experiment_setting = get_experiment_setting()
    
    model = get_model()
    criterion = torch.nn.CrossEntropyLoss(reduction = 'none').cuda()
    
    # print(model)    
    train_transform, val_transform = data.get_transform()
    # print(train_transform, val_transform)
    train_set, val_set = data.dataset(train_transform, val_transform)
    # print(train_set, val_set)
    train_dataloader, val_dataloader = data.get_data_loader(train_set, val_set)
    # print(train_dataloader, val_dataloader)
    test_loader = None
    model = get_model().cuda()
    optimizer = get_optimizer(model)
    lr_scheduler = get_scheduler(optimizer)
    log_dir = FLAGS.log_dir
    log_dir = os.path.join(log_dir, experiment_setting)

    if FLAGS.test_only and (test_loader is not None):
        print('Start testing')
        test_meters = get_meters('test')
        with torch.no_grad():
            run_one_epoch(
                -1, test_loader,
                model, criterion, optimizer,
                test_meters, phase = 'test')
        return

    last_epoch = 0
    best_val = 1.
    train_meters = get_meters('train')
    val_meters = get_meters('val')
    
    if  getattr(FLAGS, 'log_dir', None):
        try:
            os.mkdir(log_dir)
        except OSError:
            pass

    # check resume training
    print('start training.')
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        print(' train '.center(40, '*'))
        run_one_epoch(
            epoch, train_dataloader, model, criterion, optimizer, 
            train_meters, phase = 'train', scheduler = lr_scheduler)
        
        print(' validation '.center(40, '~'))
        if val_meters is not None:
            val_meters['best_val'].cache(best_val)
        with torch.no_grad():
            bit_discretizing(model)
            setattr(FLAGS, 'hard_offset', 0)
    
            top1_error= run_one_epoch(
                epoch, val_dataloader, model, criterion, optimizer,
                val_meters, phase = 'val')
        
        if top1_error < best_val:
            best_val = top1_error
            torch.save(
                os.path.join(log_dir, 'best_model.pt'),
                {
                    'model': model.state_dict()
                }
                )
            print('New best validation top1 error: {:.3f}'.format(best_val))

            # save latest checkpoint
            torch.save(
                os.path.join(log_dir, 'latest_checkpoint.pt'),
                {
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'last_epoch' : epoch,
                    'best_val' : best_val,
                    'meters': (train_meters, val_meters),
                })
        for m in model.modules():
            if hasattr(m, 'alpha'):
                print(m, m.alpha)
            if hasattr(m, 'lamda_w'):
                print(m, m.lamda_w)
            if hasattr(m, 'lamda_a'):
                print(m, m.lamda_a)
        return


                

def forward_and_loss(model, criterion, input, target, meter):
    """ forward model and return loss"""
    if getattr(FLAGS, 'normalize', False):
        input = input # normalized input
    else:
        input = ( 255 * input ).round_()
    output = model(input)
    loss = torch.mean(criterion(output, target))
    
    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    res = torch.cat(correct_k, dim=0)
    res = res.cpu().detach().numpy()
    bs = (res.size - 1) // len(FLAGS.topk)
    for i, k in enumerate(FLAGS.topk):
        error_list = list(1. - res[i*bs:(i+1)*bs])
        if meter is not None:
            meter['top{}_error'.format(k)].cache_list(error_list)
    if meter is not None:
        meter['loss'].cache(loss.tolist())
    return loss

  

def bit_discretizing(model):
    print('hard offset', FLAGS.hard_offset)
    for m in model.modules():
        if hasattr(m, 'bit_discretizing'):
            pirnt('bit discretized for ', m)
            m.bit_discretizing()

def get_comp_cost_loss(model):
    loss = 0.0
    for m in model.modules():
        loss += getattr(m, 'comp_cost_loss', 0.0)
    target_bitops = getattr(FLAGS, 'target_bitops', False)
    if target_bitops:
        loss = torch.abs(loss - target_bitops)
    return loss

def get_model_size_loss(model):
    loss = 0.0
    for m in model.modules():
        loss += getattr(m, 'model_size_loss', 0.0)
    target_size = getattr(FLAGS, 'target_size', False)
    if target_size:
        loss = torch.abs(loss - target_size)
    return loss

def get_pre_trained():
    if not os.path.isfile(FLAGS.fp_pretrained_file):
        pretrain_dir = os.path.dirname(FLAGS.fp_pretrained_file)
        print(FLAGS.fp_pretrained_file)
        os.system(f"wget -P {pretrained_dir} {model_link[FLAGS.model]}")
    checkpoint = torch.load(FLAGS.fp_pretrained_file)

    # update keys from external models
    if type(checkpoint) == dict and 'model' in checkpoint:
        checkpoint = checkpoint['model']
    if getattr(FLAGS, 'pretrained_model_remap_keys', False):
        new_checkpoint = {}
        new_keys = list(model_wrapper.state_dict().keys())
        old_keys = list(checkpoint.keys())
        for key_new in new_keys:
            for i, key_old in enumerate(old_keys):
                if key_old.split('.')[-1] in key_new:
                    new_checkpoint[key_new] = checkpoint[key_old]
                    print('remap {} to {}'.format(key_new, key_old))
                    old_keys.pop(i)
                    break

def get_check_point(model):
    PATH = os.path.join(log_dir, 'latest_checkpoint.pt')
    if os.path.exist(PATH):
       checkpoint = torch.load(
               os.patj.join(log_dir, 'latest_checkpoint.pt'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['last_epoch']
    if FLAGS.lr_scheduler in ['cos_annealing_iter']:
        lr_scheduler = get_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
    best_val = checkpoint['best_val']
    train_meters, val_meters = checkpoint['meters']

    print('Loaded checkpoint {} at epoch {}.'.format(log_dir, last_epoch))
    return model, optimizer, lr_scheduler, best_val, train_meters, val_meters, last_epoch



def main():
    train_val_test()

if __name__ == "__main__":
    main()
