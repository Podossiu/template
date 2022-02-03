import datetime
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
from utils.AverageMeter import *
import utils.data as data
import random 
import wandb

# experiment setting
def get_experiment_setting():
    experiment_setting = 'weight_bitwidth_{}_activation_bitwidth_{}'.format(getattr(FLAGS, 'weight_bitwidth',8), getattr(FLAGS, 'activation_bitwidth', 8))
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
        model = q_resnet.Model(FLAGS.num_classes)
    print(model)
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
        meters['loss'] = ScalarMeter('{}_loss/{}'.format(phase, suffix))
        for k in FLAGS.topk:
             meters['top{}_error'.format(k)] = ScalarMeter(
		    '{}_top{}_error/{}'.format(phase, k, suffix))
        return meters
    
    assert phase in ['train', 'val', 'test'], 'Invalid phase.'
    
    meters = get_single_meter(phase)
    if phase == 'val':
        meters['best_val'] = ScalarMeter('best_val')
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
    
    eval_acc_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    loss_acc_list = []
    acc1_iter_list = []
    acc1_avg_list = []

    for batch_idx, (input, target) in enumerate(loader):
        
        #print(input.is_cuda, target.is_cuda)
        if phase == 'cal':
            if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                break
        target = target.cuda(non_blocking = True)
        if train:
            optimizer.zero_grad()
            output, loss = forward_and_loss(model, criterion, input, target, meters)
            
            loss.backward()
            optimizer.step()
            if scheduler != None:
                scheduler.step()
            
            # meter tracking
            acc1, acc5 = accuracy(output.data, target.data, top_k = (1,5))
            eval_acc_loss.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            acc1_iter_list.append(acc1.item())
            acc1_avg_list.append(top1.avg.item())
            

            if batch_idx % getattr(FLAGS, "log_frequency", 10) == 0:
                if getattr(FLAGS, 'log_wandb', False):
                    log_dict = {'acc1_iter' : acc1.item(),
                                'acc1_avg'  : top1.avg,
                                'acc5_avg'  : top5.avg,
                                'loss'      : loss.item()}
                    wandb.log(log_dict)
                curr = batch_idx * len(input)
                total = len(loader.dataset)
                loss_sentence = f'Loss_acc : {eval_acc_loss.avg:5.3f} |'
                print(f'[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Train Epoch: '\
                    f'{epoch:3d} Phase: {phase} Process: {curr:5d}/{total:5d}  '\
                    + loss_sentence + \
                    f'top1.avg: {top1.avg:.3f} % | '\
                    f'top5.avg: {top5.avg:.3f} % | ')   ## me!! eval_loss -> eval_acc_loss ##
                print(f'loss : {loss}')
        else:
            output, _ = forward_and_loss(model, criterion, input, target, meters)
            acc1, acc5 = accuracy(output.data, target.data, top_k=(1,5))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            
    val_top1 = None
    try:
        print('{:.1f}s\t{}\t{}: '.format(
        time.time() - t_start, phase, epoch, FLAGS.num_epochs)) # +
        #', '.join('{}: {}'.format(k, v) for k, v in results.items()))
        val_top1 = top1.avg
        #val_top1 = results['top1_error']
    except:
        val_top1 = top1.avg
    if phase == 'val':
        if getattr(FLAGS, 'log_wandb', False):
            wandb.log({'eval_top1': top1.avg,
                    'eval_top5': top5.avg})
    return val_top1

def get_pretrained_model(model):
    model_link = {'models.q_mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
                'models.q_resnet': 'https://download.pytorch.org/models/resnet18-f37072fd.pth'}

    # full precision pretrained
    if getattr(FLAGS, 'fp_pretrained_file', None):
        if not os.path.isfile(FLAGS.fp_pretrained_file):
            pretrained_dir = os.path.dirname(FLAGS.fp_pretrained_file)
            print(FLAGS.fp_pretrained_file)
            os.system(f"wget -p {pretrained_dir} {model_link[FLAGS.model]}")
        checkpoint = torch.load(FLAGS.fp_pretrained_file)

        if type(checkpoint) == dict and 'model' in checkpoint:
            checkpoint = checkpoint['model']
        if getattr(FLAGS, 'pretrained_model_remap_keys', False):
            new_checkpoint = {}
            new_keys = list(model.state_dict().keys())
            old_keys = list(checkpoint.keys())
            for key_new in new_keys:
                for i, key_old in enumerate(old_keys):
                    if key_old.split('.')[-1] in key_new:
                        new_checkpoint[key_new] = checkpoint[key_old]
                        print('remap {} to {}'.format(key_new, key_old))
                        old_keys.pop(i)
                        break
            checkpoint = new_checkpoint
        model_dict = model.state_dict()

        checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
        # remove unexpected keys
        for k in list(checkpoint_keys()):
            if k not in model_dict.keys():
                checkpoint.pop(k)
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
        return model

def get_checkpoint_model(model, optimizer):
    checkpoint = torch.load(os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['last_epoch']
    
    if FLAGS.lr_scheduler in ['cos_annealing_iter']:
        lr_scheduler = get_scheduler(optimizer)
        lr_scheduler.last_epoch = last_epoch
    
    best_val = checkpoint['best_val']
    train_meters, val_meters = checkpoint['meters']
    print('loaded checkpoint {} at epoch {}.'.format(FLAGS.log_dir, last_epoch))
    return model, optimizer, last_epoch, lr_scheduler, best_val, train_meters, val_meters


def train_val_test():
    """ train and val"""
    torch.backends.cudnn.benchmark = True
    set_random_seed()
    experiment_setting = get_experiment_setting()
    
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
    
    if getattr(FLAGS, 'fp_pretrained_model', None):
        model = get_pretrained_model(model)
        print('loaded full precision model {}.'.format(FLAGS.fp_pretrained_file))
    else :
        print('loaded random value model')

    # check resume
    if os.path.exists(os.path.join(log_dir, 'latest_checkpoint.pt')):
        if os.path.isfile(os.path.join(log_dir, 'latest_checkpoint.pt')):
            model, optimizer, last_epoch, lr_scheduler, best_val, train_meters, val_meters = get_checkpoint_model(model, optimizer)
    else:
        last_epoch = 0
        best_val = 0.
        train_meters = get_meters('train')
        val_meters = get_meters('val')
        print('Training from scratch')
    
    if getattr(FLAGS, 'log_wandb', False):
        PROJECT_NAME = 'DoReFa'
        wandb.init(project = PROJECT_NAME, dir = FLAGS.log_dir)
        wandb.config.update(FLAGS)


    if FLAGS.test_only and (test_loader is not None):
        print('Start testing')
        test_meters = get_meters('test')
        with torch.no_grad():
            run_one_epoch(
                -1, test_loader,
                model, criterion, optimizer,
                test_meters, phase = 'test')
        return
    
    if getattr(FLAGS, 'log_dir', None):
        print(os.path.exists(log_dir))
        if os.path.exists(log_dir) == False:
            print(os.path.exists(log_dir))
            os.makedirs(log_dir)
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
            top1_acc = run_one_epoch(
                epoch, val_dataloader, model, criterion, optimizer,
                val_meters, phase = 'val')
        
        if top1_acc > best_val:
            best_val = top1_acc
            torch.save(
                {
                    'model': model.state_dict()
                },
                os.path.join(FLAGS.log_dir, 'best_model.pt'),
                )

            print('==> New best validation top1 error: {:.3f} %'.format(best_val))
            
            # save latest checkpoint
            torch.save(
                {
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'last_epoch' : epoch,
                    'best_val' : best_val,
                    'meters': (train_meters, val_meters),
                },
                os.path.join(FLAGS.log_dir, 'latest_checkpoint.pt')
                )
    return


                

def forward_and_loss(model, criterion, input, target, meter):
    """ forward model and return loss"""
    output = model(input.cuda())
    loss = torch.mean(criterion(output, target.cuda()))
    
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
    return output, loss


def main():
    train_val_test()

if __name__ == "__main__":
    main()

