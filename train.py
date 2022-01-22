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
import utils.data as data
import random 
import wandb

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
    model = torch.nn.DataParallel(model).cuda()
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
        if phase == 'cal':
            if batch_idx == getattr(FLAGS, 'bn_cal_batch_num', -1):
                break
        target = target.cuda(non_blocking = True)
        if train:
            optimizer.zero_grad()
            loss = forward_loss(
                    model, criterion, input, target, meters)
            
            if epoch >= FLAGS.warmup_epochs and not getattr(FLAGS, 'hard_assignment', False):
                if getattr(FLAGS, 'weight_only', False):
                    loss += getattr(FLAGS, 'kappa', 1.0) * get_model_size_loss(model)
                else:
                    loss += getattr(FLAGS, 'kappa', 1.0) * get_model_cost_loss(model)
            loss.backward()
            optimizer.step()
            
            # if FLAGS.lr_scheduler in ['cos_annealing_iter']:
            


            

def train_val_test():
    """ train and val"""
    torch.backends.cudnn.benchmark = True
    
    

def forward_and_loss(model, criterion, input, target, meter):
    """ forward model and return loss """
    if getattr(FLAGS, 'normalize', False):
        input = input # ( 128 * input ).round_().clamp_(-128, 127)
    else :
        input = ( 255 * input).round_()
    output = model(input)
    loss = torch.mean(criterion(output, target))
    # topk --> value, index 
    _, pred = output.topk(max(FLAGS.topk)) 
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = []
    for k in FLAGS.topk:
        correct_k.append(correct[:k].float().sum(0))
    res = torch.cat(correct_k, dim = 0)

    res = res.cpu().detach().numpy()
    bs = (res.size - 1) // len(FLAGS.topk)

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
    # check the save_dir exists or not
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    #모델 설정
    set_random_seed()
    
    # print(model)    
    train_transform, val_transform = data.get_transform()
    # print(train_transform, val_transform)
    train_set, val_set = data.dataset(train_transform, val_transform)
    # print(train_set, val_set)
    train_dataloader, val_dataloader = data.get_data_loader(train_set, val_set)
    # print(train_dataloader, val_dataloader)
    
    model = torch.nn.DataParallel(get_model()).cuda()
    optimizer = get_optimizer(model)
    lr_scheduler = get_scheduler(optimizer)


    log_dir = FLAGS.log_dir
    log_dir = os.path.join(log_dir, experiment_setting)
    print(log_dir, model, optimizer, lr_scheduler)
    """
    # train
    print(' train '.center(40, '*'))
    for epoch in range(last_epoch+1, FLAGS.num_epochs):
        run_one_epoch(epoch, train_loader, model, criterion, optimizer, train_meters, phase = 'train',
                    scheduler = lr_scheduler)

    # val
    print(' validation '.center(40, '~'))
    if val_meters is not None:
        val_meters['best_val'].cache(best_val)
    with torch.no_grad():
        if epoch == getattr(FLAGS, 'hard_assign_epoch', float('inf')):
            print('Start to use hard assignment')
            setattr(FLAGS, 'hard_assignment', True)
            lower_offset = -1
            higher_offset = 0
            setattr(FLAGS, 'hard_offset', 0)
   """     
if __name__ == "__main__":
    main()
