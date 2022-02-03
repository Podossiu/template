import torch
import torch.nn
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.nn.modules.utils import _pair
from utils.config import FLAGS
import random
import os
def dataset(train_transforms, val_transforms):
    # dataset이 imagenet일 때
    datasetdir = getattr(FLAGS, 'datasetdir', '/data')
    if FLAGS.dataset == 'imagenet':
        train_set = datasets.ImageFolder(
                os.path.join(datasetdir, 'imagenet', 'train'),
                transform = train_transforms)
        val_set = datasets.ImageFolder(
                os.path.join(datasetdir, 'imagenet', 'val'),
                transform = val_transforms)
    elif FLAGS.dataset == 'cifar100':
        train_set = datasets.CIFAR100(root = "/data", train = True, transform = train_transforms)
        val_set = datasets.CIFAR100(root = "/data", train = False, transform = val_transforms)
    
    elif FLAGS.dataset == 'cifar10':
        train_set = datasets.CIFAR10(root = "/data", train = True, transform = train_transforms)
        val_set = datasets.CIFAR10(root = "/data", train = False, transform = val_transforms)
    return train_set, val_set

def get_transform():

    if FLAGS.dataset == 'imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
       
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),
            ])
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),
            ])

    elif FLAGS.dataset == 'cifar100':

        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),
            ])
    elif FLAGS.dataset == 'cifar10':
        
        mean = [ 0.4914, 0.4822, 0.4465 ]
        std = [ 0.2023, 0.1994, 0.2010 ]
        
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),
            ])

        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std),
            ])

    return train_transforms, val_transforms

def get_data_loader(train_set, val_set):
    """get data loader"""
    train_loader = None
    val_loader = None

    train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = FLAGS.batch_size,
            shuffle = True,
            num_workers = FLAGS.data_loader_workers,
            pin_memory = True)
    val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size = getattr(FLAGS, 'val_batch_size', 128),
            shuffle = False,
            num_workers = FLAGS.data_loader_workers,
            pin_memory = True)

    return train_loader, val_loader


