from collections import OrderedDict

import os
import functools

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors
from torch._utils import _take_tensors

# init_dist 함수 정의 launcher & backend
def init_dist(launcher = 'pytorch', backend = 'nccl', **kwargs):
    # 프로세스 그룹이 초기화되어있는지 확인
    if dist.is_initialized():
        torch.cuda.current_device()
    
