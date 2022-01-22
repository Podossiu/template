import math
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from utils.config import FLAGS
# Batch normalization calibartion

# output shape : ( input + 2 * padding - (dilation ) * (kernel_size - 1) - 1) // stride + 1
def out_shape(i ,p , d, k , s):
    return (i + 2 * p - d * ( k - 1 ) - 1) // s + 1


# EMA 

# Quantize function
"""
class Quantize_k(Function):
"""
"""
        This is the quantization function
        The input and output should be all on the interval [0, 1].
        bit is only defined on nonnegative integer values
        zero_point is the value used for 0-bit, and should be on the interval [0, 1].
"""
"""
    @staticmethod
    def forward(ctx, input, bit = torch.tensor([8]), align_dim = 0, zero_point = 0, scheme = 'modified'):
        assert torch.all(bit >= 0)
        assert torch.all(input>=0) and torch.all(input <= 1)
        assert zero_point >= 0 and zero_point <= 1
        # dorefanet scheme
        if scheme == 'original':

"""
if __name__ == "__main__":
    print(1)
    print(FLAGS['lr'])
