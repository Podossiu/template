import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from utils.config import FLAGS

def quantize_k(input, bit):
    Nlv = pow(2, bit) - 1
    res = torch.round(input * Nlv) / Nlv
    return res

class ActQuant(Function):
    @staticmethod
    def forward(ctx, input, bit):
        if bit == 32:
            out = input
        else:
            out = quantize_k(torch.clamp(input, 0, 1), bit)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class weight_DoReFaQuant(Function):
    @staticmethod
    def forward(ctx, input, bit):
        epsilon = 1e-7
        if bit == 32:
            return input
        elif bit == 1 :
            E = torch.mean(torch.abs(input)).detach()
            res = E * torch.sign(input /(E + epsilon))
        else :
            tanh = torch.tanh(input)
            res = 2 * quantize_k(tanh / (2 * torch.max(torch.abs(tanh)) + epsilon) + 0.5, bit) - 1
        return res
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



class q_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                dilation = 1, groups = 1, bias = False):
                super(q_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, groups,
                                             dilation, bias)
                self.weight_quantize = weight_DoReFaQuant.apply
                self.weight_bitwidth = FLAGS.weight_bitwidth
    
    def forward(self, x):
        v_q = self.weight_quantize(self.weight, self.weight_bitwidth)
        
        return F.conv2d(x, v_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class q_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
                super(q_Linear, self).__init__(in_features, out_features, bias)
                self.weight_quantize = weight_DoReFaQuant.apply
                self.weight_bitwidth = FLAGS.weight_bitwidth

    def forward(self, x):
        v_q = self.weight_quantize(self.weight, self.weight_bitwidth)
        
        return F.linear(x, v_q, self.bias)
