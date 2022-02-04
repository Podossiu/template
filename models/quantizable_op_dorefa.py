import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
from utils.config import FLAGS

torch.autograd.set_detect_anomaly(True)
class quantize_k(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit):
        Nlv = float(2**bit - 1)
        res = torch.round(input * Nlv) / Nlv
        return res

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

class ActQuant(nn.Module):
    def __init__(self):
        super(ActQuant,self).__init__()
        self.a_bit = FLAGS.activation_bitwidth
        self.quantize_k = quantize_k.apply
        
    def forward(self, input):
        if self.a_bit == 32:
            out = input
        else:
            out = self.quantize_k(torch.clamp(input, 0, 1), self.a_bit)
        return out

class weight_DoReFaQuant(nn.Module):
    def __init__(self):
        super(weight_DoReFaQuant, self).__init__()
        self.quantize_k = quantize_k.apply
        self.w_bit = FLAGS.weight_bitwidth

    def forward(self, input):
        epsilon = 1e-7
        if self.w_bit == 32:
            return input
        elif self.w_bit == 1 :
            E = torch.mean(torch.abs(input)).detach()
            res = torch.sign(input)
            res[res == 0] == 1
            res = E * torch.sign(input / E)
        else :
            tanh = torch.tanh(input)
            res = 2 *self.quantize_k(tanh / (2 * torch.max(torch.abs(tanh)).detach()) + 0.5, self.w_bit) - 1
        return res

class q_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                dilation = 1, groups = 1, bias = False):
                super(q_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, groups,
                                             dilation, bias)
                self.weight_quantize = weight_DoReFaQuant()
    def forward(self, x):
        v_q = self.weight_quantize(self.weight)
        return F.conv2d(x, v_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

class q_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
                super(q_Linear, self).__init__(in_features, out_features, bias)
                self.weight_quantize = weight_DoReFaQuant()

    def forward(self, x):
        v_q = self.weight_quantize(self.weight)
        return F.linear(x, v_q, self.bias)
