import math

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from utils.config import FLAGS

# Batch normalization calibration : 학습 후 BN 만 학습
def bn_calibration(m):
    if getattr(m, 'track_running_stats', False):
        m.reset_running_stats()
        m.train()
        #m.training = True
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None
    # if isinstance(m, SwitchableBatchNorm2d):
    #     m.idx = 0


# output shape : ( input + 2 * padding - ( dilation ) * ( kernel size - 1 ) - 1 ) // stride + 1
def out_shape(i, p, d, k, s):
    return (i + 2 * p - d * (k - 1) - 1) // s + 1


# EMA for BN calibration
class EMA():
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}
        self.param_copy = {}
        self.ignore_model_profiling = True

    def shadow_register(self, model, bn_stats=True):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        if bn_stats:
            bn_idx = 0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.shadow['bn{}_mean'.format(bn_idx)] = m.running_mean.clone()
                    self.shadow['bn{}_var'.format(bn_idx)] = m.running_var.clone()
                    bn_idx += 1

    def shadow_update(self, model, bn_stats=True):
        for name, param in model.named_parameters():
            if param.requires_grad:
                #assert name in self.shadow, '{} is not in {}.'.format(name, self.shadow.keys())
                self.shadow[name] -= (1.0 - self.decay) * (self.shadow[name] - param.data)
        if bn_stats:
            bn_idx = 0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    #assert 'bn{}_mean'.format(bn_idx) in self.shadow, '{} is not in {}.'.format('bn{}_mean'.format(bn_idx), self.shadow.keys())
                    self.shadow['bn{}_mean'.format(bn_idx)] -= (1.0 - self.decay) * (self.shadow['bn{}_mean'.format(bn_idx)] - m.running_mean)
                    #assert 'bn{}_var'.format(bn_idx) in self.shadow, '{} is not in {}.'.format('bn{}_var'.format(bn_idx), self.shadow.keys())
                    self.shadow['bn{}_var'.format(bn_idx)] -= (1.0 - self.decay) * (self.shadow['bn{}_var'.format(bn_idx)] - m.running_var)
                    bn_idx += 1

    def shadow_apply(self, model, bn_stats=True):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_copy[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        if bn_stats:
            bn_idx = 0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.param_copy['bn{}_mean'.format(bn_idx)] = m.running_mean.clone()
                    m.running_mean.copy_(self.shadow['bn{}_mean'.format(bn_idx)])
                    self.param_copy['bn{}_var'.format(bn_idx)] = m.running_var.clone()
                    m.running_var.copy_(self.shadow['bn{}_var'.format(bn_idx)])
                    bn_idx += 1

    def weight_recover(self, model, bn_stats=True):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.param_copy[name])
        if bn_stats:
            bn_idx = 0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.running_mean.copy_(self.param_copy['bn{}_mean'.format(bn_idx)])
                    m.running_var.copy_(self.param_copy['bn{}_var'.format(bn_idx)])
                    bn_idx += 1
        self.param_copy = {}


# Quantize function
class Quantize_k(Function):
    """
        This is the quantization function.
        The input and output should be all on the interval [0, 1].
        bit is only defined on nonnegative integer values.
        zero_point is the value used for 0-bit, and should be on the interval [0, 1].
    """
    @staticmethod
    def forward(ctx, input, bit=torch.tensor([8]), align_dim=0, zero_point=0, scheme='modified'):
        assert torch.all(bit >= 0)
        assert torch.all(input >= 0) and torch.all(input <= 1)
        assert zero_point >= 0 and zero_point <= 1
        # dorefaNet scheme
        if scheme == 'original':
            # 2^bit - 1 1개 
            a = torch.pow(2, bit) - 1
            # input dimension의 개수 - align_dim ?? - 1
            expand_dim = input.dim() - align_dim - 1
            # a는 원래 있던거 + input과 dimension 맞춰줌
            a = a[(...,) + (None,) * expand_dim]
            # a * input을 round시켜 input_bar 구함
            res = torch.round(a * input)

            # bit == 0인경우 대비하여 만든거 같은데...
            res.div_(1 + torch.relu(a - 1))
            # zero point 바꾸기 0.5로 --> 알고보니까 input값 [0,1] 음수부분 맞춰주기 위해서  ..? 왜.. ? 
            res.add_(zero_point * torch.relu(1 - a))
        elif scheme == 'modified':
            # modifed version ( adabits ), 2^bit 1개
            a = torch.pow(2, bit)
            # dim 맞춰주기
            expand_dim = input.dim() - align_dim - 1
            # dim 맞춰주기
            a = a[(...,) + (None,) * expand_dim]
            # floor .2^k * input 
            res = torch.floor(a * input)
            # min(floor.2^k * input, 2^k - 1)
            res = torch.clamp(res, max=a - 1)
            # a로 나눔 0되는거 없으니까 relu안했나보다 0 bit도 1
            res.div_(a)
            # 근데 여기서 zero point에서 relu쓰네.. 
            res.add_(zero_point * torch.relu(2 - a))
        else:
            raise NotImplementedError
        assert torch.all(res >= 0) and torch.all(res <= 1)
        return res

    # STE
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None

class QuantizableConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 same_padding=False,
                 lamda_w_min=None, lamda_a_min=None,
                 double_side=False,
                 weight_only=False,
                 input_size=(0, 0)):
        super(QuantizableConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding if not same_padding else 0,
            dilation=dilation,
            groups=groups, bias=bias)
        # same padding
        self.same_padding = same_padding
        
        # bitwidth lamda value for w, a
        self.lamda_w_min = lamda_w_min
        self.lamda_a_min = lamda_a_min
        
        # double side ( clamping double side )
        self.double_side = double_side
        
        # weight only quant
        self.weight_only = weight_only or getattr(FLAGS, 'weight_only', False)
        
        # Quantization autograd apply
        self.quant = Quantize_k.apply
        
        # alpha for Clamp activation ( PACT )
        self.alpha = nn.Parameter(torch.tensor(8.0))
        
        # initial bitwidth 
        init_bit = getattr(FLAGS, 'init_bit', 7.5)
        
        # when channel wise Quantization
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            # Parameter out_channel수만큼 ( channel_wise는 activation 기준이므로 Filter Quantization )
            self.lamda_w = nn.Parameter(torch.ones(self.out_channels) * init_bit)
        else:
            # 아니면 layerwise layer 당 1개 
            self.lamda_w = nn.Parameter(torch.tensor(init_bit))
        # when channel wise Quantization
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            # Parameter in_channel 수만큼 ( channel_wise Quantization )
            self.lamda_a = nn.Parameter(torch.ones(self.in_channels) * init_bit)
        else:
            # layer_wise Quantization
            self.lamda_a = nn.Parameter(torch.tensor(init_bit))
        # for div 0 + eps, gumbel softmax 
        self.eps = 0.00001
        # input_size
        self.input_size = input_size

    def forward(self, input):
        # when same padding
        if self.same_padding:
            # input_height, input width size()[-2:]는 ( B, C, H, W )에서 H, W
            ih, iw = input.size()[-2:]
            # 같은 방식으로 H, W
            kh, kw = self.weight.size()[-2:]
            # stride for height, width
            sh, sw = self.stride
            # output feature map size ( oh, ow ) ceil ih / sh,
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            # (oh - 1 ) * self.stride[0] + (kh - 1 ) * self.dilation + 1 ?  --> padding 양 
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0: # padding 해야하는 경우 
                input = nn.functional.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        
        # lamda_w, lamda_a
        lamda_w = self.lamda_w
        lamda_a = self.lamda_a
        
        # 최소 비트와 최대 비트 내에서 clamp시킴 보통 min = 2, max = 8
        lamda_w = torch.clamp(lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
        if self.lamda_w_min is not None:
            # 최소 비트 제한이 있는 경우 
            lamda_w = torch.clamp(lamda_w, min=self.lamda_w_min)
        # activation 도bitrange  가져와
        act_bits_list = getattr(FLAGS,'act_bits_list',FLAGS.bits_list)
        # 똑같이 clamp시켜 
        lamda_a = torch.clamp(self.lamda_a, min(act_bits_list), max(act_bits_list))
        if self.lamda_a_min is not None:
            lamda_a = torch.clamp(lamda_a, min=self.lamda_a_min)
        
        # weight는 modified로, activation은 original로 맞춤
        weight_quant_scheme = getattr(FLAGS, 'weight_quant_scheme', 'modified')
        act_quant_scheme = getattr(FLAGS, 'act_quant_scheme', 'original')
        
        # weight (-1, 1)로 tanh이용하여 clamp --> tanh() / max(abs(tanh))
        weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
        # tanh() / (2 * max(abs(tanh))) + 1/2--> quantize_k를 위해 ( 0, 1 )로 바뀜 
        weight.add_(1.0)
        weight.div_(2.0)
        
        # get window_size
        window_size = getattr(FLAGS, 'window_size', 2.)
        # importance 계산
        weight_bits_list = torch.ToTensor(FLAGS.bits_list).to(weight.device)
        importance = torch.ToTensor(FLAGS.bits_list).to(weight.device)
        importance = (importance - lamda_w).abs()
        importance = torch.relu(-importance + torch.ToTensor([window_size/2]).to(weight.device))
        importance = importance / importance.sum()
       
        
        # w = f_floor(lamda)(x) + ( lamda - floor(lamda))( f_ceil(lamda)(X) - f_floor(lamda)(x) )
        # floor importance = 1 + floor(lamda_w) - lamda_w
        # p_l = 1 + torch.floor(lamda_w) - lamda_w
        # 더해서 1 
        # p_h = 1 - p_l

        # gumbel_softmax assignment
        if not getattr(FLAGS, 'hard_assignment', False) and getattr(FLAGS,'gumbel_softmax',False):
            # gumbel soft max logits (log (p_l) , log(p_h))
            logits = torch.Tensor([torch.log(p_l)+ self.eps, torch.log(p_h) + self.eps]).view(1,2).cuda()
            
            one_hot = gumbel_softmax(logits, getattr(FLAGS, 'temperature',1.0))
            p_l = one_hot[0,0]
            p_h = one_hot[0,1]

        # p_h.view() for batch size * self.quant(weight, torch.ceil(lamda_w), 0( align dim ), zero_point, weight_quentscheme)
        # zero_point 0.5 인 이유는 음수 [ 0, 0.5 ], 양수 [ 0.5, 1 ]로 맞추려고
        #weight = p_h.view(-1,1,1,1) * self.quant(weight, torch.ceil(lamda_w), 0, 0.5, weight_quant_scheme) \
                # low value
          #  + p_l.view(-1,1,1,1) * self.quant(weight, torch.floor(lamda_w), 0, 0.5, weight_quant_scheme)
        # 음수 양수 조절 ( 0 ~ 1 ) -> ( 0 ~ 2 ) -> ( -1 ~ 1 )
        weight = (importance * weight_bits_list.sum())
        weight.mul_(2.0)
        weight.sub_(1.0)
        
        # SAT 같은데...
        if getattr(FLAGS, 'rescale_conv', False):
            rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
            if rescale_type == 'stddev':
                weight_scale = torch.std(self.weight.detach())
            elif rescale_type == 'constant':
                weight_scale = 1.0 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1]) ** 0.5
            else:
                raise NotImplementedError
            weight_scale /= torch.std(weight.detach())
            weight.mul_(weight_scale)

        ######-------------------------------Activation Quant--------------------------------######
        
        # weight_only일 때 안해
        if self.weight_only:
            input_val = input
        else:
            # double side면 Quant면 clamping 양쪽으로해 - value도 고려해줌
            if self.double_side:
                # 음수 clamping [ - alpha, input ]
                input_val = torch.where(input > -torch.abs(self.alpha), input, -torch.abs(self.alpha))
            else:
                # 아니면 relu 후 clamp [ 0, input ]
                input_val = torch.relu(input)
            # 양수 clamping 위에서 double side면 [ -alpha, alpha ], 아니면, [ 0, alpha ]
            input_val = torch.where(input_val < torch.abs(self.alpha), input_val, torch.abs(self.alpha))
            
            # alpha로 나눠 ( torch.abs(self.alpha) ) ( - 1 ~ 1 ) or ( 0 ~ 1 )
            input_val.div_(torch.abs(self.alpha))

            # double side면 0 ~ 1로  영점 맞춰줘야해
            if self.double_side:
                input_val.add_(1.0)
                input_val.div_(2.0)
            # weight랑 같은 방식으로 importance 구함
            p_a_l = 1 + torch.floor(lamda_a) - lamda_a
            p_a_h = 1 - p_a_l
            
            # gumbel 인 경우 
            if not getattr(FLAGS, 'hard_assignment', False) and getattr(FLAGS,'gumbel_softmax',False):
                # 같은 방식으로 p_a_l, p_a_h logit
                logits = torch.Tensor([torch.log(p_a_l)+ self.eps, torch.log(p_a_h) + self.eps]).view(1,2).cuda()
                one_hot = gumbel_softmax(logits, getattr(FLAGS, 'temperature',1.0))
                p_a_l = one_hot[0,0]
                p_a_h = one_hot[0,1]

            # input value 같은 방식으로
            input_val = p_a_h.view(-1,1,1) * self.quant(input_val, torch.ceil(lamda_a), 1, 0, act_quant_scheme) \
                + p_a_l.view(-1,1,1) * self.quant(input_val, torch.floor(lamda_a), 1, 0, act_quant_scheme)
            
            # double_side면 음수 양수 조절
            if self.double_side:
                input_val.mul_(2.0)
                input_val.sub_(1.0)
            # alpha값 곱해서 ( 0 ~ alpha ) rescaling 
            input_val.mul_(torch.abs(self.alpha))
        
        y = nn.functional.conv2d(
            input_val, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y
    
    @property
    def output_size(self):
        ih, iw = self.input_size
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if self.same_padding:
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        else:
            ph, pw = self.padding
            dh, dw = self.dilation
            oh, ow = out_shape(ih, ph, dh, kh, sh), out_shape(iw, pw, dw, kw, sw)
        return oh, ow
    
    @property
    def comp_cost_loss(self):
        oh, ow = self.output_size
        
        lamda_w = torch.clamp(self.lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
        if self.lamda_w_min is not None:
            lamda_w = torch.clamp(lamda_w, min=self.lamda_w_min)
        act_bits_list = getattr(FLAGS,'act_bits_list',FLAGS.bits_list)
        lamda_a = torch.clamp(self.lamda_a, min(act_bits_list), max(act_bits_list))
        if self.lamda_a_min is not None:
            lamda_a = torch.clamp(lamda_a, min=self.lamda_a_min)
        
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            lamda_w = lamda_w.view(self.groups, -1, 1)
        else:
            lamda_w = lamda_w.repeat(self.groups, self.out_channels // self.groups, 1)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            lamda_a = lamda_a.view(self.groups, 1, -1)
        else:
            lamda_a = lamda_a.repeat(self.groups, 1, self.in_channels // self.groups)
        
        bw_l = lamda_w.floor()
        bw_h = 1 + bw_l
        ba_l = lamda_a.floor()
        ba_h = 1 + ba_l
        
        cc_  = self.kernel_size[0] * self.kernel_size[1] * oh * ow * 1e-9
        cc_wh_ah = cc_ * ((lamda_w - bw_l) * (lamda_a - ba_l) * bw_h * ba_h).sum()
        cc_wh_al = cc_ * ((lamda_w - bw_l) * (ba_h - lamda_a) * bw_h * ba_l).sum()
        cc_wl_ah = cc_ * ((bw_h - lamda_w) * (lamda_a - ba_l) * bw_l * ba_h).sum()
        cc_wl_al = cc_ * ((bw_h - lamda_w) * (ba_h - lamda_a) * bw_l * ba_l).sum()
        
        loss =  cc_wh_ah + cc_wh_al + cc_wl_ah + cc_wl_al
        return loss

    @property
    def model_size_loss(self):
        lamda_w = torch.clamp(self.lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
        if self.lamda_w_min is not None:
            lamda_w = torch.clamp(lamda_w, min=self.lamda_w_min)
        
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            pass
        else:
            lamda_w = lamda_w.repeat(self.out_channels)

        bw_l = lamda_w.floor()
        bw_h = 1 + bw_l
        
        s_ = self.in_channels * self.kernel_size[0] * self.kernel_size[1] // self.groups / 8e6
        s_wh = s_ * ((lamda_w - bw_l) * bw_h).sum()
        s_wl = s_ * ((bw_h - lamda_w) * bw_l).sum()
        
        loss = s_wh + s_wl
        if self.bias is not None:
            loss += self.out_channels * 4e-6
        return loss

    def bit_discretizing(self):
        self.lamda_w = torch.nn.Parameter(torch.round(self.lamda_w + FLAGS.hard_offset))
        self.lamda_a = torch.nn.Parameter(torch.round(self.lamda_a + FLAGS.hard_offset))
        return 0
      
class QuantizableLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, lamda_w_min=None, lamda_a_min=None, weight_only=False):
        super(QuantizableLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.lamda_w_min = lamda_w_min
        self.lamda_a_min = lamda_a_min
        self.weight_only = weight_only or getattr(FLAGS, 'weight_only', False)
        self.quant = Quantize_k.apply
        self.alpha = nn.Parameter(torch.tensor(10.0))
        init_bit = getattr(FLAGS, 'init_bit', 7.5)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            self.lamda_w = nn.Parameter(torch.ones(self.out_features) * init_bit)
        else:
            self.lamda_w = nn.Parameter(torch.tensor(init_bit))
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            self.lamda_a = nn.Parameter(torch.ones(self.in_features) * init_bit)
        else:
            self.lamda_a = nn.Parameter(torch.tensor(init_bit))
        self.eps = 0.00001

    def forward(self, input):
        lamda_w = self.lamda_w
        lamda_a = self.lamda_a
        if getattr(FLAGS, 'hard_assignment',False):
            lamda_w = torch.round(lamda_w+FLAGS.hard_offset).detach()
            lamda_a = torch.round(lamda_a+FLAGS.hard_offset).detach()
        lamda_w = torch.clamp(lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
        if self.lamda_w_min is not None:
            lamda_w = torch.clamp(lamda_w, min=self.lamda_w_min)
        act_bits_list = getattr(FLAGS,'act_bits_list',FLAGS.bits_list)
        lamda_a = torch.clamp(self.lamda_a, min(act_bits_list), max(act_bits_list))
        if self.lamda_a_min is not None:
            lamda_a = torch.clamp(lamda_a, min=self.lamda_a_min)
        
        weight_quant_scheme = getattr(FLAGS, 'weight_quant_scheme', 'modified')
        act_quant_scheme = getattr(FLAGS, 'act_quant_scheme', 'original')

        weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
        weight.add_(1.0)
        weight.div_(2.0)
        p_l = 1 + torch.floor(lamda_w) - lamda_w
        p_h = 1 - p_l
        if not getattr(FLAGS, 'hard_assignment', False) and getattr(FLAGS,'gumbel_softmax',False):
            logits = torch.Tensor([torch.log(p_l)+ self.eps, torch.log(p_h) + self.eps]).view(1,2).cuda()
            one_hot = gumbel_softmax(logits, getattr(FLAGS, 'temperature',1.0))
            p_l = one_hot[0,0]
            p_h = one_hot[0,1]
        weight = p_h.view(-1,1) * self.quant(weight, torch.ceil(lamda_w), 0, 0.5, weight_quant_scheme) \
            + p_l.view(-1,1) * self.quant(weight, torch.floor(lamda_w), 0, 0.5, weight_quant_scheme)
        weight.mul_(2.0)
        weight.sub_(1.0)
        if getattr(FLAGS, 'rescale', True):
            rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
            if rescale_type == 'stddev':
                weight_scale = torch.std(self.weight.detach())
            elif rescale_type == 'constant':
                weight_scale = 1.0 / (self.out_features) ** 0.5
            else:
                raise NotImplementedError
            weight_scale /= torch.std(weight.detach())
            if self.training:
                weight.mul_(weight_scale)
        if self.bias is not None:
            bias = self.bias
            if getattr(FLAGS, 'rescale', True) and not self.training:
                bias = bias / weight_scale
        else:
            bias = self.bias

        if self.weight_only:
            input_val = input
        else:
            input_val = torch.where(input < torch.abs(self.alpha), input, torch.abs(self.alpha))
            input_val.div_(torch.abs(self.alpha))
            p_a_l = 1 + torch.floor(lamda_a) - lamda_a
            p_a_h = 1 - p_a_l
            if not getattr(FLAGS, 'hard_assignment', False) and getattr(FLAGS,'gumbel_softmax',False):
                logits = torch.Tensor([torch.log(p_a_l)+ self.eps, torch.log(p_a_h) + self.eps]).view(1,2).cuda()
                one_hot = gumbel_softmax(logits, getattr(FLAGS, 'temperature',1.0))
                p_a_l = one_hot[0,0]
                p_a_h = one_hot[0,1]
            input_val = p_a_h * self.quant(input_val, torch.ceil(lamda_a), 1, 0, act_quant_scheme) \
                + p_a_l * self.quant(input_val, torch.floor(lamda_a), 1, 0, act_quant_scheme)
            input_val.mul_(torch.abs(self.alpha))
        return nn.functional.linear(input_val, weight, bias)
    
    @property
    def comp_cost_loss(self):
        lamda_w = torch.clamp(self.lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
        if self.lamda_w_min is not None:
            lamda_w = torch.clamp(lamda_w, min=self.lamda_w_min)
        act_bits_list = getattr(FLAGS,'act_bits_list',FLAGS.bits_list)
        lamda_a = torch.clamp(self.lamda_a, min(act_bits_list), max(act_bits_list))
        if self.lamda_a_min is not None:
            lamda_a = torch.clamp(lamda_a, min=self.lamda_a_min)
        
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            lamda_w = lamda_w.view(-1, 1)
        else:
            lamda_w = lamda_w.repeat(self.out_features, 1)
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_activation', False):
            lamda_a = lamda_a.view(1, -1)
        else:
            lamda_a = lamda_a.repeat(1, self.in_features)

        bw_l = lamda_w.floor()
        bw_h = 1 + bw_l
        ba_l = lamda_a.floor()
        ba_h = 1 + ba_l
        
        cc_ = 1e-9
        cc_wh_ah = cc_ * ((lamda_w - bw_l) * (lamda_a - ba_l) * bw_h * ba_h).sum()
        cc_wh_al = cc_ * ((lamda_w - bw_l) * (ba_h - lamda_a) * bw_h * ba_l).sum()
        cc_wl_ah = cc_ * ((bw_h - lamda_w) * (lamda_a - ba_l) * bw_l * ba_h).sum()
        cc_wl_al = cc_ * ((bw_h - lamda_w) * (ba_h - lamda_a) * bw_l * ba_l).sum()
        
        loss = cc_wh_ah + cc_wh_al + cc_wl_ah + cc_wl_al
        return loss

    @property
    def model_size_loss(self):
        lamda_w = torch.clamp(self.lamda_w, min(FLAGS.bits_list), max(FLAGS.bits_list))
        if self.lamda_w_min is not None:
            lamda_w = torch.clamp(lamda_w, min=self.lamda_w_min)
        
        if getattr(FLAGS, 'per_channel', False) or getattr(FLAGS, 'per_channel_weight', False):
            pass
        else:
            lamda_w = lamda_w.repeat(self.out_features)

        bw_l = lamda_w.floor()
        bw_h = 1 + bw_l
        
        s_ = self.in_features / 8e6
        s_wh = s_ * ((lamda_w - bw_l) * bw_h).sum()
        s_wl = s_ * ((bw_h - lamda_w) * bw_l).sum()
        
        loss = s_wh + s_wl
        if self.bias is not None:
            loss += self.out_features * 4e-6
        return loss

    def bit_discretizing(self):
        self.lamda_w = torch.nn.Parameter(torch.round(self.lamda_w + FLAGS.hard_offset))
        self.lamda_a = torch.nn.Parameter(torch.round(self.lamda_a + FLAGS.hard_offset))
        return 0

class MaxPool2d(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0,
                 dilation=1, input_size=(0, 0)):
        super(MaxPool2d, self).__init__(kernel_size, stride, padding,
                                        dilation)
        self.input_size = input_size
    
    @property
    def output_size(self):
        ih, iw = self.input_size
        kh, kw = _pair(self.kernel_size)
        sh, sw = _pair(self.stride)
        ph, pw = _pair(self.padding)
        dh, dw = _pair(self.dilation)
        oh, ow = out_shape(ih, ph, dh, kh, sh), out_shape(iw, pw, dw, kw, sw)
        return oh, ow
