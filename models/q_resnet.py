import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .quantizable_op_dorefa import q_Conv2d, q_Linear, ActQuant
from utils.config import FLAGS

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        assert stride in [1, 2]

        self.l1 = q_Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.l2 = q_Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)

        self.ActQuant = ActQuant()
        self.a_bitwidth = FLAGS.activation_bitwidth

        self.residual_connection = stride == 1 and in_channels == out_channels
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                q_Conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, bias = False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.l1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ActQuant(out)
        
        out = self.l2(out)
        out = self.bn2(out)

        if self.residual_connection:
            out += x
        else:
            out += self.shortcut(x)
        out = self.relu(out)
        out = self.ActQuant(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Bottleneck, self).__init__()
        assert stride in [ 1, 2 ]

        mid_channels = out_channels // 4
        self.l1 = q_Conv2d(in_channels, mid_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.l2 = q_Conv2d(mid_channels, mid_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.l3 = q_Conv2d(mid_channels, out_channels, kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)

        self.ActQuant = ActQuant()
        self.a_bitwidth = FLAGS.activation_bitwidth

        self.residual_connection = stride == 1 and in_channels == out_channels
        
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                q_Conv2d(in_channels, out_channels, kernel_size = 1, bias = False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.l1(x)))
        out = self.relu(self.bn2(self.l2(out)))
        out = self.bn3(self.l3(out))

        if self.residual_connection:
            out += x
        else:
            out += self.shortcut(x)
        out = self.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes = 1000):
        super(Model, self).__init__()

        if FLAGS.depth in [20, 56, 110]:
            block = BasicBlock
        elif FLAGS.depth in [50, 101, 152]:
            block = Bottleneck

        # head
        channels = 16
        self.l_head = q_Conv2d(in_channels = 3, out_channels = channels, kernel_size = 3, stride = 1, padding = 1,
                            bias = False)
        self.bn_head = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace = True)
        
        self.block_setting_dict = {
            # [stage 1, stage 2, stage 3, stage 4]
            20: [3, 3, 3],
            56: [9, 9, 9],
            110: [18, 18, 18]
       }
        self.block_setting = self.block_setting_dict[FLAGS.depth]

        feats = [16, 32, 64]

        # body
        
        for idx, n in enumerate(self.block_setting):
            out_channels = feats[idx] * block.expansion
            for i in range(n):
                if i == 0 and idx != 0:
                    layer = block(channels, out_channels, stride = 2)
                else:
                    layer = block(channels, out_channels, stride = 1)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = out_channels
        self.classifier = q_Linear(out_channels, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.a_bitwidth = FLAGS.activation_bitwidth
        self.ActQuant = ActQuant()
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.ActQuant(x)
        x = self.l_head(x)
        x = self.bn_head(x)
        x = self.relu(x)
        x = self.ActQuant(x)

        for idx, n in enumerate(self.block_setting):
            for i in range(n):        
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            else:
                pass



