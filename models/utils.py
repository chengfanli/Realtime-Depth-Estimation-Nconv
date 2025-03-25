from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict
import torch.distributed as dist
#from einops.layers.torch import Rearrange
from timm.models.layers import DropPath
from torch.cuda.amp import custom_fwd, custom_bwd
import functools
from datetime import datetime, timedelta

class Basic2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1, padding_mode='zeros',
                 act=nn.ReLU, stride=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=False, padding_mode=padding_mode)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=True, padding_mode=padding_mode)
        self.conv = nn.Sequential(OrderedDict([('conv', conv)]))
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', act())

    def forward(self, x):
        out = self.conv(x)
        return out

class Coef(nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels=3, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=1, padding=padding, bias=True)

    def forward(self, x):
        feat = self.conv(x)
        XF, XB, XW = torch.split(feat, [1, 1, 1], dim=1)
        return XF, XB, XW

class Prop(nn.Module):
    """
    """

    def __init__(self, Cfi, Cfp=3, Cfo=2, act=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        """
        """
        self.dist = lambda x: (x * x).sum(1)
        Ct = Cfo + Cfi + Cfi + Cfp
        self.convXF = nn.Sequential(
            Basic2d(in_channels=Ct, out_channels=128, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=128, out_channels=128, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=128, out_channels=128, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=128, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=Cfi, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
        )
        self.convXL = nn.Sequential(
            Basic2d(in_channels=Cfi, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=64, norm_layer=norm_layer, act=act, kernel_size=1,
                    padding=0),
            Basic2d(in_channels=64, out_channels=Cfi, norm_layer=norm_layer, act=nn.Identity, kernel_size=1,
                    padding=0),
        )
        self.act = act()
        self.coef = Coef(Cfi, 3)

    def forward(self, If, Pf, Ofnum, args):
        """
        """
        # print(If.shape)
        num = args.shape[-2]
        B, Cfi, H, W = If.shape
        N = H * W
        B, Cfp, M = Pf.shape
        If = If.view(B, Cfi, 1, N)
        Pf = Pf.view(B, Cfp, 1, M)
        Ifnum = If.expand(B, Cfi, num, N)  ## Ifnum is BxCfixnumxN
        IPfnum = torch.gather(
            input=If.expand(B, Cfi, num, N),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfi, num, N))  ## IPfnum is BxCfixnumxN
        Pfnum = torch.gather(
            input=Pf.expand(B, Cfp, num, M),
            dim=-1,
            index=args.view(B, 1, num, N).expand(B, Cfp, num, N))  ## Pfnum is BxCfpxnumxN
        X = torch.cat([Ifnum, IPfnum, Pfnum, Ofnum], dim=1)
        XF = self.convXF(X)
        XF = self.act(XF + self.convXL(XF))
        Alpha, Beta, Omega = self.coef(XF)
        Omega = torch.softmax(Omega, dim=2)
        dout = torch.sum(((Alpha + 1) * Pfnum[:, -1:] + Beta) * Omega, dim=2, keepdim=True)
        return dout.view(B, 1, H, W)