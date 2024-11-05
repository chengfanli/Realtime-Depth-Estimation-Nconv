import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import *
from torch.cuda.amp import custom_fwd, custom_bwd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import cv2
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import torch.nn as nn
from models.step1 import SETP1_NCONV


class SETP2_BP_TRAIN(nn.Module):

    def __init__(self, step1_checkpoint_name):
        super().__init__() 

        self.step1 = SETP1_NCONV()

        checkpoint = torch.load("./checkpoints/{}.pth.tar".format(step1_checkpoint_name))
        state_dict = checkpoint["state_dict"]

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        self.step1.load_state_dict(new_state_dict, strict=False)
        
        # Disable Training for the unguided module
        for p in self.step1.parameters():            
            p.requires_grad=False

        self.rgb_encoder0 = RGBEncoder(3, 32, 1)
        self.rgb_encoder1 = RGBEncoder(32, 32, 2)
        self.rgb_encoder2 = RGBEncoder(32, 64, 2)
        self.rgb_encoder3 = RGBEncoder(64, 64, 2)
        self.rgb_encoder4 = RGBEncoder(64, 64, 2)

        self.rgb_encoder0 = RGBEncoder(3, 32, 1)
        self.rgb_encoder1 = RGBEncoder(32, 64, 2)
        self.rgb_encoder2 = RGBEncoder(64, 64, 2)
        self.rgb_encoder3 = RGBEncoder(64, 64, 2)
        # self.rgb_encoder4 = RGBEncoder(64, 64, 2)

        self.fuse0 = FusionResolution0(64,8)
        self.fuse1 = FusionResolutionBlock(64, 64, 4)
        self.fuse2 = FusionResolutionBlock(64, 32, 2)
        self.fuse3 = FusionResolutionBlock(32, 32, 1)
        # self.fuse4 = FusionResolutionBlock(32, 32, 1)
            
    def forward(self, rgb0, depth0, rgb1, depth1): 
        
        sparse = self.step1(depth0)
        sparse = torch.cat((sparse, sparse), dim=0) # Hack
        rgb = torch.cat((rgb0, rgb1), dim=0)

        rgb0 = self.rgb_encoder0(rgb)
        rgb1 = self.rgb_encoder1(rgb0) # 480 -> 240
        rgb2 = self.rgb_encoder2(rgb1) # 240 -> 120
        rgb3 = self.rgb_encoder3(rgb2) # 120 -> 60
        # rgb4 = self.rgb_encoder4(rgb3) # 60 -> 30

        out_fusion0, out_depth0 = self.fuse0(rgb3, sparse)
        out_fusion1, out_depth1 = self.fuse1(rgb2, sparse, out_fusion0, out_depth0)
        out_fusion2, out_depth2 = self.fuse2(rgb1, sparse, out_fusion1, out_depth1)
        out_fusion3, out_depth3 = self.fuse3(rgb0, sparse, out_fusion2, out_depth2)
        # out_fusion4, out_depth4 = self.fuse4(rgb0, sparse, out_fusion3, out_depth3)

        return [out_depth0[0:1], out_depth1[0:1], out_depth2[0:1], out_depth3[0:1]], [out_depth0[1:2], out_depth1[1:2], out_depth2[1:2], out_depth3[1:2]]


class SETP2_BP_EXPORT(nn.Module):

    def __init__(self):
        super().__init__() 

        self.step1 = SETP1_NCONV()
        self.step1 = self.step1.compile()

        # self.rgb_encoder0 = RGBEncoder(3, 32, 1)
        # self.rgb_encoder1 = RGBEncoder(32, 32, 2)
        # self.rgb_encoder2 = RGBEncoder(32, 64, 2)
        # self.rgb_encoder3 = RGBEncoder(64, 64, 2)
        # self.rgb_encoder4 = RGBEncoder(64, 64, 2)

        self.rgb_encoder0 = RGBEncoder(3, 32, 1)
        self.rgb_encoder1 = RGBEncoder(32, 64, 2)
        self.rgb_encoder2 = RGBEncoder(64, 64, 2)
        self.rgb_encoder3 = RGBEncoder(64, 64, 2)
        # self.rgb_encoder4 = RGBEncoder(64, 64, 2)

        self.fuse0 = FusionResolution0(64,8)
        self.fuse1 = FusionResolutionBlock(64, 64, 4)
        self.fuse2 = FusionResolutionBlock(64, 32, 2)
        self.fuse3 = FusionResolutionBlock(32, 32, 1)
        # self.fuse4 = FusionResolutionBlock(32, 32, 1)
            
    def forward(self, rgb0, depth0, rgb1, depth1): 
        
        sparse = self.step1(depth0, depth1)
        rgb = torch.cat((rgb0, rgb1), dim=0)

        rgb0 = self.rgb_encoder0(rgb)
        rgb1 = self.rgb_encoder1(rgb0) # 480 -> 240
        rgb2 = self.rgb_encoder2(rgb1) # 240 -> 120
        rgb3 = self.rgb_encoder3(rgb2) # 120 -> 60
        # rgb4 = self.rgb_encoder4(rgb3) # 60 -> 30

        out_fusion0, out_depth0 = self.fuse0(rgb3, sparse)
        out_fusion1, out_depth1 = self.fuse1(rgb2, sparse, out_fusion0, out_depth0)
        out_fusion2, out_depth2 = self.fuse2(rgb1, sparse, out_fusion1, out_depth1)
        out_fusion3, out_depth3 = self.fuse3(rgb0, sparse, out_fusion2, out_depth2)
        # out_fusion4, out_depth4 = self.fuse4(rgb0, sparse, out_fusion3, out_depth3)

        out_depth3[:, :, :45, :] = 0
        out_depth3[:, :, -45:, :] = 0
        out_depth3[:, :, :, :20] = 0
        
        return out_depth3[0:1], out_depth3[1:2]



def Conv1x1(in_planes, out_planes, stride, bias=False, groups=1, dilation=1, padding_mode='zeros'):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

class RGBEncoder(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(RGBEncoder, self).__init__()
        self.stride = stride
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.downsample = nn.Sequential(
            Conv1x1(in_channel, out_channel, self.stride)
        )
    
    def forward(self, x):
        identity = x
        out = self.encoder(x)
        out = out + self.downsample(x)

        return out

def Conv3x3(in_planes, stride=1, groups=1, dilation=1, padding_mode='zeros', bias=False):
    return nn.Conv2d(in_planes, 1, kernel_size=3, stride=stride,
                     padding=dilation, padding_mode=padding_mode, groups=groups, bias=bias, dilation=dilation)

class UpCat(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, kernel_size=3, padding=1,
                 padding_mode='zeros', act=nn.ReLU):
        super().__init__()
        self.upf = Basic2dTrans(in_channels + 1, out_channels, norm_layer=norm_layer, act=act)
        self.conv = Basic2d(out_channels * 2, out_channels,
                            norm_layer=norm_layer, kernel_size=kernel_size,
                            padding=padding, padding_mode=padding_mode, act=act)
                            

    def forward(self, y, x, d):
        """
        x is
        """
        fout = self.upf(torch.cat([x, d], dim=1))
        fout = self.conv(torch.cat([fout, y], dim=1))
        return fout

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

class Basic2dTrans(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None, act=nn.ReLU):
        super().__init__()
        if norm_layer is None:
            bias = True
            norm_layer = nn.Identity
        else:
            bias = False
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4,
                                       stride=2, padding=1, bias=bias)
        self.bn = norm_layer(out_channels)
        self.relu = act()

    def forward(self, x):
        out = self.conv(x.contiguous())
        out = self.bn(out)
        out = self.relu(out)
        return out

class NewFusionBlock(nn.Module):
    def __init__(self, rgb_channels, out_channels):
        super(NewFusionBlock, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.depth_conv = ConvBlock(1, rgb_channels)

        self.fuse_conv1 = ConvBlock(rgb_channels * 2, rgb_channels)

        self.fuse_conv2 = ConvBlock(rgb_channels, out_channels)

        self.fuse_conv3 = ConvBlock(out_channels, out_channels)

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_conv(rgb)
        depth_feat = self.depth_conv(depth)

        fused = torch.cat((rgb_feat, depth_feat),1)
        fused = self.fuse_conv1(fused)
        fused = self.fuse_conv2(fused)
        fused = self.fuse_conv3(fused)
        return fused

class FusionResolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample_factor):
        super(FusionResolutionBlock, self).__init__()

        self.upcat = UpCat(in_channel, in_channel)

        self.fuse = NewFusionBlock(in_channel, out_channel)
        self.conv = Conv3x3(out_channel, 1)
        self.downsample_factor = downsample_factor
    
    def forward(self, rgb, depth, depth_last_step, fusion_festure):

        fout = self.upcat(rgb, fusion_festure, depth_last_step)

        depth = F.interpolate(depth, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=True)

        fout = self.fuse(fout, depth)
        res = self.conv(fout)
        dout = depth
        dout = dout + res

        return fout, dout


class FusionResolution0(nn.Module):
    def __init__(self, in_channel, downsample_factor):
        super(FusionResolution0, self).__init__()

        self.fuse = NewFusionBlock(in_channel, in_channel)
        self.conv = Conv3x3(in_channel, 1)
        self.downsample_factor = downsample_factor
    
    def forward(self, rgb, depth):
        depth = F.interpolate(depth, scale_factor=1/self.downsample_factor, mode='bilinear', align_corners=True)

        fout = self.fuse(rgb, depth)
        res = self.conv(fout)
        dout = depth
        dout = dout + res

        return fout, dout

class ConvBlockBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))



class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv1(x)
        return self.conv2(x)


class FusionBlock(nn.Module):
    def __init__(self, rgb_channels, sparse_channels, fusion_channels):
        super(FusionBlock, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.sparse_conv = ConvBlock(sparse_channels, sparse_channels)
        self.fuse_conv1 = ConvBlock(rgb_channels * 2, sparse_channels)

        
        self.fuse_conv2 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv3 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse, dense):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)

        fused = torch.cat((rgb_feat, dense),1)
        fused = self.fuse_conv1(fused)
        
        fused = self.fuse_conv2(fused + sparse_feat)
        fused = self.fuse_conv3(fused)
        return fused

class Fusion0(nn.Module):
    def __init__(self, rgb_channels, sparse_channels, fusion_channels):
        super(Fusion0, self).__init__()
        self.rgb_conv = ConvBlock(rgb_channels, rgb_channels)
        self.sparse_conv = ConvBlock(sparse_channels, sparse_channels)

        self.fuse_conv2 = ConvBlock(sparse_channels, fusion_channels)
        self.fuse_conv3 = ConvBlock(fusion_channels, fusion_channels)

    def forward(self, rgb, sparse):
        rgb_feat = self.rgb_conv(rgb)
        sparse_feat = self.sparse_conv(sparse)
        
        fused = self.fuse_conv2(rgb_feat + sparse_feat)
        fused = self.fuse_conv3(fused)
        return fused


# Non-negativity enforcement class        
class EnforcePos(object):
    def __init__(self, pos_fn, name):
        self.name = name
        self.pos_fn = pos_fn


    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(pos_fn, name)
        
        module.register_forward_pre_hook(fn)                    

        return fn

    def __call__(self, module, inputs):
       if module.training:
            weight = getattr(module, self.name)
            weight.data = self._pos(weight).data
       else:
            pass

    def _pos(self, p):
        pos_fn = self.pos_fn.lower()
        if pos_fn == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0],p_sz[1], -1)
            p = F.softmax(p, -1)
            return p.view(p_sz)
        elif pos_fn == 'exp':
            return torch.exp(p)
        elif pos_fn == 'softplus':
            return F.softplus(p, beta=10)
        elif pos_fn == 'sigmoid':
            return F.sigmoid(p)
        else:
            print('Undefined positive function!')
            return      

class GenKernel(nn.Module):
    def __init__(self, in_channels, pk, norm_layer=nn.BatchNorm2d, act=nn.ReLU, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.conv = nn.Sequential(
            Basic2d(in_channels, in_channels, norm_layer=norm_layer, act=act),
            Basic2d(in_channels, pk * pk - 1, norm_layer=norm_layer, act=nn.Identity),
        )

    def forward(self, fout):
        weight = self.conv(fout)
        weight_sum = torch.sum(weight.abs(), dim=1, keepdim=True)
        weight = torch.div(weight, weight_sum + self.eps)
        weight_mid = 1 - torch.sum(weight, dim=1, keepdim=True)
        weight_pre, weight_post = torch.split(weight, [weight.shape[1] // 2, weight.shape[1] // 2], dim=1)
        weight = torch.cat([weight_pre, weight_mid, weight_post], dim=1).contiguous()
        return weight[:, weight.shape[1] // 2, :, :]
