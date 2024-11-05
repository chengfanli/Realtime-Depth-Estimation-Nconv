import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from .utils import *
import cv2
import torch
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal

class SETP1_NCONV(nn.Module):
    def __init__(self):
        super().__init__() 

        self.d_net = DNET(32)
               
            
    def forward(self, x0_d): 

        # Depth Network
        xout_d = self.d_net(x0_d)

        return xout_d


class DNET(nn.Module):
    def __init__(self, out_ch):
        super().__init__()
 
        pos_fn = "softplus"
        #pos_fn = None
        num_channels=8

        self.nconv1 = NConv2d(1, num_channels, (5,5), pos_fn, 'p', padding='same')
        self.nconv2 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding='same')

        self.nconv_down1 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding='same')
        self.nconv_down2 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding='same')
        self.nconv_down3 = NConv2d(num_channels, num_channels, (5,5), pos_fn, 'p', padding='same')

        self.nconv4 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding='same')
        self.nconv5 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding='same')
        self.nconv6 = NConv2d(2*num_channels, num_channels, (3,3), pos_fn, 'p', padding='same')

        self.nconv7 = NConv2d(num_channels, 1, (1,1), pos_fn, 'k', padding='same')

    def forward(self, S):

        c0 = (S > 0.01).float()
        x0 = S

        x1, c1 = self.nconv1(x0, c0)
        x1, c1 = self.nconv2(x1, c1)


        # Downsample 1
        ds = 2
        c1_ds = F.max_pool2d(c1, ds, ds)

        x1_ds = F.max_pool2d(x1, ds, ds)
        x2_ds, c2_ds = self.nconv_down1(x1_ds, c1_ds)                

        # Downsample 2
        c2_dss = F.max_pool2d(c2_ds, ds, ds)
        x2_dss = F.max_pool2d(x2_ds, ds, ds)
        x3_ds, c3_ds = self.nconv_down2(x2_dss, c2_dss)        

        # Downsample 3
        c3_dss = F.max_pool2d(c3_ds, ds, ds)
        x3_dss = F.max_pool2d(x3_ds, ds, ds)
        x4_ds, c4_ds = self.nconv_down3(x3_dss, c3_dss)                

        # Upsample 1
        x4 = F.interpolate(x4_ds, c3_ds.size()[2:], mode='nearest') 
        c4 = F.interpolate(c4_ds, c3_ds.size()[2:], mode='nearest')   
        x34_ds, c34_ds = self.nconv4(torch.cat((x3_ds,x4), 1),  torch.cat((c3_ds,c4), 1))       

        # Upsample 2
        x34 = F.interpolate(x34_ds, c2_ds.size()[2:], mode='nearest') 
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        x23_ds, c23_ds = self.nconv5(torch.cat((x2_ds,x34), 1), torch.cat((c2_ds,c34), 1))

        # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest') 
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest') 
        xout, cout = self.nconv6(torch.cat((x23,x1), 1), torch.cat((c23,c1), 1))

        xout, cout = self.nconv7(xout, cout)

        if xout.max() > 10000:
            # Things have gone bad...
            breakpoint()
            

        return xout


class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='k', stride=(1, 1), padding=(2, 2), dilation=(1, 1), groups=1, bias=True):
        
        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, output_padding=(0, 0), groups=groups, bias=bias, padding_mode='zeros')

        self.eps = 1e-3
        self.pos_fn = pos_fn
        self.init_method = init_method
        
        # Initialize weights and bias
        self.init_parameters()

        self.bnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()
        
        if self.pos_fn is not None :
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, data, conf):
        
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data*conf, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)
        nconv = nomin / (denom+self.eps)
        
        
        # Add biasf
        bias = self.bias
        sz = bias.size(0)
        bias = bias.view(1,sz,1,1)
        bias = bias.expand_as(nconv)
        nconv += bias

        #nconv = self.bnorm(nconv)
        #nconv = self.activation(nconv)
        
        # Propagate confidence
        cout = denom
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)
        
        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)
        s = torch.sum(k, dim=-1, keepdim=True)   

        #print(f"s = {s}") 

        cout = cout / s
        cout = cout.view(sz)
                
        return nconv, cout
    
    
    def init_parameters(self):
        # Init weights
        if self.init_method == 'x': # Xavier            
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k': # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'p': # Poisson
            mu=self.kernel_size[0]/2 
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x),1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w,0)
            w = torch.unsqueeze(w,1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)
            
        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels)+0.01)

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
        elif pos_fn == 'relu':
            return F.relu(p)
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
