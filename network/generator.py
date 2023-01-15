from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def sameBlock_RELU_BN_5x5(in_planes, out_planes):
    block = nn.Sequential(conv5x5(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.ReLU() )
    return block

def sameBlock_RELU_BN_7x7(in_planes, out_planes):
    block = nn.Sequential(conv7x7(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.ReLU() )
    return block

def sameBlock_RELU_BN_1x1(in_planes, out_planes):
    block = nn.Sequential(conv1x1(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.ReLU() )
    return block

def sameBlock_RELU_BN_9x9(in_planes, out_planes):
    block = nn.Sequential(conv9x9(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.ReLU() )
    return block

def sameBlock_RELU_BN_3x3(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.ReLU() )
    return block

def conv7x7(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride= 1, padding= 7//2, bias=False)

def conv9x9(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=9, stride=1, padding=4, bias=False)

def conv5x5(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1, padding=5//2, bias=False)

def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


class GET_IMAGE_Tanh1x1(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv1x1(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)

class AdaIN2d_fc_z1z2(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc1 = nn.Linear(style_dim, num_features)
        self.fc2 = nn.Linear(style_dim, num_features)
    def forward(self, x, s1, s2 ):
        h1 = self.fc1(s1)
        h2 = self.fc2(s2)
        gamma = h1.view(h1.size(0), h1.size(1), 1, 1)
        beta = h2.view(h2.size(0), h2.size(1), 1, 1)
        return (1 + gamma) * self.norm(x) + beta

class AdaIN2d__Noise(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        self.noise = conv1x1(num_features, num_features)
    def forward(self, x, s, w):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        self.norm_x = self.norm(x) + w * self.noise(torch.randn_like(x))
        return (1 + gamma) * self.norm_x + beta


class ProbeModule_Digits(nn.Module):
    def __init__(self, n=3, w_noise =0.2, kernelsize=3, imdim=3, imsize=[192, 320]):

        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize
        self.w = w_noise
        self.conv1 = sameBlock_RELU_BN_5x5(3, n)
        self.adain2_fore = AdaIN2d__Noise(zdim, n)
        self.fore_style = GET_IMAGE_Tanh1x1(n)

    def forward(self, x):
        ''' x '''
        x = self.conv1(x)

        z1 = torch.randn(len(x), self.zdim).cuda()
        x_fore_style = self.adain2_fore(x, z1, self.w)
        fore_style = self.fore_style(x_fore_style)

        return fore_style

class ProbeModule_CIFAR10(nn.Module):
    def __init__(self, n=16, w_noise =0.2, kernelsize=3, imdim=3, imsize=[192, 320]):

        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize
        self.w = w_noise

        self.conv1 = sameBlock_RELU_BN_5x5(3, n)
        self.adain2_fore = AdaIN2d__Noise(zdim, n)
        self.fore_style = GET_IMAGE_Tanh1x1(n)

    def forward(self, x):
        x = self.conv1(x)

        z1 = torch.randn(len(x), self.zdim).cuda()
        x_fore_style = self.adain2_fore(x, z1, self.w)
        fore_style = self.fore_style(x_fore_style)

        return fore_style

class TraceModule_All(nn.Module):
    def __init__(self, n=16, kernelsize=3, imdim=3, imsize=[192, 320]):

        super().__init__()
        stride = (kernelsize-1)//2
        self.zdim = zdim = 10
        self.imdim = imdim
        self.imsize = imsize

        self.conv1 = nn.Conv2d(imdim, n, kernelsize, 1, stride)
        self.conv2 = nn.Conv2d(n, 2*n, kernelsize, 1, stride)
        self.adain2 = AdaIN2d_fc_z1z2(zdim, 2*n)
        self.conv3 = nn.Conv2d(2*n, 4*n, kernelsize, 1, stride)
        self.conv4 = nn.Conv2d(4*n, imdim, kernelsize, 1, stride)

    def forward(self, x, rand=True):
        ''' x '''
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        if rand:
            z1 = torch.randn(len(x), self.zdim).cuda()
            z2 = torch.randn(len(x), self.zdim).cuda()
            x = self.adain2(x, z1, z2)
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        return x


