import numpy as np
import os
import time
import torch.backends.cudnn as cudnn
import torch
import torchvision.utils as vutils
import sys
import shutil
import torch.nn as nn
import math
from torch.nn import functional as F
from torchvision import transforms
from torchvision import datasets
from network.wideresnet import WideResNet
from torch.utils.data import DataLoader
from network import mnist_net, generator
import pandas as pd
import random
from PIL import Image

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def log(txt, path):
    print(txt)
    with open(path, 'a') as f:
        f.write(txt + '\n')
        f.flush()
        f.close()

def to_rad(deg):
    return deg/180*math.pi

def random_grid(size,bs):
    min_s, max_s = 0.8, 1.5
    min_r, max_r = -10, 10
    min_t, max_t = -0.5, 0.5
    rand_s = torch.FloatTensor(bs, 1).uniform_(min_s, max_s)
    rand_r = torch.FloatTensor(bs, 1).uniform_(min_r, max_r)
    rand_tx = torch.FloatTensor(bs, 1).uniform_(min_t, max_t)
    rand_ty = torch.FloatTensor(bs, 1).uniform_(min_t, max_t)
    theta = [[[rand_s[i]*torch.cos(to_rad(rand_r[i])), -rand_s[i]*torch.sin(to_rad(rand_r[i])), rand_tx[i]],
              [rand_s[i]*torch.sin(to_rad(rand_r[i])), rand_s[i]*torch.cos(to_rad(rand_r[i])), rand_ty[i]]]
              for i in range(bs)]
    theta = torch.tensor(theta)

    grid = F.affine_grid(theta, size, align_corners=True).cuda()
    return grid

class CrossEntropy():
    def __init__(self):
        self.code_loss = nn.CrossEntropyLoss()

    def __call__(self, prediction, label):

        if label.max(dim=1)[0].min() == 1:
            return self.code_loss(prediction, torch.nonzero(label.long())[:, 1])
        else:
            log_prediction = torch.log_softmax(prediction, dim=1)
            return (- log_prediction * label).sum(dim=1).mean(dim=0)



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, alpha):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_reg = cos(negative - anchor, positive - anchor).sum(0)
        losses = F.relu(distance_positive - distance_negative + self.margin - self.alpha * cos_reg)  # 2e-2

        return losses.mean()


def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss = 0

    if ver == 1:
        for i in range(batch_size):
            s1, s2 = i, (i + 1) % batch_size
            t1, t2 = s1 + batch_size, s2 + batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        loss = loss.abs_() / float(batch_size)
    elif ver == 2:
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
    else:
        raise ValueError('ver == 1 or 2')

    return loss

def conditional_mmd_rbf(source, target, label, num_class, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    for i in range(num_class):
        source_i = source[label==i]
        target_i = target[label==i]
        loss += mmd_rbf(source_i, target_i)
    return loss / num_class

def domain_mmd_rbf(source, target, num_domain, d_label, kernel_mul=2.0, kernel_num=5, fix_sigma=None, ver=2):
    loss = 0
    loss_overall = mmd_rbf(source, target)
    for i in range(num_domain):
        source_i = source[d_label == i]
        target_i = target[d_label == i]
        loss += mmd_rbf(source_i, target_i)
    return loss_overall - loss / num_domain

def domain_conditional_mmd_rbf(source, target, num_domain, d_label, num_class, c_label):
    loss = 0
    for i in range(num_class):
        source_i = source[c_label == i]
        target_i = target[c_label == i]
        d_label_i = d_label[c_label == i]
        loss_c = mmd_rbf(source_i, target_i)
        loss_d = 0
        for j in range(num_domain):
            source_ij = source_i[d_label_i == j]
            target_ij = target_i[d_label_i == j]
            loss_d += mmd_rbf(source_ij, target_ij)
        loss += loss_c - loss_d / num_domain

    return loss / num_class

def reparametrize(mu, logvar, factor=0.2):
    std = logvar.div(2).exp()
    eps = std.data.new(std.size()).normal_()
    return mu + factor*std*eps


def loglikeli(mu, logvar, y_samples):
    return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean()

def club(mu, logvar, y_samples):

    sample_size = y_samples.shape[0]
    # random_index = torch.randint(sample_size, (sample_size,)).long()
    random_index = torch.randperm(sample_size).long()

    positive = - (mu - y_samples) ** 2 / logvar.exp()
    negative = - (mu - y_samples[random_index]) ** 2 / logvar.exp()
    upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
    return upper_bound / 2.


def save_img_results(imgs_tcpu, fake_imgs, count, epoch, image_dir, flage, nrow=8):

    num =  8888

    if imgs_tcpu is not None:
        if flage == 0:
            real_img = imgs_tcpu[:][0:num]
            vutils.save_image(
                real_img, '%s/count_%09d_fake_bg%d.png' % (image_dir, count, epoch),
                scale_each=True, normalize=True, nrow=nrow)
            real_img_set = vutils.make_grid(real_img).numpy()
            real_img_set = np.transpose(real_img_set, (1, 2, 0))
            real_img_set = real_img_set * 255
            real_img_set = real_img_set.astype(np.uint8)

        elif flage == 1:
            real_img = imgs_tcpu[:][0:num]
            vutils.save_image(
                real_img, '%s/count_%09d_fake_fore%d.png' % (image_dir, count, epoch),
                scale_each=True, normalize=True, nrow=nrow)
            real_img_set = vutils.make_grid(real_img).numpy()
            real_img_set = np.transpose(real_img_set, (1, 2, 0))
            real_img_set = real_img_set * 255
            real_img_set = real_img_set.astype(np.uint8)
        elif flage == 2:
            real_img = imgs_tcpu[:][0:num]
            vutils.save_image(
                real_img, '%s/count_%09d_fake_images%d.png' % (image_dir, count, epoch),
                scale_each=True, normalize=True, nrow=nrow)
            real_img_set = vutils.make_grid(real_img).numpy()
            real_img_set = np.transpose(real_img_set, (1, 2, 0))
            real_img_set = real_img_set * 255
            real_img_set = real_img_set.astype(np.uint8)

    if fake_imgs is not None:

        if flage == 0:
            fake_img = fake_imgs
            vutils.save_image(
                fake_img.data, '%s/count_%09d_real_samples%d.png' %
                (image_dir, count, epoch), scale_each=True, normalize=True, nrow=nrow)

            fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

            fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
            fake_img_set = fake_img_set * 255
            fake_img_set = fake_img_set.astype(np.uint8)
        elif flage ==1:
            fake_img = fake_imgs
            vutils.save_image(
                fake_img.data, '%s/count_%09d_real_mask%d.png' %
                (image_dir, count, epoch), scale_each=True, normalize=True, nrow=nrow)

            fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

            fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
            fake_img_set = fake_img_set * 255
            fake_img_set = fake_img_set.astype(np.uint8)

def make_output_dir(dir):
    root_path = './result'
    args = dir
    if len(args)==0:
        raise RuntimeError('output folder must be specified')
    new_output = args
    path = os.path.join(root_path, new_output)
    if os.path.exists(path):
        if len(args)==2 and args =='-f':
            print('WARNING: experiment directory exists, it has been erased and recreated')
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print('WARNING: experiment directory exists, it will be erased and recreated in 3s')
            time.sleep(3)
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)
    return path

def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off

