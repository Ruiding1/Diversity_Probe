
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import models
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from network.wideresnet import WideResNet
from torchvision import datasets
from network import mnist_net
import os
import click
import time
import numpy as np
from torchvision.transforms import ToPILImage
from torch.autograd import Variable
# from network.cifar10_net import wrn
import data_loader
from network import wideresnet
# import data_loader
from data import data_helper
import argparse
import pandas as pd

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]



def evaluate(net, teloader):
    correct, count = 0, 0
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(teloader):
        with torch.no_grad():
            x1 = x1.cuda()
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    # 计算评价指标
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    return acc

def evaluate_digit_save(gpu, modelpath, svpath, channels=3, batchsize=128):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    # 加载模型
    if channels == 3:
        cls_net = mnist_net.ConvNet().cuda()
    elif channels == 1:
        cls_net = mnist_net.ConvNet(imdim=channels).cuda()

    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight)
    cls_net.eval()
    # 测试
    str2fun = {
        'mnist': data_loader.load_mnist,
        'svhn': data_loader.load_svhn,
        'mnist_m': data_loader.load_mnist_m,
        'syndigit': data_loader.load_syndigit,
        'usps': data_loader.load_usps,
        }
    columns = ['mnist', 'svhn', 'mnist_m', 'syndigit', 'usps']
    rst = []
    for data in columns:
        teset = str2fun[data]('test', channels=channels)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8)
        teacc = evaluate(cls_net, teloader)
        rst.append(teacc)
    ave = sum(rst[1:5]) / (len(rst) - 1)
    columns.append('ave')
    rst.append(ave)
    df = pd.DataFrame([rst], columns=columns)
    print(df)
    print(f'ave +{ave}')
    if svpath is not None:
        df.to_csv(svpath, index=False, mode='a+', header=False)
    return rst


def evaluate_cifar10(gpu, modelpath, svpath, c_path='./data/CIFAR-10-C'):
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    cls_net = WideResNet(depth=16, num_classes=10, widen_factor=4).cuda()

    preprocess = transforms.Compose(
        [
         transforms.Resize([32,32]),
         transforms.ToTensor(),
         transforms.Normalize([0.5] * 3, [0.5] * 3)])

    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['state'])
    cls_net.eval()

    test_data = datasets.CIFAR10('data', train=False, transform=preprocess, download=True)
    rst = [[]]
    for count, corruption in enumerate(CORRUPTIONS):
        datas = np.load(os.path.join(c_path, corruption + '.npy'))
        test_data.targets = torch.LongTensor(np.load(os.path.join(c_path, 'labels.npy')))[:10000 - 1]
        i = 4
        data = datas[i * 10000:(i + 1) * 10000 - 1]
        test_data.data = data
        teloader = DataLoader(test_data, batch_size=128, num_workers=8)
        teacc = evaluate_cifar10(cls_net, teloader)
        rst[count].append(teacc)
        rst.append([])

    rst.pop()
    rst = np.array(rst)
    result = {}

    result['noise'] = np.mean(rst[:3], 0)
    result['blur'] = np.mean(rst[3:7], 0)
    result['weather'] = np.mean(rst[7:10], 0)
    result['digital'] = np.mean(rst[10:], 0)
    for key, value in result.items():
        print("{0}: {1} ".format(key, value))
    Avg = (result['noise']  + result['blur'] + result['weather']  + result['digital'] ) /4.

    avg = Avg
    print(f'avg: {avg}')


def evaluate_P(model, test_loader, device='cuda'):
    model.eval()
    class_correct = 0
    total = 0
    for it, ((data, nouse, class_l), _, _) in enumerate(test_loader):
        data, nouse, class_l = data.to(device), nouse.to(device), class_l.to(device)
        z = model(data)
        _, cls_pred = z.max(dim=1)
        total += data.size(0)
        class_correct += torch.sum(cls_pred == class_l.data)

    acc = 100 * class_correct / total
    return acc

def evaluate_ACS_L2D_save(gpu, modelpath, data_loader, svpath, device='cuda'):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    cls_net = caffenet_L2D(7).cuda()
    saved_weight = torch.load(modelpath, map_location='cuda')
    cls_net.load_state_dict(saved_weight)
    cls_net.eval()

    total = len(data_loader)
    avg_acc=0
    rst = []
    for loader in data_loader:
        teacc = evaluate_P(cls_net, loader)
        avg_acc += teacc
        rst.append(teacc.item())

    acc = (avg_acc / total).item()

    columns_PACS = ['A', 'C', 'S', 'ave']
    rst.append(acc)
    df = pd.DataFrame([rst], columns=columns_PACS)
    print(df)
    print(f' ave +{acc}')

    if svpath is not None:
        df.to_csv(svpath, index=False, mode='a+', header=False)

    return rst


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  OK  ---")
    else:
        print("---  There is this folder!  ---")


def saveModel(model, epoch, test_acc, train_loss, path):
    torch.save(model, path + '/model.pkl')

    epoch_save = np.array(epoch)
    np.save(path + '/learning_rate.npy', epoch_save)
    test_acc = np.array(test_acc)
    np.save(path + '/test_acc.npy', test_acc)
    train_loss = np.array(train_loss)
    np.save(path + '/train_loss.npy', train_loss)


def smooth_step(a, b, c, d, x):
    level_s = 0.01
    level_m = 0.1
    level_n = 0.01
    level_r = 0.005
    if x <= a:
        return level_s
    if a < x <= b:
        return (((x - a) / (b - a)) * (level_m - level_s) + level_s)
    if b < x <= c:
        return level_m
    if c < x <= d:
        return level_n
    if d < x:
        return level_r


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

