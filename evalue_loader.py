'''
训练 base 模型
'''

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

HOME = os.environ['HOME']


def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", help="Source", nargs='+')
    parser.add_argument("--target", help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0., type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float,
                        help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.002, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=300, help="Number of epochs")
    parser.add_argument("--network", help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=0.9, type=float,
                        help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool,
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")
    parser.add_argument("--visualization", default=False, type=bool)
    parser.add_argument("--epochs_min", type=int, default=1,
                        help="")
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--ckpt", default="logs/model", type=str)
    #
    parser.add_argument("--alpha1", default=1, type=float)
    parser.add_argument("--alpha2", default=1, type=float)
    parser.add_argument("--beta", default=0.1, type=float)
    parser.add_argument("--lr_sc", default=0.005, type=float)
    parser.add_argument("--task", default='PACS', type=str)

    return parser.parse_args()


@click.command()
@click.option('--gpu', type=str, default='0', help='选择gpu')
@click.option('--data', type=str, default='PACS', help='数据集名称')
@click.option('--ntr', type=int, default=None, help='选择训练集前ntr个样本')
@click.option('--translate', type=float, default=None, help='随机平移数据增强')
@click.option('--autoaug', type=str, default=None, help='AA FastAA RA')
@click.option('--epochs', type=int, default=300)
@click.option('--nbatch', type=int, default=None, help='每个epoch中batch的数量')
@click.option('--batchsize', type=int, default=128, help='每个batch中样本的数量')
@click.option('--lr', type=float, default=0.1)
@click.option('--lr_scheduler', type=str, default='none', help='是否选择学习率衰减策略')
@click.option('--svroot', type=str, default='./saved', help='项目文件保存路径')
def experiment(gpu, data, ntr, translate, autoaug, epochs, nbatch, batchsize, lr, lr_scheduler, svroot):
    settings = locals().copy()
    print(settings)

    # 全局设置
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    if not os.path.exists(svroot):
        os.makedirs(svroot)
    writer = SummaryWriter(svroot)

    # 加载数据集和模型
    if data in ['mnist', 'mnist_t']:
        # 加载数据集
        if data == 'mnist':
            trset = data_loader.load_mnist('train', translate=translate, ntr=ntr, autoaug=autoaug)
        elif data == 'mnist_t':
            trset = data_loader.load_mnist_t('train', translate=translate, ntr=ntr)
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                              sampler=RandomSampler(trset, True, nbatch * batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = mnist_net.ConvNet().cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)

    elif data == 'mnistvis':
        trset = data_loader.load_mnist('train')
        teset = data_loader.load_mnist('test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, \
                              sampler=RandomSampler(trset, True, nbatch * batchsize))
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = mnist_net.ConvNetVis().cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)

    elif data == 'cifar10':
        # 加载数据集
        trset = data_loader.load_cifar10(split='train')
        teset = data_loader.load_cifar10(split='test')
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        cls_net = wrn(depth=16, num_classes=10, widen_factor=4, dropRate=0.4, nc=3).cuda()
        cls_opt = optim.SGD(cls_net.parameters(), lr=smooth_step(10, 40, 100, 150, 0), momentum=0.9, weight_decay=1e-5)
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs)
    elif data == 'PACS':
        # trset =
        # teset =
        # trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True, drop_last=True)
        # teloader = DataLoader(teset, batch_size=batchsize, num_workers=8, shuffle=False)
        # cls_net =
        args = get_args()
        args.n_classes = 7
        args.source = ['photo']
        args.target = ['art_painting', 'cartoon', 'sketch']
        trloader, teloader = data_helper.get_train_dataloader(args, patches=False)
        cls_net = caffenet(7).cuda()
        cls_opt = torch.optim.SGD(cls_net.parameters(), lr=0.002, nesterov=True, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(cls_opt, step_size=int(epochs * 0.8))

    elif 'synthia' in data:
        # 加载数据集
        branch = data.split('_')[1]
        trset = data_loader.load_synthia(branch)
        trloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        teloader = DataLoader(trset, batch_size=batchsize, num_workers=8, shuffle=True)
        imsize = [192, 320]
        nclass = 14
        # 加载模型
        cls_net = fcn.FCN_resnet50(nclass=nclass).cuda()
        cls_opt = optim.Adam(cls_net.parameters(), lr=lr)  # , weight_decay=1e-4) # 对于synthia 加上weigh_decay会掉1-2个点
        if lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(cls_opt, epochs * len(trloader))

    cls_criterion = nn.CrossEntropyLoss()

    # 开始训练
    best_acc = 0
    for epoch in range(epochs):
        t1 = time.time()

        loss_list = []
        cls_net.train()
        for i, ((x, _, y), _, idx) in enumerate(trloader):
            x = x.type(torch.FloatTensor)
            x, y = x.cuda(), y.cuda()

            p = cls_net(x)
            cls_loss = cls_criterion(p, y)
            # torch.cuda.synchronize()
            cls_opt.zero_grad()
            cls_loss.backward()
            cls_opt.step()

            loss_list.append([cls_loss.item()])

            scheduler.step()

        cls_loss, = np.mean(loss_list, 0)

        cls_net.eval()
        if data in ['mnist', 'mnist_t', 'cifar10', 'mnistvis']:
            teacc = evaluate_cifar10(cls_net, teloader)
        elif 'synthia' in data:
            teacc = evaluate_seg(cls_net, teloader, nclass)
        if data == 'PACS':
            teacc = evaluate_P(cls_net, teloader)

        if best_acc < teacc:
            best_acc = teacc
            torch.save({'cls_net': cls_net.state_dict()}, os.path.join(svroot, 'best.pkl'))

        t2 = time.time()
        print(f'epoch {epoch}, time {t2 - t1:.2f}, cls_loss {cls_loss:.4f} teacc {teacc:2.2f}')
        writer.add_scalar('scalar/cls_loss', cls_loss, epoch)
        writer.add_scalar('scalar/teacc', teacc, epoch)

    writer.close()


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

