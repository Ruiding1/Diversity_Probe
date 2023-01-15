
import torch
import torch.nn as nn
import torch.nn.functional as F

#from utils import reparametrize

class ConvNet_TSNE(nn.Module):
    ''' cvpr2020 M-ADA '''

    def __init__(self, imdim=3):
        super(ConvNet_TSNE, self).__init__()

        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU(inplace=True)

        self.cls_head_src = nn.Linear(1024, 10)
        self.cls_head_tgt = nn.Linear(1024, 10)
        self.pro_head = nn.Linear(1024, 128)

        # l2D
        self.classifier_l = nn.Linear(512, 10)
        self.p_logvar = nn.Sequential(nn.Linear(1024, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(1024, 512),
                                  nn.LeakyReLU())

    def reparametrize(self, mu, logvar, factor=0.2):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + factor * std * eps

    def forward(self, x, mode='test'):
        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))

        # if mode == 'test':
        p = self.cls_head_src(out4)
        return p
        # elif mode == 'train':
        #     p = self.cls_head_src(out4)
        #     z = self.pro_head(out4)
        #     z = F.normalize(z)
        #     return p,z
        # elif mode == 'L2D':
        #     end_points = {}
        #     logvar = self.p_logvar(out4)
        #     mu = self.p_mu(out4)
        #     end_points['logvar'] = logvar
        #     end_points['mu'] = mu
        #
        #     ## reparametrize
        #     x = self.reparametrize(mu, logvar)
        #     end_points['Embedding'] = x
        #     x = self.classifier_l(x)
        #     end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)
        #
        #     return x, end_points
        # elif mode == 'target':
        #    p = self.cls_head_tgt(out4)
        #    z = self.pro_head(out4)
        #    z = F.normalize(z)
        #    return p,z

class ConvNet(nn.Module):
    ''' cvpr2020 M-ADA '''
    def __init__(self, imdim=3):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.cls_head_src = nn.Linear(1024, 10)
        self.cls_head_tgt = nn.Linear(1024, 10)
        self.pro_head = nn.Linear(1024, 128)

        # l2D
        self.classifier_l = nn.Linear(512, 10)
        self.p_logvar = nn.Sequential(nn.Linear(1024, 512),
                                      nn.ReLU())
        self.p_mu = nn.Sequential(nn.Linear(1024, 512),
                                  nn.LeakyReLU())

    def reparametrize(self, mu, logvar, factor=0.2):
        std = logvar.div(2).exp()
        eps = std.data.new(std.size()).normal_()
        return mu + factor * std * eps

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        
        if mode == 'test':
            p = self.cls_head_src(out4)
            return p
        elif mode == 'train':
            p = self.cls_head_src(out4)
            z = self.pro_head(out4)
            z = F.normalize(z)
            return p,z
        elif mode == 'L2D':
            end_points = {}
            logvar = self.p_logvar(out4)
            mu = self.p_mu(out4)
            end_points['logvar'] = logvar
            end_points['mu'] = mu

            ## reparametrize
            x = self.reparametrize(mu, logvar)
            end_points['Embedding'] = x
            x = self.classifier_l(x)
            end_points['Predictions'] = nn.functional.softmax(input=x, dim=-1)

            return x, end_points
        #elif mode == 'target':
        #    p = self.cls_head_tgt(out4)
        #    z = self.pro_head(out4)
        #    z = F.normalize(z)
        #    return p,z
    
class ConvNetVis(nn.Module):

    def __init__(self, imdim=3):
        super(ConvNetVis, self).__init__()

        self.conv1 = nn.Conv2d(imdim, 64, kernel_size=5, stride=1, padding=0)
        self.mp = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128*5*5, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 2)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.cls_head_src = nn.Linear(2, 10)
        self.cls_head_tgt = nn.Linear(2, 10)
        self.pro_head = nn.Linear(2, 128)

    def forward(self, x, mode='test'):

        in_size = x.size(0)
        out1 = self.mp(self.relu1(self.conv1(x)))
        out2 = self.mp(self.relu2(self.conv2(out1)))
        out2 = out2.view(in_size, -1)
        out3 = self.relu3(self.fc1(out2))
        out4 = self.relu4(self.fc2(out3))
        
        if mode == 'test':
            p = self.cls_head_src(out4)
            return p
        elif mode == 'train':
            p = self.cls_head_src(out4)
            z = self.pro_head(out4)
            z = F.normalize(z)
            return p,z
        elif mode == 'p_f':
            p = self.cls_head_src(out4)
            return p, out4
        #elif mode == 'target':
        #    p = self.cls_head_tgt(out4)
        #    z = self.pro_head(out4)
        #    z = F.normalize(z)
        #    return p,z
    

