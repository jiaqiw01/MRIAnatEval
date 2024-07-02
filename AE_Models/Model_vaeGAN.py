import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self, channel=512,out_class=1): #144,192,144
        super(Discriminator, self).__init__()
        
        self.channel = channel
        n_class = out_class
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1) #72,96,72
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1) #36,48,36
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)#18,24,18
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)#9,12,9
        self.bn4 = nn.BatchNorm3d(channel)
        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=2, padding=1) #4, 6, 4
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        
        # self.conv6 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=0)

        
    def forward(self, x):
        batch_size = x.size()[0]
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2, inplace=False)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2, inplace=False)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2, inplace=False)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2, inplace=False)
        h5 = self.conv5(h4)
        h6 = self.pool(h5)
        # print("Here")
        # print(h5.shape)
        # print(h6.shape)
        output = F.sigmoid(h6.view(h6.size()[0],-1))
        return output

class Encoder(nn.Module):
    def __init__(self, channel=512,out_class=1): # 144, 192, 144
        super(Encoder, self).__init__()
        
        self.channel = channel
        n_class = out_class 
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1) # 72, 96, 72
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1) # 36, 48, 36
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # 18, 24, 18
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # 9, 12, 9
        self.bn4 = nn.BatchNorm3d(channel)
        self.pool = nn.AdaptiveAvgPool3d((4,4,4))

        self.mean = nn.Sequential(
            nn.Linear(32768, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1000))
        self.logvar = nn.Sequential(
            nn.Linear(32768, 2048),
            # nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1000))
        
    def forward(self, x, _return_activations=False):
        batch_size = x.size()[0]
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        h4 = self.pool(h4)
        
        mean = self.mean(h4.view(batch_size,-1))
        logvar = self.logvar(h4.view(batch_size,-1))

        std = logvar.mul(0.5).exp_()
        reparametrized_noise = torch.autograd.Variable(torch.randn((batch_size, 1000))).cuda()
        reparametrized_noise = mean + std * reparametrized_noise
        return mean,logvar ,reparametrized_noise
    
class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel
        
        self.noise = noise
        self.fc = nn.Linear(1000,512*4*6*4)
        self.bn1 = nn.BatchNorm3d(_c*8)
        
        self.tp_conv2 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # self.tp_conv6 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        
    def forward(self, noise):
        noise = noise.view(-1, 1000)
        h = self.fc(noise)
        h = h.view(-1,512,4,6,4)
        h = F.relu(self.bn1(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv2(h)
        h = F.relu(self.bn2(h))
    
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv3(h)
        h = F.relu(self.bn3(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = F.relu(self.bn4(h))
        
        h = F.upsample(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = F.upsample(h, scale_factor=(2.25, 2, 2.25))
        # h = self.tp_conv6(h)

        h = F.tanh(h)

        assert h.shape[1:] == (1, 144, 192, 144)
        
        return h
