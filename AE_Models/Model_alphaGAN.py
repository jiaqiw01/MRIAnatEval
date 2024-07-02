import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F

#***********************************************
#Encoder and Discriminator has same architecture
#***********************************************
class Discriminator(nn.Module):
    def __init__(self, channel=256,out_class=1,is_dis =True):
        super(Discriminator, self).__init__()
        self.is_dis=is_dis
        self.channel = channel
        n_class = out_class 
        
        self.conv1 = nn.Conv3d(1, channel//8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(channel//8, channel//4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(channel)

        self.conv5 = nn.Conv3d(channel, n_class, kernel_size=4, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

        # alternatively adaptive global pooling
        # self.global_pool = nn.AdaptiveAvgPool3d(n_class)
        
    def forward(self, x, _return_activations=False):
        # #print("Inside Disciminator/Encoder!")
        # #print("input shape is: ", x.shape)
        h1 = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        #print(h1.shape)
        h2 = F.leaky_relu(self.bn2(self.conv2(h1)), negative_slope=0.2)
        #print("h2")
        #print(h2.shape)
        h3 = F.leaky_relu(self.bn3(self.conv3(h2)), negative_slope=0.2)
        #print("h3")
        #print(h3.shape)
        h4 = F.leaky_relu(self.bn4(self.conv4(h3)), negative_slope=0.2)
        #print("h4")
        #print(h4.shape)
        # h5 = F.leaky_relu(self.bn5(self.conv5(h4)), negative_slope=0.2)
        #print("h5")
        #print(h5.shape)

        h6 = self.pool(self.conv5(h4))

        # #print("h6")
        # #print(h6.shape)

        if self.is_dis:
            output = F.sigmoid(h6.view(h6.size()[0],-1))
        else:
            output = h6.view(h6.size()[0],-1)
        # #print("output shape: ", output.shape)
        return output
    
class Code_Discriminator(nn.Module):
    def __init__(self, code_size=100,num_units=750):
        super(Code_Discriminator, self).__init__()
        n_class = 1
        self.l1 = nn.Sequential(nn.Linear(code_size, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(nn.Linear(num_units, num_units),
                                nn.BatchNorm1d(num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = nn.Linear(num_units, 1)
        
    def forward(self, x):
        h1 = self.l1(x)
        h2 = self.l2(h1)
        h3 = self.l3(h2)
        output = F.sigmoid(h3)
            
        return output

class Generator(nn.Module):
    def __init__(self, noise:int=100, channel:int=64):
        super(Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.noise = noise
        self.tp_conv1 = nn.ConvTranspose3d(noise, _c*4, kernel_size=(4,6,4), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv2 = nn.Conv3d(_c*4, _c*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(_c*4)
        
        self.tp_conv3 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(_c*2)
        
        self.tp_conv4 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm3d(_c)
        
        self.tp_conv5 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)

        # self.tp_conv6 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, noise):

        noise = noise.view(-1,self.noise,1,1,1)
        h = self.tp_conv1(noise)
        #print('///')
        #print(h.shape)
        h = self.relu(self.bn1(h))
        
        h = F.interpolate(h,scale_factor=2)
        #print("....")
        #print(h.shape)
        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))
     
        h = F.interpolate(h,scale_factor = 2)
        #print("!!!!")
        #print(h.shape)
        h = self.tp_conv3(h)
        h = self.relu(self.bn3(h))

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv4(h)
        h = self.relu(self.bn4(h))

        h = F.interpolate(h,scale_factor = 2)
        h = self.tp_conv5(h)

        h = F.upsample(h, scale_factor=(2.25, 2, 2.25))
        # h = self.tp_conv6(h)

        h = F.tanh(h)

        return h