import numpy as np
import torch
import os
from torch import nn
from torch import optim
from torch.nn import functional as F
import sys
from .layers import SNConv3d, SNLinear

class Code_Discriminator(nn.Module):
    def __init__(self, code_size, num_units=256):
        super(Code_Discriminator, self).__init__()

        self.l1 = nn.Sequential(SNLinear(code_size, num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l2 = nn.Sequential(SNLinear(num_units, num_units),
                                nn.LeakyReLU(0.2,inplace=True))
        self.l3 = SNLinear(num_units, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x

class Sub_Encoder(nn.Module): # [64, 32, 48, 36]
    def __init__(self, channel=256, latent_dim=1024):
        super(Sub_Encoder, self).__init__()

        self.latent = latent_dim

        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(channel//4, channel//4, kernel_size=4, stride=2, padding=1) # out:[16,24,18]
        self.bn2 = nn.GroupNorm(8, channel//4)
        self.conv3 = nn.Conv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # out:[8,12,9]
        self.bn3 = nn.GroupNorm(8, channel//2)
        self.conv4 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[4,6,4]
        self.bn4 = nn.GroupNorm(8, channel)
        self.conv5 = nn.Conv3d(channel, latent_dim, kernel_size=4, stride=1, padding=0) # out:[1,1,1,1]
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))



    def forward(self, h):
        h = self.conv2(h)
        h = self.relu(self.bn2(h))
        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        h = self.conv4(h)
        h = self.relu(self.bn4(h))
        h = self.conv5(h)
        # print(h.shape)
        h = self.avg_pool(h).squeeze()
        ##print("-------Finished sub E-------")
        #print(h.shape)
        # print(h.shape)
        assert h.shape[1:] == (self.latent,)
        return h

class Encoder(nn.Module):
    def __init__(self, channel=64): #input: 24,192,144 TODO
        super(Encoder, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv3d(1, channel//2, kernel_size=4, stride=2, padding=1) # in:[16,128,128], out:[8,64,64] ours:[12,96,72]
        self.bn1 = nn.GroupNorm(8, channel//2)
        self.conv2 = nn.Conv3d(channel//2, channel//2, kernel_size=3, stride=1, padding=1) # out:[8,64,64] #ours: [12,96,72]
        self.bn2 = nn.GroupNorm(8, channel//2)
        self.conv3 = nn.Conv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[4,32,32] #[6,48,36]
        self.bn3 = nn.GroupNorm(8, channel)

    def forward(self, h):
        h = self.conv1(h)
        h = self.relu(self.bn1(h))

        h = self.conv2(h)
        h = self.relu(self.bn2(h))

        h = self.conv3(h)
        h = self.relu(self.bn3(h))
        return h

class Sub_Discriminator(nn.Module): # input: 36, 48, 36
    def __init__(self, num_class=0, channel=256):
        super(Sub_Discriminator, self).__init__() #TODO
        self.channel = channel
        self.num_class = num_class

        self.conv2 = SNConv3d(1, channel//4, kernel_size=4, stride=2, padding=1) # out:[16,16,16] #[18, 24, 18]
        self.conv3 = SNConv3d(channel//4, channel//2, kernel_size=4, stride=2, padding=1) # out:[8,8,8] #[9,12,9]
        self.conv4 = SNConv3d(channel//2, channel, kernel_size=4, stride=2, padding=1) # out:[4,4,4] #[4,6,4]
        self.conv5 = SNConv3d(channel, 1+num_class, kernel_size=4, stride=1, padding=0) # out:[1,1,1,1] 
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))

    def forward(self, h):
        # print("..........Inside sub-discriminator...........????")
        #print(h.shape)
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        #print(h.shape)
        if self.num_class == 0:
            # print(h.shape)
            h = self.pool(self.conv5(h)).view((-1,1))
            return h
        else:
            h = self.pool(self.conv5(h)).view((-1,1+self.num_class))
            return h[:,:1], h[:,1:]

class Discriminator(nn.Module): 
    def __init__(self, num_class=0, channel=512):
        super(Discriminator, self).__init__()        
        self.channel = channel
        self.num_class = num_class
        # TODO: input 24, 192, 144
        # D^H
        self.conv2 = SNConv3d(1, channel//16, kernel_size=4, stride=2, padding=1) # out:[8,64,64,64]  # [32,12,96,72]
        self.conv3 = SNConv3d(channel//16, channel//8, kernel_size=4, stride=2, padding=1) # out:[4,32,32,32] # [64,6,48,36]
        self.conv4 = SNConv3d(channel//8, channel//4, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[2,16,16,16] #[3,24,18]
        self.conv5 = SNConv3d(channel//4, channel//2, kernel_size=(2,4,4), stride=(2,2,2), padding=(0,1,1)) # out:[1,8,8,8] #[1,1,12,9]
        self.conv6 = SNConv3d(channel//2, channel, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1)) # out:[1,4,4,4] #[1,1,6,4]
        self.conv7 = SNConv3d(channel, channel//4, kernel_size=(1,4,4), stride=1, padding=0) # out:[1,1,1,1]
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc1 = SNLinear(channel//4+1, channel//8)
        self.fc2 = SNLinear(channel//8, 1)
        if num_class>0:
            self.fc2_class = SNLinear(channel//8, num_class)

        # D^L
        self.sub_D = Sub_Discriminator(num_class)

    def forward(self, h, h_small, crop_idx):
        #print("Inside Discriminator forward")
        h = F.leaky_relu(self.conv2(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv3(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv4(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv5(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv6(h), negative_slope=0.2)
        #print(h.shape)
        h = F.leaky_relu(self.conv7(h), negative_slope=0.2)
        # print(h.shape)
        h = self.pool(h).squeeze()
        # print(h.shape)
        h = torch.cat([h, (crop_idx / 112. * torch.ones((h.size(0), 1))).cuda()], 1) # 128*7/8
        #print(h.shape)
        h = F.leaky_relu(self.fc1(h), negative_slope=0.2)
        h_logit = self.fc2(h) # 2,64
        #print('Before sub model')
        #print(h.shape)
        if self.num_class>0:
            h_class_logit = self.fc2_class(h)

            h_small_logit, h_small_class_logit = self.sub_D(h_small)
            return (h_logit+ h_small_logit)/2., (h_class_logit+ h_small_class_logit)/2.
        else:
            # print("??????????????????")
            # print(h_logit.shape)
            h_small_logit = self.sub_D(h_small)
            # print(h_small_logit.shape)
            return (h_logit+ h_small_logit)/2.


class Sub_Generator(nn.Module): #(36, 48, 36)
    def __init__(self, channel:int=16):
        super(Sub_Generator, self).__init__()
        _c = channel

        self.relu = nn.ReLU()
        self.tp_conv1 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*2)

        self.tp_conv2 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c)

        self.tp_conv3 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, h):

        h = self.tp_conv1(h)
        h = self.relu(self.bn1(h))

        h = self.tp_conv2(h)
        h = self.relu(self.bn2(h))

        h = self.tp_conv3(h)
        h = torch.tanh(h)
        return h

class Generator(nn.Module):
    def __init__(self, mode="train", latent_dim=1024, channel=32, num_class=0):
        super(Generator, self).__init__()
        _c = channel

        self.mode = mode
        self.relu = nn.ReLU()
        self.num_class = num_class

        # G^A and G^H
        self.fc1 = nn.Linear(latent_dim+num_class, 4*4*4*_c*16) #???

        self.tp_conv1 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.GroupNorm(8, _c*16)

        self.tp_conv2 = nn.Conv3d(_c*16, _c*16, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.GroupNorm(8, _c*16)

        self.tp_conv3 = nn.Conv3d(_c*16, _c*8, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.GroupNorm(8, _c*8)

        self.tp_conv4 = nn.Conv3d(_c*8, _c*4, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4 = nn.GroupNorm(8, _c*4)

        self.tp_conv5 = nn.Conv3d(_c*4, _c*2, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5 = nn.GroupNorm(8, _c*2)

        self.tp_conv6 = nn.Conv3d(_c*2, _c, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6 = nn.GroupNorm(8, _c)

        self.tp_conv7 = nn.Conv3d(_c, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # G^L
        self.sub_G = Sub_Generator(channel=_c//2)
 
    def forward(self, h, crop_idx=None, class_label=None): # output: 18, 
        #print("-----Inside generator forward-----")

        # Generate from random noise
        if crop_idx != None or self.mode=='eval':
            if self.num_class > 0:
                h = torch.cat((h, class_label), dim=1)

            h = self.fc1(h)

            h = h.view(-1,512,4,4,4)
            h = self.tp_conv1(h)
            h = self.relu(self.bn1(h))

            #print(h.shape)

            h = F.interpolate(h,scale_factor = 2) #8,8,8
            h = self.tp_conv2(h)
            h = self.relu(self.bn2(h))

            #print(h.shape)

            h = F.interpolate(h,scale_factor = 2) #16,16,16
            h = self.tp_conv3(h)
            h = self.relu(self.bn3(h))

            #print(h.shape)

            h = F.interpolate(h,scale_factor = (2.25, 3, 2.25)) #36,48,36
            h = self.tp_conv4(h)
            h = self.relu(self.bn4(h))

            #print(h.shape)

            h = self.tp_conv5(h)
            h_latent = self.relu(self.bn5(h)) # (32, 32, 32), channel:128 #(36,48,36)

            #print(h_latent.shape)

            if self.mode == "train":
                h_small = self.sub_G(h_latent)
                #print("Finished sub generator")
                #print(h_small.shape)
                h = h_latent[:,:,crop_idx//4:crop_idx//4+6,:,:] # Crop sub-volume, out: (4, 32, 32) # TODO: ours: (6,48,36)
            else:
                h = h_latent

        # Generate from latent feature
        h = F.interpolate(h,scale_factor = 2) # (6, 48, 36)
        #print(h.shape)
        h = self.tp_conv6(h)
        h = self.relu(self.bn6(h)) # (64, 64, 64) #(12,96,72)

        h = F.interpolate(h,scale_factor = 2) #  (2,1,24,192,144)
        h = self.tp_conv7(h)

        #print("Finished main generator")
        #print(h.shape)
        # assert h.shape[2:] == (24, 192, 144)

        h = torch.tanh(h) # (128, 128, 128)

        if crop_idx != None and self.mode == "train":
            return h, h_small
        return h
