import numpy as np
import torch
import os
import time
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from Model_WGAN import *
from nilearn import plotting
from pathlib import Path
from dataset import MRIDataset
from tensorboardX import SummaryWriter
import argparse
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='PyTorch Alpha-GAN Training')
parser.add_argument('--data-dir', type=str,
                    help='path to the preprocessed data folder')
parser.add_argument('--iter', type=int, default=150000,
                    help='number of iterations')
parser.add_argument('--latent-dim', type=int, default=1000,
                    help='latent dim')
parser.add_argument('--g-iter', type=int, default=1,
                    help='number of g iterations')
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size')

parser.add_argument('--img-size', type=int, default=64,
                    help='batch size')
parser.add_argument('--exp-name', type=str,
                    help='where to save?')

parser.add_argument('--continue-iter', type=int, default=0,
                    help='continue iteration')


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            assert images.shape[1:] == (1,144,192,144)
            yield images

def calc_gradient_penalty(netD, real_data, fake_data):    
    alpha = torch.rand(real_data.size(0),1,1,1,1)
    alpha = alpha.expand(real_data.size())
    
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

BATCH_SIZE=1
max_epoch = 100
lr = 0.0001
gpu = True
latent_dim=1000
workers = 4

LAMBDA= 10


lr_g = 0.0001
lr_d = 0.0001
    
def train(args):
    print("Training WGAN hyperparameters: ")
    print('lr_g: {:<8.3}'.format(lr_g), 
          'lr_d: {:<8.3}'.format(lr_d),
          'g_iter: {}'.format(args.g_iter),
          'latent_dim: {}'.format(args.latent_dim),
          'batch_size: {}'.format(args.batch_size)
        )
    trainset = MRIDataset(path=args.data_dir, resize=False)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                          shuffle=True,num_workers=workers)
    D = Discriminator().cuda()
    G = Generator(noise = latent_dim).cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=lr_g)
    d_optimizer = optim.Adam(D.parameters(), lr=lr_d)

    real_y = Variable(torch.ones((args.batch_size, 1)).cuda())
    fake_y = Variable(torch.zeros((args.batch_size, 1)).cuda())
    loss_f = nn.BCELoss()

    if args.continue_iter != 0:
        ckpt_path = './checkpoint/'+args.exp_name+'/G_W_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
        g_optimizer.load_state_dict(ckpt['optimizer'])

        ckpt_path = './checkpoint/'+args.exp_name+'/D_W_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
        
        del ckpt
        print("Ckpt", args.exp_name, args.continue_iter, "loaded.")

    d_real_losses = list()
    d_fake_losses = list()
    d_losses = list()
    g_losses = list()
    divergences = list()
    restart = True

    summary_writer = SummaryWriter("./checkpoint/"+args.exp_name)

    gen_load = inf_train_gen(train_loader)
    for iteration in range(args.continue_iter, args.iter):
        ###############################################
        # Train D 
        ###############################################
        if restart:
            start = time.time()
            restart = False

        for p in D.parameters():  
            p.requires_grad = True 

        real_images = gen_load.__next__()
        D.zero_grad()
        real_images = Variable(real_images).cuda()

        _batch_size = real_images.size(0)


        y_real_pred = D(real_images)

        d_real_loss = y_real_pred.mean()
        
        noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1, 1)),volatile=True).cuda()
        fake_images = G(noise)
        y_fake_pred = D(fake_images.detach())

        d_fake_loss = y_fake_pred.mean()

        gradient_penalty = calc_gradient_penalty(D,real_images.data, fake_images.data)
    
        d_loss = - d_real_loss + d_fake_loss +gradient_penalty
        d_loss.backward()
        Wasserstein_D = d_real_loss - d_fake_loss

        d_optimizer.step()

        ###############################################
        # Train G 
        ###############################################
        for p in D.parameters():
            p.requires_grad = False
            
        for iters in range(5):
            G.zero_grad()
            noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1 ,1)).cuda())
            fake_image =G(noise)
            y_fake_g = D(fake_image)

            g_loss = -y_fake_g.mean()

            g_loss.backward()
            g_optimizer.step()

        ###############################################
        # Visualization
        ###############################################
        if (iteration+1)%1000 == 0:
            end = time.time()
            duration = int(end - start)
            restart = True
            d_real_losses.append(d_real_loss.item())
            d_fake_losses.append(d_fake_loss.item())
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

            print('[{}/{}]'.format(iteration,args.iter),
                'D: {:<8.3}'.format(d_loss.item()), 
                'D_real: {:<8.3}'.format(d_real_loss.item()),
                'D_fake: {:<8.3}'.format(d_fake_loss.item()), 
                'G: {:<8.3}'.format(g_loss.item()),
                'Time: {}'.format(duration)
                )
            summary_writer.add_scalar('D', d_loss.item(), iteration)
            summary_writer.add_scalar('Ge', g_loss.item(), iteration)

            featmask = np.squeeze((0.5*real_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Real', fig, iteration, close=True)
            
            featmask = np.squeeze((0.5*fake_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Fake', fig, iteration, close=True)
        
        if (iteration+1)%5000 ==0: 
            torch.save({'model': G.state_dict(), 'optimizer': g_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/G_W_iter'+str(iteration+1)+'.pth')
            torch.save({'model': D.state_dict(), 'optimizer': d_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/D_W_iter'+str(iteration+1)+'.pth')
            
    print("Writing tensorboard information")
    summary_writer.flush()
    summary_writer.close()
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)