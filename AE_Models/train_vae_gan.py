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
from Model_vaeGAN import *
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
parser.add_argument('--iter', type=int, default=100,
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
    

BATCH_SIZE=1
gpu = True
workers = 1
LAMBDA= 10
_eps = 1e-15

Use_BRATS = False
Use_ATLAS = False

gamma = 20
beta = 10

#setting latent variable sizes
latent_dim = 1000
lr_g = 0.0001
lr_e = 0.0001
lr_d = 0.0001


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            assert images.shape[1:] == (1,144,192,144)
            yield images

def train(args):
    print("Training VAE GAN hyperparameters: ")
    print('lr_g: {:<8.3}'.format(lr_g), 
          'lr_e: {:<8.3}'.format(lr_e),
          'lr_d: {:<8.3}'.format(lr_d),
          'g_iter: {}'.format(args.g_iter),
          'latent_dim: {}'.format(args.latent_dim),
          'batch_size: {}'.format(args.batch_size)
        )
    trainset = MRIDataset(path=args.data_dir, resize=False)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                          shuffle=True,num_workers=workers)

    G = Generator(noise = latent_dim)
    D = Discriminator()
    E = Encoder()

    G.cuda()
    D.cuda()
    E.cuda()

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    E = nn.DataParallel(E)

    g_optimizer = optim.Adam(G.parameters(), lr=lr_g)
    d_optimizer = optim.Adam(D.parameters(), lr=lr_d)
    e_optimizer = optim.Adam(E.parameters(), lr=lr_e)

    if args.continue_iter != 0:
        ckpt_path = './checkpoint/'+args.exp_name+'/G_VG_ep_'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt)
        # g_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/D_VG_ep_'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt)
        # d_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/E_VG_ep_'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        E.load_state_dict(ckpt)
        # e_optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        print("Ckpt", args.exp_name, args.continue_iter, "loaded.")

    real_y = Variable(torch.ones((args.batch_size, 1)).cuda())
    fake_y = Variable(torch.zeros((args.batch_size, 1)).cuda())

    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    gen_load = inf_train_gen(train_loader)
    MAX_ITER = args.iter

    restart = True

    summary_writer = SummaryWriter("./checkpoint/"+args.exp_name)

    for iteration in range(args.continue_iter, args.iter):
        print(f"....Iteration {iteration}....")
        start = time.time()
        for step, real_images in enumerate(train_loader):
            _batch_size = real_images.size(0)
            real_images = Variable(real_images,requires_grad=False).cuda()
            z_rand = Variable(torch.randn((_batch_size, latent_dim)),requires_grad=False).cuda()
            mean,logvar,code = E(real_images)
            x_rec = G(code)
            assert x_rec.shape == real_images.shape
            x_rand = G(z_rand)
            ###############################################
            # Train D 
            ###############################################
            d_optimizer.zero_grad()
            
            d_real_loss = criterion_bce(D(real_images),real_y[:_batch_size])
            d_recon_loss = criterion_bce(D(x_rec), fake_y[:_batch_size])
            d_fake_loss = criterion_bce(D(x_rand), fake_y[:_batch_size])
            
            dis_loss = d_recon_loss+d_real_loss + d_fake_loss
            dis_loss.backward(retain_graph=True)
            
            d_optimizer.step()
        
            ###############################################
            # Train G
            ###############################################
            g_optimizer.zero_grad()
            output = D(real_images)
            d_real_loss = criterion_bce(output,real_y[:_batch_size])
            output = D(x_rec)
            d_recon_loss = criterion_bce(output,fake_y[:_batch_size])
            output = D(x_rand)
            d_fake_loss = criterion_bce(output,fake_y[:_batch_size])
            
            d_img_loss = d_real_loss + d_recon_loss+ d_fake_loss
            gen_img_loss = -d_img_loss
            
            rec_loss = ((x_rec - real_images)**2).mean()
            
            err_dec = gamma* rec_loss + gen_img_loss
            
            err_dec.backward(retain_graph=True)
            g_optimizer.step()
            
            ###############################################
            # Train E
            ###############################################
            prior_loss = 1+logvar-mean.pow(2) - logvar.exp()
            prior_loss = (-0.5*torch.sum(prior_loss))/torch.numel(mean.data)
            rec_loss = ((G(code) - real_images)**2).mean()
            err_enc = prior_loss + beta*rec_loss # TODO: ?????
            
            e_optimizer.zero_grad()
            err_enc.backward()
            e_optimizer.step()
            # TODO: moved g_optimizer here according to 
            # https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-code-worked-in-pytorch-1-2-but-not-in-1-5-after-updating/87327/4
            # g_optimizer.step()
        # print("Finished one iteration")
            
        if (iteration+1) % 5 == 0:
            end = time.time()
            duration = int(end - start)
            print('[{}/{}]'.format(iteration,MAX_ITER),
                'D: {:<8.3}'.format(dis_loss.item()), 
                'En: {:<8.3}'.format(err_enc.item()),
                'De: {:<8.3}'.format(err_dec.item()),
                f'Time: {duration}'
                )
            
            summary_writer.add_scalar('D', dis_loss.item(), iteration)
            summary_writer.add_scalar('En_Ge', err_enc.item(), iteration)
            summary_writer.add_scalar('Code', err_dec.item(), iteration)

            featmask = np.squeeze((0.5*real_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Real', fig, iteration, close=True)

            featmask = np.squeeze((0.5*x_rec[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REC",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Rec', fig, iteration, close=True)
            
            featmask = np.squeeze((0.5*x_rand[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Fake', fig, iteration, close=True)
                
            torch.save(G.state_dict(),f'./checkpoint/{args.exp_name}/G_VG_ep_'+str(iteration+1)+'.pth')
            print(f"Saved at './checkpoint/{args.exp_name}/G_VG_ep_'+str(iteration+1)+'.pth'")
            torch.save(D.state_dict(),f'./checkpoint/{args.exp_name}/D_VG_ep_'+str(iteration+1)+'.pth')
            torch.save(E.state_dict(),f'./checkpoint/{args.exp_name}/E_VG_ep_'+str(iteration+1)+'.pth')
    
    print("Writing tensorboard information")
    summary_writer.flush()
    summary_writer.close()
    
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)