import numpy as np
import torch
import os
import time
from skimage.transform import resize
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import nibabel as nib
from torch.utils.data.dataset import Dataset
from Model_alphaWGAN import *
from nilearn import plotting
from pathlib import Path
from dataset import MRIDataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import argparse
from utils import sinkhorn_pointcloud as sp

DATASET = '/scratch/groups/kpohl/tmp_Wei_Project/DATA/MRI_High/Lab_data/img_orig_longitudinal/'

parser = argparse.ArgumentParser(description='PyTorch Alpha-GAN Training')
parser.add_argument('--data-dir', type=str,
                    help='path to the preprocessed data folder')
parser.add_argument('--iter', type=int, default=120000,
                    help='number of iterations')
parser.add_argument('--latent-dim', type=int, default=1000,
                    help='latent dim')
parser.add_argument('--g-iter', type=int, default=1,
                    help='number of g iterations')
parser.add_argument('--batch-size', type=int, default=4,
                    help='batch size')

parser.add_argument('--img-size', type=int, default=64,
                    help='batch size')
parser.add_argument('--exp-name', type=str,
                    help='where to save?')

parser.add_argument('--continue-iter', type=int, default=0,
                    help='continue iteration')


BATCH_SIZE= 8
TEST_BATCH_SIZE=2
gpu = True
workers = 1

LAMBDA= 10
_eps = 1e-15

#setting latent variable sizes
latent_dim = 1000

torch_seed = 0
r_g = torch.manual_seed(torch_seed)


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            # assert images.shape[1:] == (1,144,192,144)
            yield images

def calc_gradient_penalty(model, x, x_gen, w=10):
    """WGAN-GP gradient penalty"""
    assert x.size()==x_gen.size(), "real and sampled sizes do not match"
    alpha_size = tuple((len(x), *(1,)*(x.dim()-1)))
    alpha_t = torch.cuda.FloatTensor if x.is_cuda else torch.Tensor
    alpha = alpha_t(*alpha_size).uniform_()
    x_hat = x.data*alpha + x_gen.data*(1-alpha)
    x_hat = Variable(x_hat, requires_grad=True)

    def eps_norm(x):
        x = x.view(len(x), -1)
        return (x*x+_eps).sum(-1).sqrt()
    def bi_penalty(x):
        return (x-1)**2

    grad_xhat = torch.autograd.grad(model(x_hat).sum(), x_hat, create_graph=True, only_inputs=True)[0]

    penalty = w*bi_penalty(eps_norm(grad_xhat)).mean()
    return penalty
    


def train(args):
    print("Training CCE GAN hyperparameters: ")
    print('latent_dim: {}'.format(args.latent_dim),
          'batch_size: {}'.format(args.batch_size)
        )
    trainset = MRIDataset(path=args.data_dir, resize=False)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                          shuffle=True,num_workers=workers)
    G = Generator(noise = latent_dim)
    D = Discriminator(is_dis=True)
    E = Discriminator(out_class = latent_dim,is_dis=False)

    G.cuda()
    D.cuda()
    E.cuda()

    g_optimizer = optim.Adam(G.parameters(), lr=0.02)
    d_optimizer = optim.Adam(D.parameters(), lr=0.02)
    e_optimizer = optim.Adam(E.parameters(), lr=0.02)
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    if args.continue_iter != 0:
        ckpt_path = './checkpoint/'+args.exp_name+'/G_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
        g_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/D_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/E_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        E.load_state_dict(ckpt['model'])
        e_optimizer.load_state_dict(ckpt['optimizer'])
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        del ckpt
        print("Ckpt", args.exp_name, args.continue_iter, "loaded.")

    g_iter = 1
    d_iter = 2
    gen_load = inf_train_gen(train_loader)

    summary_writer = SummaryWriter("./checkpoint/"+args.exp_name)
    count = 0
    
    for iteration in range(args.continue_iter, args.iter):
        for p in D.parameters():  
            p.requires_grad = False
        for p in E.parameters():  
            p.requires_grad = True
        for p in G.parameters():  
            p.requires_grad = True
        
        ######i#########################################
        # Train Generator 
        ###############################################
        for iters in range(g_iter):
            real_images = gen_load.__next__().cuda()
            _batch_size = real_images.size(0)
            # if not _batch_size == BATCH_SIZE: 
            #     print(f'batch size {_batch_size} is not consitent with {BATCH_SIZE}')
            #     break
            z_rand = torch.randn((_batch_size,latent_dim)).cuda()
            z_hat = E(real_images).view(_batch_size,-1)
            x_hat = G(z_hat)
            x_rand = G(z_rand)
            
            d_recon_loss = D(x_hat).mean()
            d_fake_loss = D(x_rand).mean()
            l1_loss = 100 * criterion_l1(x_hat.cuda(),real_images)
            ### L2 loss(MSE loss) for reconstruction of the Encoder 
            z_ee, z_re = E(x_hat), E(x_rand)
            z_e_l2, z_r_l2 = 50 * criterion_mse(z_hat, z_ee), 50 * criterion_mse(z_rand, z_re)
            ###############################################
            loss1 = l1_loss - d_fake_loss - d_recon_loss + z_e_l2 + z_r_l2 
            
            G.zero_grad()
            E.zero_grad()
            if iters<g_iter-1:
                loss1.backward()
            else:
                loss1.backward(retain_graph=True)
            g_optimizer.step()
            e_optimizer.step()

            ######i#########################################
            # Train Encoder
            ###############################################
        for p in D.parameters():  
            p.requires_grad = False
        for p in E.parameters():  
            p.requires_grad = True
        for p in G.parameters(): 
            p.requires_grad = False

        for iters in range(g_iter):
            z_rand = torch.randn((_batch_size,latent_dim)).cuda()
            z_hat = E(real_images).view(_batch_size,-1)
            ### wasserstein loss between z_e and z_r ###################
            w_dist = 100 * sp.sinkhorn_loss(torch.transpose(z_rand, 0, 1), torch.transpose(z_hat, 0, 1), 0.1, 1000, 100, gpu=0)
            ###############################################
            
            e_loss = w_dist
            E.zero_grad()
            if iters<g_iter-1:
                e_loss.backward(retain_graph=True)
            else:
                e_loss.backward()
            e_optimizer.step()

        ###############################################
        # Train D
        ###############################################
        for p in D.parameters():  
            p.requires_grad = True
        for p in E.parameters():  
            p.requires_grad = False
        for p in G.parameters():  
            p.requires_grad = False
            
        for iters in range(d_iter):
            d_optimizer.zero_grad()
            real_images = gen_load.__next__().cuda()
            _batch_size = real_images.size(0)
            
            # if not _batch_size == BATCH_SIZE: 
            #     print(f'batch size {_batch_size} is not consitent with {BATCH_SIZE}')
            #     break
                
            x_loss2 = -2*D(real_images).mean()+D(x_hat.detach()).mean()+D(x_rand.detach()).mean()
            gradient_penalty_r = calc_gradient_penalty(D,real_images, x_rand)
            gradient_penalty_h = calc_gradient_penalty(D,real_images, x_hat)
            loss2 = x_loss2+gradient_penalty_r+gradient_penalty_h
            if iters < d_iter - 1:
                loss2.backward(retain_graph=True)
            else:
                loss2.backward()
            d_optimizer.step()
        

        if (iteration+1) % 5 == 0:
            print('[{}/{}]'.format(iteration,args.iter),
                'D: {:<8.3}'.format(loss2.item()), 
                'En_Ge: {:<8.3}'.format(loss1.item()),
                'Code: {:<8.3}'.format(e_loss.item()),
                )
            
            summary_writer.add_scalar('D', loss2.item(), iteration)
            summary_writer.add_scalar('En_Ge', loss1.item(), iteration)
            summary_writer.add_scalar('Code', e_loss.item(), iteration)

            featmask = np.squeeze((0.5*real_images[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REAL",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Real', fig, iteration, close=True)

            featmask = np.squeeze((0.5*x_hat[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="REC",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Rec', fig, iteration, close=True)
            
            featmask = np.squeeze((0.5*x_rand[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            fig=plt.figure()
            plotting.plot_img(featmask,title="FAKE",cut_coords=(args.img_size//2,args.img_size//2,args.img_size//16),figure=fig,draw_cross=False,cmap="gray")
            summary_writer.add_figure('Fake', fig, iteration, close=True)

        if (iteration+1) >= 40000 and (iteration+1) % 5000 == 0 and count < 10:
            featmask = np.squeeze((0.5*x_rand[0]+0.5).data.cpu().numpy())
            featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
            nib.save(featmask, f'./checkpoint/{args.exp_name}/CCEGAN_training_sample_{count}_epoch_{iteration}.nii.gz')
            count += 1
            print("Sample saved!")


        if (iteration)%500==0: 
            os.makedirs(f'./checkpoint/{args.exp_name}', exist_ok=True)
            torch.save({'model': G.state_dict(), 'optimizer': g_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/G_iter'+str(iteration+1)+'.pth')
            # print(f"Saved at './checkpoint/{args.exp_name}/G_iter'+str(iteration+1)+'.pth'")
            print("Ckpt saved!")
            torch.save({'model': D.state_dict(), 'optimizer': d_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/D_iter'+str(iteration+1)+'.pth')
            torch.save({'model': E.state_dict(), 'optimizer': e_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/E_iter'+str(iteration+1)+'.pth')
            
    print("Writing tensorboard information")
    summary_writer.flush()
    summary_writer.close()
    
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)