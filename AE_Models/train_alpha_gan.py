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
from Model_alphaGAN import *
from nilearn import plotting
from pathlib import Path
from dataset import MRIDataset
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import argparse

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
parser.add_argument('--batch-size', type=int, default=1,
                    help='batch size')

parser.add_argument('--img-size', type=int, default=64,
                    help='batch size')
parser.add_argument('--exp-name', type=str,
                    help='where to save?')

parser.add_argument('--continue-iter', type=int, default=0,
                    help='continue iteration')


gpu = True
workers = 1
LAMBDA= 10
_eps = 1e-15

Use_BRATS = False
Use_ATLAS = False

#setting latent variable sizes
latent_dim = 1000

lr_g = 0.0001
lr_e = 0.0001
lr_d = 0.0001
lr_cd = 0.0001


def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            # assert images.shape[1:] == (1,144,192,144)
            yield images

def train(args):
    print("Training Alpha GAN hyperparameters: ")
    print('lr_g: {:<8.3}'.format(lr_g), 
          'lr_e: {:<8.3}'.format(lr_e),
          'lr_d: {:<8.3}'.format(lr_d),
          'lr_cd: {:<8.3}'.format(lr_cd),
          'g_iter: {}'.format(args.g_iter),
          'latent_dim: {}'.format(args.latent_dim),
          'batch_size: {}'.format(args.batch_size)
        )
    trainset = MRIDataset(path=args.data_dir, resize=False)
    # trainset = BrainMRIDataset(DATASET)
    train_loader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,
                                          shuffle=True,num_workers=workers)
    G = Generator(noise = latent_dim)
    CD = Code_Discriminator(code_size = latent_dim ,num_units = 512)
    D = Discriminator(is_dis=True)
    E = Discriminator(out_class = latent_dim ,is_dis=False)

    G.cuda()
    D.cuda()
    CD.cuda()
    E.cuda()

    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    E = nn.DataParallel(E)
    CD = nn.DataParallel(CD)

    g_optimizer = optim.Adam(G.parameters(), lr=lr_g)
    d_optimizer = optim.Adam(D.parameters(), lr=lr_d)
    e_optimizer = optim.Adam(E.parameters(), lr=lr_e)
    cd_optimizer = optim.Adam(CD.parameters(), lr=lr_cd)

    if args.continue_iter != 0:
        ckpt_path = './checkpoint/'+args.exp_name+'/G_noW_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
        g_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/D_noW_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        D.load_state_dict(ckpt['model'])
        d_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/E_noW_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        E.load_state_dict(ckpt['model'])
        e_optimizer.load_state_dict(ckpt['optimizer'])
        ckpt_path = './checkpoint/'+args.exp_name+'/CD_noW_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        # ckpt['model'] = trim_state_dict_name(ckpt['model'])
        CD.load_state_dict(ckpt['model'])
        cd_optimizer.load_state_dict(ckpt['optimizer'])
        # sub_e_optimizer.load_state_dict(ckpt['optimizer'])
        del ckpt
        print("Ckpt", args.exp_name, args.continue_iter, "loaded.")


    real_y = Variable(torch.ones((args.batch_size, 1)).cuda())
    fake_y = Variable(torch.zeros((args.batch_size, 1)).cuda())

    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()

    gen_load = inf_train_gen(train_loader)
    restart = True

    summary_writer = SummaryWriter("./checkpoint/"+args.exp_name)

    for iteration in range(args.continue_iter, args.iter):
        if restart:
            start = time.time()
            restart = False
        # print("Training encoder!!!")
        ###############################################
        # Train Encoder - Generator 
        ###############################################
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in CD.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in E.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in G.parameters():  # reset requires_grad
            p.requires_grad = True

        g_optimizer.zero_grad()
        e_optimizer.zero_grad()


        for iters in range(args.g_iter):
            real_images = gen_load.__next__()
            real_images = Variable(real_images,volatile=True).cuda()
            _batch_size = real_images.size(0)
            z_hat = E(real_images).view(_batch_size,-1)
            # print("z hat shape is: ", z_hat.shape)
            # print("------------Finished Encoding------------")
            z_rand = Variable(torch.randn((_batch_size,latent_dim)),requires_grad=False).cuda()
            # print("------------Start Generating-------------")
            x_hat = G(z_hat)
            # print("~~~~HERE~~~~")
            # print("x_hat shape is: ", x_hat.shape)
            x_rand = G(z_rand)
            # print(x_rand.shape)
            # assert x_rand.shape[2:] == (144, 192, 144)

            l1_loss = 10 * criterion_l1(x_hat, real_images)
            res = CD(z_hat)
            # print(res.shape)
            # print(real_y[:_batch_size].shape)
            c_loss = criterion_bce(CD(z_hat), real_y[:_batch_size])
            d_real_loss = criterion_bce(D(x_hat), real_y[:_batch_size]) 
            d_fake_loss = criterion_bce(D(x_rand), real_y[:_batch_size])

            loss1 = l1_loss + c_loss + d_real_loss + d_fake_loss

            loss1.backward(retain_graph=True)
            e_optimizer.step()

            g_optimizer.step()
            g_optimizer.step()

        ###############################################
        # Train D
        ###############################################
        # print("Training D!!")
        for p in D.parameters():  
            p.requires_grad = True
        for p in CD.parameters():  
            p.requires_grad = False
        for p in E.parameters():  
            p.requires_grad = False
        for p in G.parameters():  
            p.requires_grad = False

        for iters in range(1):
            d_optimizer.zero_grad()

            z_rand = Variable(torch.randn((_batch_size,latent_dim)),volatile=True).cuda()
            z_hat = E(real_images).view(_batch_size,-1)
            x_hat = G(z_hat)
            x_rand = G(z_rand)

            x_loss2 = 2.0 * criterion_bce(D(real_images), real_y[:_batch_size])+criterion_bce(D(x_hat), fake_y[:_batch_size])
            z_loss2 = criterion_bce(D(x_rand), fake_y[:_batch_size])
            loss2 = x_loss2 + z_loss2
            # print(loss2)

            if iters<4:
                loss2.backward(retain_graph=True)
            else:
                loss2.backward(retain_graph=True)
            d_optimizer.step()
        ###############################################
        # Train CD
        ###############################################
        # print("Training CD!!!")
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in CD.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in E.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in G.parameters():  # reset requires_grad
            p.requires_grad = False

        for iters in range(1):
            cd_optimizer.zero_grad()
            z_hat = E(real_images).view(_batch_size,-1)
            x_loss3 = criterion_bce(CD(z_hat), fake_y[:_batch_size])
            z_rand = Variable(torch.randn((_batch_size,latent_dim)),volatile=True).cuda()
            z_loss3 = criterion_bce(CD(z_rand), real_y[:_batch_size])
            loss3 = x_loss3 + z_loss3
            loss3.backward(retain_graph=True)
            cd_optimizer.step()

        if (iteration+1) % 1000 == 0:
            end = time.time()
            duration = int(end - start)
            restart = True

            print('[{}/{}]'.format(iteration,args.iter),
                'D: {:<8.3}'.format(loss2.item()), 
                'En_Ge: {:<8.3}'.format(loss1.item()),
                'Code: {:<8.3}'.format(loss3.item()),
                f'Time: {duration}'
                )
            
            summary_writer.add_scalar('D', loss2.item(), iteration)
            summary_writer.add_scalar('En_Ge', loss1.item(), iteration)
            summary_writer.add_scalar('Code', loss3.item(), iteration)

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

        # if (iteration+1) >= 40000 and (iteration+1) % 5000 == 0 and count < 10:
        #     # featmask = np.squeeze((0.5*x_rand[0]+0.5).data.cpu().numpy())
        #     featmask = nib.Nifti1Image(featmask.transpose((2,1,0)),affine = np.eye(4))
        #     nib.save(featmask, f'./checkpoint/{args.exp_name}/alphaGAN_training_sample_{count}_epoch_{iteration}.nii.gz')
        #     count += 1
        #     print("Sample saved!")


        if (iteration+1)%3000==0: 
            os.makedirs(f'./checkpoint/{args.exp_name}', exist_ok=True)
            torch.save({'model': G.state_dict(), 'optimizer': g_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/G_noW_iter'+str(iteration+1)+'.pth')
            # print(f"Saved at './checkpoint/{args.exp_name}/G_iter'+str(iteration+1)+'.pth'")
            print("Ckpt saved!")
            torch.save({'model': D.state_dict(), 'optimizer': d_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/D_noW_iter'+str(iteration+1)+'.pth')
            torch.save({'model': E.state_dict(), 'optimizer': e_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/E_noW_iter'+str(iteration+1)+'.pth')
            torch.save({'model': CD.state_dict(), 'optimizer': cd_optimizer.state_dict()},f'./checkpoint/{args.exp_name}/CD_noW_iter'+str(iteration+1)+'.pth')
            
    print("Writing tensorboard information")
    summary_writer.flush()
    summary_writer.close()
    
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)