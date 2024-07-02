#!/usr/bin/env python3
'''
    Code is from https://github.com/batmanlab/HA-GAN
'''

import os
import time
from argparse import ArgumentParser

import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
from torch.nn import functional as F

# from volume_dataset import Volume_Dataset

# from models.Model_HA_GAN_256 import Generator, Encoder, Sub_Encoder
from resnet3D import resnet50
# from utils import get_resnet_extractor

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

# parser = ArgumentParser()
# parser.add_argument('--path', type=str, default='')
# parser.add_argument('--real_suffix', type=str, default='eval_600_size_256_resnet50_fold')
# parser.add_argument('--img_size', type=int, default=256)
# parser.add_argument('--batch_size', type=int, default=2)
# parser.add_argument('--num_workers', type=int, default=8)
# parser.add_argument('--num_samples', type=int, default=2048)
# parser.add_argument('--dims', type=int, default=2048)
# parser.add_argument('--ckpt_step', type=int, default=80000)
# parser.add_argument('--latent_dim', type=int, default=1024)
# parser.add_argument('--basename', type=str, default="256_1024_Alpha_SN_v4plus_4_l1_GN_threshold_600_fold")
# parser.add_argument('--fold', type=int)

def get_activations_from_dataloader(model, dataset, device, dims=2048, num_samples=400, batch_size=1):
    pred_arr = np.empty((num_samples, dims))
    for i, X in enumerate(dataset):
        if i % 10 == 0:
            print('\rPropagating batch %d' % i, end='', flush=True)
        if type(X) == np.ndarray:
            X = torch.from_numpy(X)
        X = X.unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            pred = model(X)
        if i*batch_size > pred_arr.shape[0]:
            pred_arr[i*batch_size:] = pred.cpu().numpy()
        else:
            pred_arr[i*batch_size:(i+1)*batch_size] = pred.cpu().numpy()
    print(' done')
    return pred_arr

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def post_process(act):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# def calculate_fid_real(dataset, dims, num_samples=400):
#     """Calculates the FID of two paths"""
#     assert os.path.exists("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")

#     model = get_feature_extractor()
#     #dataset = COPD_dataset(img_size=args.img_size, stage="train", fold=args.fold, threshold=600)
#     # dataset = Brain_dataset(img_size=args.img_size, stage="train", fold=args.fold)
#     # args.num_samples = len(dataset)
#     print("Number of samples:", args.num_samples)
#     data_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,drop_last=False,
#                                                shuffle=False,num_workers=args.num_workers)
#     act = get_activations_from_dataloader(model, data_loader, dims, num_samples)
#     np.save("./results/fid/pred_arr_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", act)
#     #np.save("./results/fid/pred_arr_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", act)
#     #calculate_mmd(args, act)
#     m, s = post_process(act)

#     m1 = np.load("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")
#     s1 = np.load("./results/fid/s_real_"+args.real_suffix+str(args.fold)+".npy")

#     fid_value = calculate_frechet_distance(m1, s1, m, s)
#     print('FID: ', fid_value)
#     #np.save("./results/fid/m_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", m)
#     #np.save("./results/fid/s_real_train_600_size_"+str(args.img_size)+"_resnet50_fold"+str(args.fold)+".npy", s)
#     #np.save("./results/fid/m_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", m)
#     #np.save("./results/fid/s_real_train_size_"+str(args.img_size)+"_resnet50_GSP_fold"+str(args.fold)+".npy", s)


def calculate_fid_fake(sample_feature):
    #assert os.path.exists("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")
    # act = generate_samples(args)
    m2, s2 = post_process(sample_feature)

    m1 = np.load("./results/fid/m_real_"+args.real_suffix+str(args.fold)+".npy")
    s1 = np.load("./results/fid/s_real_"+args.real_suffix+str(args.fold)+".npy")

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    print('FID: ', fid_value)



if __name__ == '__main__':
    args = parser.parse_args()
    start_time = time.time()
    calculate_fid_real(args)
    calculate_fid_fake(args)
    print("Done. Using", (time.time()-start_time)//60, "minutes.")