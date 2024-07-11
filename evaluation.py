"""
    Generative Model Evaluations
    Metrics: 
    3D MSSSIM
    3D FID using Resnet50/101/others
    3D MMD in image space & feature space (resnet50)
    PCA
"""
import os
import time
from tqdm import tqdm
from argparse import ArgumentParser
from scipy import linalg
import torch.nn as nn
import seaborn as sns
import torch
import numpy as np
import pandas as pd
import torchvision
from torch.nn import functional as F
from torch.autograd import Variable
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from collections import OrderedDict
from dataset import MRIDataset
from utils import *
from pytorch_ssim import msssim_3d
from fid_score import post_process, get_activations_from_dataloader, calculate_frechet_distance
# from torch_two_sample.torch_two_sample.statistics_diff import MMDStatistic
from torch.autograd import Variable
import matplotlib.pyplot as plt
# pretrained 3D resnet from https://github.com/Tencent/MedicalNet
from resnet3D import resnet50
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from PIL import Image

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

train_on_gpu = torch.cuda.is_available()
device = torch.device('cuda' if train_on_gpu else 'cpu')
# device = torch.device('cpu')

MODEL_NAME = "VAE_LDM"

NUM_SAMPLES = 400

PATH = '/scratch/project_2001654/Wpeng/data/MRI_Three/MRI_High/'
BATCH_SIZE = 2
NUM_WORKERS = 2

# TODO: Use your own way of loading model
model_config = {
    'HA-GAN':  {
        "name": "HA-GAN",
        "checkpoint_root": "./HAGAN/checkpoint/",
        "exp_name": "HA_GAN_144_b16",
        "ckpt_fname": "G_iterSTEP.pth",
        "latent_dim": 1024,
        "step": 50000,
        "img_size": 144,
        "save_dir": "./BrainSample/HA-GAN_retrained",
        "feature_extractor_ckpt": "./pretrain/resnet_50.pth"
        },
    'alpha-WGAN': {
        "name": "alpha-WGAN",
        "checkpoint_root": "./AE_Models/checkpoint/",
        "exp_name": "alphaWGAN_144_b8",
        "ckpt_fname": "G_iterSTEP.pth",
        "latent_dim": 1000,
        "step": 165000,
        "img_size": 144,
        "save_dir": "./BrainSample/alpha-WGAN_retrained",
        "feature_extractor_ckpt": "./pretrain/resnet_50.pth"
    },
    'vae-GAN': {
        "name": "vae-GAN",
        "checkpoint_root": "./AE_Models/checkpoint/",
        "exp_name": "vaeGAN_144_b4",
        "ckpt_fname": "G_VG_ep_STEP.pth",
        "latent_dim": 1000,
        "step": 130,
        "img_size": 144,
        "save_dir": "./BrainSample/VAE-GAN_retrained",
        "feature_extractor_ckpt": "./pretrain/resnet_50.pth"
    }
}

def infinite_dataloder(dl):
    while True:
        for x in iter(dl): yield x

def save_slice_3d(sample: np.ndarray, model_name, save_dir, fname="combined.png", idx=72, dim=0):
    '''
        Saves slices in all 3 dimensions
    '''
    print("=========> Saving Slice in 3 Axis <==========")
    # plt.rcParams['axes.facecolor'] = 'black'
    plt.rcParams['savefig.facecolor'] = 'black'
    sl1 = (sample[60, :, :] + 1) * 0.5
    sl2 = (sample[:, 96, :] + 1) * 0.5
    sl3 = (sample[:, :, 72] + 1) * 0.5
    fig, axes = plt.subplots(1, 3, figsize=(19, 14))
    axes[0].imshow(sl1, cmap='gray')
    axes[0].axis('off')
    axes[1].imshow(sl2, cmap='gray')
    axes[1].axis('off')
    axes[2].imshow(sl3, cmap='gray')
    axes[2].axis('off')
    plt.savefig(os.path.join(save_dir, f"{model_name}_combined.png"), bbox_inches='tight', dpi=300)
    os.makedirs(save_dir, exist_ok=True)
    Image.fromarray(np.uint8(sl1*255)).convert("L").save(os.path.join(save_dir, f"{model_name}_saggital.png"))
    Image.fromarray(np.uint8(sl2*255)).convert("L").save(os.path.join(save_dir, f"{model_name}_coronal.png"))
    Image.fromarray(np.uint8(sl3*255)).convert("L").save(os.path.join(save_dir, f"{model_name}_axial.png"))
    print("========== Finished Saving ==============")

def sample_slice_visualization(fake_samples: list, model_name, cut_coords=[30, 50, 100, 130], num_vis=10, output_dir='./Sample_Vis'):
    output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for i in range(0, len(fake_samples), len(fake_samples)//num_vis):
        featmask = nib.load(fake_samples[i])
        # featmask = nib.Nifti1Image(featmask, affine = np.eye(4))
        dest = os.path.join(output_dir, f"{i}_vis.png")
        disp = plotting.plot_anat(featmask,cut_coords=cut_coords,draw_cross=False,annotate=False,black_bg=True,display_mode='x', output_file=dest)
    print("Finished exporting visualization slices")


def pca_tsne(real: np.ndarray, fake: np.ndarray, model_name: str, mode=3, n_pca=100, n_tsne=2, save_dir="./results/pca_tsne", feature=True, resnet=101):
    '''
        This function calculates the pca and/or tsne embedding for both image space / feature space
        arguments:
        mode: 1 - pca only; 2 - tsne only; 3 - pca + tsne (default)
        feature - False (image space), True (feature space)
        resnet: the version of ResNet extractor / or any other feature extractor you use, please modify accordingly
    '''
    
    save_dir = save_dir+'_'+str(resnet)
    os.makedirs(save_dir, exist_ok=True)
    if mode == 1: # pca only!
        print("=========> Computing PCA only <=============")
        pca= PCA(n_components=n_pca)
        results_real = pca.fit_transform(real)
        results_fake = pca.fit_transform(fake)
        save_name = "pca_only"
    elif mode == 2: # tsne only!
        print("=========> Computing TSNE only <=============")
        tsne = TSNE(n_components=n_tsne)
        results_real = tsne.fit_transform(real)
        results_fake = tsne.fit_transform(fake)
        save_name = "tsne_only"
    elif mode == 3: # pca + tsne
        pca_ = PCA(n_components=n_pca)
        pca_result_real = pca_.fit_transform(real)
        pca_result_fake = pca_.fit_transform(fake)
        tsne = TSNE(n_components=n_tsne, perplexity=5)
        results_real = tsne.fit_transform(pca_result_real)
        results_fake = tsne.fit_transform(pca_result_fake)
        save_name = "pca_tsne"
    fig = plt.figure()
    plt.rcParams['savefig.facecolor'] = 'white'
    for i in range(results_real.shape[0]):
        real_tx = results_real[i, 0]
        real_ty = results_real[i, 1]
        fake_tx = results_fake[i, 0]
        fake_ty = results_fake[i, 1]
        plt.scatter(real_tx, real_ty, c='red')
        plt.scatter(fake_tx, fake_ty, c='blue')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    if feature:
        fig.savefig(f"{save_dir}/{save_name}_feature_comparison_{model_name}_plot.png") 
    else:
        fig.savefig(f"{save_dir}/{save_name}_volume_comparison_{model_name}_plot.png")
    print("=========> Finished PCA TSNE <=============")


def sample_generation_gan(model: torch.nn.Module, model_config: dict, num_samples=400) -> list:
    '''
        Generates GAN samples by loading the model and its config, please replace this with your own model generation
        model: a generator model
    '''
    dest = model_config['save_dir']
    if os.path.isdir(dest):
        inp = input("Sample folder detected, want to regenerate? 0: No; 1: Yes")
        if int(inp) == 0:
            return
    # sample generation
    os.makedirs(dest, exist_ok=True)
    ckpt = os.path.join(model_config['checkpoint_root'], model_config['exp_name'], model_config['ckpt_fname'].replace("STEP", str(model_config['step'])))
    model = load_model(model, ckpt)
    print("Ckpt loaded")
    model.eval()
    fake_samples = []
    total_time = 0
    for i in tqdm(range(num_samples)):     
        with torch.no_grad():
            # print("==== Generating from random noise ====")
            dim = model_config['latent_dim']
            start = time.time()
            noise = torch.randn((1, dim)).to(device)
            fake_images = model(noise)
            dur = int(time.time() - start)
            total_time += dur
            for b in range(fake_images.shape[0]):
                featmask = np.squeeze(fake_images[b]) # Keep the image in range [-1, 1]
                if train_on_gpu:
                    featmask = featmask.data.cpu().numpy()
                # print(featmask.shape)
                featmask = nib.Nifti1Image(featmask, affine = np.eye(4))
                save_name = os.path.join(dest, f'generated_sample_{i}.nii.gz')
                fake_samples.append(save_name)
                nib.save(featmask, save_name)
    print(f"===== Finished generating  =====")
    print(f"Total time: {total_time}")
    print(f"Avg time per sample: {total_time / float(num_samples)}")
    return fake_samples



def evaluate_fid_mmd(fake_samples: list, real_sample: list, extractor_ckpt: str, save=True, save_name="./results/fid/HAGAN/", load=False, load_path=None, resnet=101) -> float:
    """
        Calculate FID + MMD in feature space
        fake_samples: list of sample file names
        load: if you have already computed fid and saved it somewhere
        resnet: the version of your feature extractor e.g., 50 for ResNet50
        Code adapted from https://github.com/batmanlab/HA-GAN
    """
    print("====> Starting FID Evaluation <===")
    if not load:
        model = get_feature_extractor(resnet, extractor_ckpt).to(device)
        torch.cuda.empty_cache()
        batch_size = 2
        num_samples = len(fake_samples)
        dims=2048
        pred_arr = np.empty((num_samples, dims))

        for i, sample in enumerate(tqdm(fake_samples)):
            with torch.no_grad():
                x_rand = sample
                x_rand = torch.from_numpy(x_rand).unsqueeze(0).unsqueeze(0).float().to(device)
                pred = model(x_rand)
                if device == torch.device('cuda'):
                    pred_arr[i, :] = pred.cpu().numpy()
                else:
                    pred_arr[i, :] = pred.numpy()
        if save:
            os.makedirs(save_name, exist_ok=True)
            np.save(os.path.join(save_name,f"resnet_{str(resnet)}_fake.npy"), pred_arr)

        act_real = get_activations_from_dataloader(model, real_sample, device, dims=dims, num_samples=num_samples, batch_size=batch_size)
        if save:
            np.save(os.path.join(save_name, f"resnet_{str(resnet)}_real.npy"), act_real)
    else:
        if load_path is None and not os.path.exists(os.path.join(save_name,f"resnet_{str(resnet)}_fake.npy")):
            print("Cannot find feature vector file, please specify in load_name argument!")
            exit()
        
        pred_arr = np.load(os.path.join(save_name, "resnet_fake.npy")) if load_path is None else np.load(load_path)
        act_real = np.load(os.path.join(save_name, "resnet_real.npy")) if load_path is None else np.load(load_path)

    m_fake, s_fake = post_process(pred_arr)
    m_real, s_real = post_process(act_real)


    fid_score = calculate_frechet_distance(m_fake, s_fake, m_real, s_real)
    print("======Finished fid score calculation! Starting MMD calculation======")
    mmd_score = compute_mmd_feature(act_real, pred_arr)
    print("FID score is: ", fid_score)
    print("MMD score is: ", mmd_score)
    return fid_score, mmd_score

def evaluate_msssim(fake_samples: list, real_samples: list, model_name: str, num_iter=10, normalize=False, window_size=11, save=False, dest=None) -> float:
    """
        3D Multi-Scale SSIM calculation between 1 fake & 1 randomly selected real volume
        normalize: to avoid numerical instability, may or may not happen so default to False
        iter: number of iterations (may be very slow)
    """
    print("====> Starting 3D SSIM Evaluation <====")
    fake_ssim = 0.0
    count = 0
    for j in range(num_iter):
        for i in tqdm(range(len(fake_samples))):
            # get one fake sample
            vol1 = Variable(torch.from_numpy(fake_samples[i]).unsqueeze(0).unsqueeze(0)) # (144, 192, 144)
            idx = np.random.randint(0, len(fake_samples), size=1)
            real = Variable(torch.from_numpy(real_samples[idx]).unsqueeze(0))
            assert vol1.shape == real.shape
            vol1 = vol1.to(device)
            vol2 = real.to(device)
            res = msssim_3d(vol1, vol2, normalize=False)
            # a sanity check, may encounter nan value here if normalize=False
            if not np.isnan(res):
                fake_ssim += res
                count += 1
            else:
                print("got a nan value... skipping it")

    avg_fake_ssim = fake_ssim/count
    print("Fake volume avg ssim: ", avg_fake_ssim)

    if save and dest:
        with open(os.path.join(dest, {model_name}.txt), 'w') as fw:
            fw.write(f"Fake volume avg ssim: {avg_fake_ssim}")

    return avg_fake_ssim

def msssim_real(real_samples: list):
    '''
        Calculating msssim for real volumes
    '''
    real_ssim = 0
    for k in range(1000):
        idx1, idx2 = np.random.choice(np.arange(len(real_samples)), size=2, replace=False)
        # Get 2 real volumes
        vol1 = torch.from_numpy(real_samples[idx1]).unsqueeze(0).unsqueeze(0).to(device)
        vol2 = torch.from_numpy(real_samples[idx2]).unsqueeze(0).unsqueeze(0).to(device)
        msssim = msssim_3d(vol1,vol2, normalize=True)
        real_ssim  += msssim
    avg_real_ssim = real_ssim/(k+1)
    print("Real volume avg ssim: ", avg_real_ssim)
    return avg_real_ssim

def compute_mmd_feature(act_real: np.ndarray, act_fake: np.ndarray, num_iter=10):
    """
        Compute feature space MMD statistics between real & fake sample
        Take 2 real & 2 fake samples at a time and compute the distance
        Can take a long time if large num_iter
        Adapted from: https://github.com/cyclomon/3dbraingen
    """
    print("====> Starting feature-space MMD evaluation <====")

    sample1 = torch.from_numpy(act_real)
    sample2 = torch.from_numpy(act_fake)

    distmean = 0
    for k in range(num_iter):
        for i in range(0, sample1.shape[0], 2):
            B = 2
            x = sample1[i:i+2]
            y = sample2[i:i+2]
            xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())
            beta = (1./(B*B))
            gamma = (2./(B*B)) 
            Dist = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
            # print(Dist)
            distmean += Dist.cpu().numpy()
    mean_feature_mmd = distmean/((k+1)*(i+1))
    return mean_feature_mmd

def pad_vol(img, shape=(144, 192, 144), pad_val=-1):
    '''
        Pad volume to desired shape, in torch format
    '''
    if img.shape[1:] == shape:
        return img
    target = torch.ones((1, 144, 192, 144)) * pad_val
    d_, h_, w_ = shape
    b, d, h, w = img.shape
    d_pad = (d_ - d) // 2
    h_pad = (h_ - h) // 2
    w_pad = (w_ - w) // 2
    target[:, d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = img
    return target.float().permute(0, 3, 2, 1)

def pad_vol_np(img, shape=(144, 192, 144), pad_val=-1):
    '''
        Pad volume in numpy format
    '''
    target = np.ones((144, 192, 144)) * pad_val
    d_, h_, w_ = shape
    d, h, w = img.shape
    d_pad = (d_ - d) // 2
    h_pad = (h_ - h) // 2
    w_pad = (w_ - w) // 2
    target[d_pad:d_pad+d, h_pad:h_pad+h, w_pad:w_pad+w] = img
    return target


def evaluate_mmd_image(fake_samples: list, real_samples: list, model_name: str, num_iter=5):
    """
        Calculate MMD on flattened volume
        Take one fake volume and one random real volume
        Returned an avg value over num_iter iterations
        Default to batch size 2
        Code adapted from https://github.com/cyclomon/3dbraingen
    """
    meanarr = []
    for s in range(num_iter):
        distmean = 0.0
        for i in tqdm(range(len(fake_samples)-1)):
            # load a fake volume
            fakes = []
            for j in range(i, i+2):
                x = Variable(torch.from_numpy(fake_samples[j]).float()).unsqueeze(0).to(device)
                fakes.append(x)
            x = torch.cat(fakes, dim=0) # 2, 144, 192, 144
            idx1, idx2 = np.random.choice(np.arange(len(fake_samples)), size=2, replace=False)
            real1 = Variable(torch.from_numpy(real_samples[idx1]).float()).unsqueeze(0)
            real2 = Variable(torch.from_numpy(real_samples[idx2]).float()).unsqueeze(0)
            y = torch.cat([real1, real2], dim=0).to(device)
            # y = Variable(real).to(device).squeeze() # 2, 144, 192, 144
            # print(y.shape)
            assert x.shape == y.shape
            b, d, h, w = y.shape    
            B = y.shape[0]
            x = x.reshape(2, 144*192*144)
            y = y.reshape(2, 144*192*144)
            xx, yy, zz = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

            beta = (1./(B*B))
            gamma = (2./(B*B)) 

            Dist = beta * (torch.sum(xx)+torch.sum(yy)) - gamma * torch.sum(zz)
            # print(Dist)
            distmean += Dist.cpu().numpy()

        print('Mean:'+str(distmean/(i+1)), flush=True)
        meanarr.append(distmean/(i+1))

    meanarr = np.array(meanarr)
    print('Total_mean:'+str(np.mean(meanarr))+' STD:'+str(np.std(meanarr)))
    return np.mean(meanarr)


def prepare_real_samples(test_dir="./test400_MRI"):
    '''
        Collects all real testing samples and return a list of np.ndarray
        test_dir: the directory in which all test nii.gz files are stored
        Please replace with your own code for preprocessing
    '''
    arr = []
    for f in os.listdir(test_dir):
        dest = os.path.join(test_dir, f)
        data = nib.load(dest).get_fdata()

        # TODO: this part is the preprocessing for our dataset
        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)

        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = (data/max_value) * 2 - 1
        vol = np.ones((144, 192, 144)) * (-1)
        vol[3:3+138,8:8+176,3:3+138] = data
        arr.append(np.expand_dims(vol, 0)) # 1, 144, 192, 144
    return np.concatenate(arr, axis=0)

def prepare_fake_samples(dir):
    '''
        Collects all fake samples in dir and return a list of np.ndarray
        Please repleace with your own code & preprocessing
    '''
    samples = load_samples(dir)
    samples.sort()
    fake_sample_arr = []
    for s in samples:
        try:
            vol = nib.load(s).get_fdata()
        except:
            print(f"Failed to load sample {s}")
            continue
        if vol.shape != (144, 192, 144): # masked
            vol = pad_vol_np(vol)
            vol = np.transpose(vol, (2, 1, 0))
        if vol.min() >= 0:
            vol = vol / vol.max() * 2 - 1
        if np.isnan(vol).any():
            print(f"Found some nan values in smaple {s}")
            nan_mask = np.isnan(vol) # Remove NaN
            vol[nan_mask] = -1
        fake_sample_arr.append(np.expand_dims(vol, 0))
    return np.concatenate(fake_sample_arr, axis=0)



TEST_PARAMS = {
    'num_iter': 5,
    'model_name': "HAGAN",
    'feature_extractor_version': 101,
    'feature_extractor_ckpt': './pretrain/resnet_101.pth',
    'save_root': './results/',
    'pca_tsne_mode': 3, # 1: pca only, 2: tsne only, 3: pca + tsne
    'num_pca': 50, # TODO: modify according to your mode
    'num_tsne': 2, # TODO: modify according to your mode
}

def prepare_pca_tsne(params: dict, real_samples: list, fake_samples: list):
    '''
        Load feature vector file & flatten the samples
    '''
    dest = os.path.join(params['save_root'], 'fid', params['model_name'])
    rn = params['feature_extractor_version']
    try:
        feature_real = np.load(os.path.join(dest, f'resnet_{str(rn)}_real.npy'))
        feature_fake = np.load(os.path.join(dest, f'resnet_{str(rn)}_fake.npy'))
    except:
        print("Ops, feature vector file not found")
        exit()

    flatten_real = np.stack([s.flatten() for s in real_samples], axis=0)
    flatten_fake = np.stack([s.flatten() for s in fake_samples], axis=0)
    return feature_real, feature_fake, flatten_real, flatten_fake


def eval_all(fake_samples: list, real_samples: list, params: dict=TEST_PARAMS, metric_list=['FID', 'MMD', 'SSIM', 'PCA_TSNE']):
    '''
        An example of evaluating multiple metrics
    '''
    results = {}
    if 'SSIM' in metric_list:
        print("====> 3D SSIM")
        res = evaluate_msssim(fake_samples, real_samples, params['model_name'])
        results['MS_SSIM'] = res

    if 'MMD' in metric_list:
        print("====> 3D MMD")  
        res = evaluate_mmd_image(fake_samples, real_samples, params['model_name'], num_iter=params['num_iter'])
        results['MMD_Image'] = res

    if 'FID' in metric_list:
        print("====> 3D FID + MMD on feature space")
        default_save_dir = os.path.join(params['save_root'], 'fid', params['model_name'])
        fid, mmd = evaluate_fid_mmd(fake_samples, real_samples, params['feature_extractor_ckpt'], save=True, save_name=default_save_dir)
        results['FID'] = fid
        results['MMD_feature'] = mmd

    if 'PCA_TSNE' in metric_list or 'PCA' in metric_list or 'TSNE' in metric_list:
        print("====> Qualitative Visualization")
        feature_real, feature_fake, flatten_real, flatten_fake = prepare_pca_tsne(params, real_samples, fake_samples)
        if 'PCA_TSNE' in metric_list:
            default_save_dir = os.path.join(params['save_root'], 'visualization_pca_tsne', params['model_name'])
            pca_tsne(feature_real, feature_fake, params['model_name'], mode=3, n_pca=params['num_pca'], n_tsne=params['num_tsne'], save_dir=default_save_dir, feature=True, resnet=params['feature_extractor_version'])
            pca_tsne(flatten_real, flatten_fake, params['model_name'], mode=3, n_pca=params['num_pca'], n_tsne=params['num_tsne'], save_dir=default_save_dir, feature=False)

        if 'PCA' in metric_list:
            print("====> PCA ONLY Visualization")
            default_save_dir = os.path.join(params['save_root'], 'visualization_pca_only', params['model_name'])
            pca_tsne(feature_real, feature_fake, params['model_name'], mode=1, n_pca=params['num_pca'], n_tsne=params['num_tsne'], save_dir=default_save_dir, feature=True, resnet=params['feature_extractor_version'])
            pca_tsne(flatten_real, flatten_fake, params['model_name'], mode=1, n_pca=params['num_pca'], n_tsne=params['num_tsne'], save_dir=default_save_dir, feature=False)

        if 'TSNE' in metric_list:
            print("====> TSNE ONLY Visualization")
            default_save_dir = os.path.join(params['save_root'], 'visualization_tsne_only', params['model_name'])
            pca_tsne(feature_real, feature_fake, params['model_name'], mode=2, n_pca=params['num_pca'], n_tsne=params['num_tsne'], save_dir=default_save_dir, feature=True, resnet=params['feature_extractor_version'])
            pca_tsne(flatten_real, flatten_fake, params['model_name'], mode=2, n_pca=params['num_pca'], n_tsne=params['num_tsne'], save_dir=default_save_dir, feature=False)
            
    dest = os.path.join(params['save_dir_evaluations'], f'{params['model_name']}_metrics.json')
    with open(dest, 'w') as fw:
        json.dump(fw, results, indent=4)
        print(f"Saved metrics in {dest}")


if __name__ == "__main__":
    models = ['masked_samples', "HA-GAN_retrained", 'alpha-WGAN_retrained', 'VAE-GAN_retrained',  '2stage_retrained', 'AE_LDM']
    real_samples = prepare_real_samples()

    for MODEL_NAME in models:
        print(f"~~~~~~~~~ Model: {MODEL_NAME} ~~~~~~~~~~~")
        fake_samples = prepare_fake_samples(f"./BrainSample/{MODEL_NAME}")
        TEST_PARAMS = {
            'num_iter': 5,
            'model_name': MODEL_NAME,
            'feature_extractor_version': 101,
            'feature_extractor_ckpt': './pretrain/resnet_101.pth',
            'save_root': './results/',
            'pca_tsne_mode': 3, # 1: pca only, 2: tsne only, 3: pca + tsne
            'num_pca': 50, # TODO: modify according to your mode
            'num_tsne': 2, # TODO: modify according to your mode
        }
        eval_all(fake_samples, real_samples, TEST_PARAMS)