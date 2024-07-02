import os

import imageio
import yaml
import torch
import torchvision
from skimage.transform import resize
from torch.utils.data.dataset import Subset
from torchvision.transforms import (CenterCrop, Compose, RandomHorizontalFlip, Resize, ToTensor)

import numpy as np
import random 
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pandas as pd
import shutil
# from mpi4py import MPI
import nibabel as nib
import pickle5 as pickle

from filelock import FileLock

LOW_THRESHOLD = -1024
HIGH_THRESHOLD = 600

class Protect(FileLock):
    """ Given a file path, this class will create a lock file and prevent race conditions
        using a FileLock. The FileLock path is automatically inferred from the file path.
    """
    def __init__(self, path, timeout=2, **kwargs):
        path = Path(path)
        lock_path = Path(path).parent / f"{path.name}.lock"
        super().__init__(lock_path, timeout=timeout, **kwargs)


video_data_paths_dict = {
    "minerl":       "datasets/minerl_navigate-torch",
    "mazes_cwvae":  "datasets/gqn_mazes-torch",
    "MRI": "/home/Nobias/data/MRI_Train_DPM/",#/home/Nobias/data/hand_orig/adni/img_orig_longitudinal/
}

default_T_dict = {
    "minerl":       500,
    "mazes_cwvae":  300,
    "MRI": 64,
}

default_image_size_dict = {
    "minerl":       64,
    "mazes_cwvae":  64,
    "MRI": 128, # Patch in
}


def create_dataset(dataset_name, T=None, deterministic=False, num_workers=1, return_dataset=False):
    data_path = video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    if dataset_name == "MRI":
        dataset = MRIDataset(train=True, path=data_path, shard=0, num_shards=0, T=T)
    else:
        raise Exception("no dataset", dataset_name)
    return dataset


def get_train_dataset(dataset_name, T=None):
    return create_dataset(
        dataset_name, return_dataset=False, T=T,
        batch_size=None, deterministic=None, num_workers=None
    )


# def get_test_dataset(dataset_name, T=None):
#     data_root = Path(os.environ["DATA_ROOT"]  if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "" else ".")
#     data_path = data_root / video_data_paths_dict[dataset_name]
#     T = default_T_dict[dataset_name] if T is None else T
#
#     if dataset_name == "MRI":
#         # This is not matter as we set all first slice to all-zeros
#         data_path = "/home/groups/kpohl/t1_data/hand_orig/lab_data/img_orig_longitudinal/"
#         print(f'Sampling is based on the first slice from {data_path}')
#         dataset = MRIDataset(train=False, path=data_path, shard=0, num_shards=1, T=T)
#     else:
#         raise Exception("no dataset", dataset_name)
#     dataset.set_test()
#     return dataset


class BaseDataset(Dataset):
    """
    Args:
        path (str): path to the dataset split
    """
    def __init__(self, path, T, resize):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False
        self.resize = resize

    def __len__(self):
        path = self.get_src_path(self.path)
        return len(list(path.iterdir()))

    def __getitem__(self, idx):
        path = self.getitem_path(idx)
        self.cache_file(path)
        try:
            video = self.loaditem(path)
        except Exception as e:
            print(f"Failed on loading {path}")
            raise e
        # video = self.postprocess_video(video)
        # age, sex, site, label = self.get_attributes(idx)
        return video
        return video, age, sex, site, label, str(path) #self.get_video_subsequence(video, self.T)

    def getitem_path(self, idx):
        raise NotImplementedError

    def get_attributes(self, idx):
        raise NotImplementedError

    def loaditem(self, path):
        raise NotImplementedError

    def postprocess_video(self, video):
        raise NotImplementedError

    def cache_file(self, path):
        # Given a path to a dataset item, makes sure that the item is cached in the temporary directory.
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            src_path = self.get_src_path(path)
            with Protect(path):
                shutil.copyfile(str(src_path), str(path))

    @staticmethod
    def get_src_path(path):
        """ Returns the source path to a file. This function is mainly used to handle SLURM_TMPDIR on ComputeCanada.
            If DATA_ROOT is defined as an environment variable, the datasets are copied to it as they are accessed. This function is called
            when we need the source path from a given path under DATA_ROOT.
        """
        if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "":
            # Verify that the path is under
            data_root = Path(os.environ["DATA_ROOT"])
            assert data_root in path.parents, f"Expected dataset item path ({path}) to be located under the data root ({data_root})."
            src_path = Path(*path.parts[len(data_root.parts):]) # drops the data_root part from the path, to get the relative path to the source file.
            return src_path
        return path

    def set_test(self):
        self.is_test = True
        print('setting test mode')

    def get_video_subsequence(self, video, T):
        # print(f'video length with {video.shape}')
        if T is None:
            return video
        if T < len(video):
            # Take a subsequence of the video.
            start_i = 0 if self.is_test else np.random.randint(len(video) - T + 1)
            video = video[start_i:start_i+T]
        assert len(video) == T
        return video



class MRIDataset(BaseDataset):
    def __init__(self, path, train=None, shard=None, num_shards=None, T=None, resize=False):
        super().__init__(path=path, T=T, resize=resize)
        test_datas = np.load('Test400subjects.npy', allow_pickle=True)
        print(f'There are {len(test_datas)} subjects in the test')
        self.fnames = []
        # Introduce 3 datasets: NCANDA, Lab, and ADNI datasets
        ## Adding site label: (in metadata) 00 ucsf, 10, LAB, 01: adni
        # main_path = "/scratch/users/wepeng/data/MRI_Train_DPM/"
        file_adni = path + 'ADNIMERGE.csv'
        file_sri = path + 'sri.csv'
        file_ncanda = path + 'ncanda.csv'

        # df_adni = pd.read_csv(r"/home/groups/kpohl/t1_data/adni_all/ADNI_T1_3_16_2021.csv", header = 0)
        df_adni = pd.read_csv(file_adni, header = 0)
        df_sri = pd.read_csv(file_sri, header = 0)
        df_ncanda = pd.read_csv(file_ncanda, header = 0)

        ### Choose the control
        # df_adni_contrl = df_adni.loc[df_adni['DX_bl'] == 'CN']
        df_adni_contrl1 = df_adni[df_adni['DX_bl'] == 'CN']
        df_sri_contrl1 = df_sri[df_sri['demo_diag'] == 'C']
        df_ncanda_contrl1 = df_ncanda[(df_ncanda['cahalan'] == 'control') | (df_ncanda['cahalan'] == 'moderate')]

        ###remove subjects which are in test
        df_sri_contrl = df_sri_contrl1[~df_sri_contrl1['subject'].str.contains('|'.join(test_datas))]
        df_ncanda_contrl = df_ncanda_contrl1[~df_ncanda_contrl1['subject'].str.contains('|'.join(test_datas))]
        df_adni_contrl = df_adni_contrl1[~df_adni_contrl1['PTID'].str.contains('|'.join(test_datas))]


        # Iterate through the DataFrame and check if the file exists
        rows_to_remove = []
        for index, row in df_sri_contrl.iterrows():
            fn = 'Lab_data/img_orig_longitudinal/'+ row['subject']+'-'+row['visit'].split('_')[0]+'.nii.gz'
            path_here = path + fn
            if not os.path.exists(path_here):
                rows_to_remove.append(index)
        df_sri_contrl = df_sri_contrl.drop(rows_to_remove)

        ###ncanda
        rows_to_remove = []
        for index, row in df_ncanda_contrl.iterrows():
            fn = 'NCANDA/'+ row['subject']+ f"_followup_{row['visit']}y.nii.gz"
            path_here = path + fn
            if not os.path.exists(path_here):
                rows_to_remove.append(index)
        df_ncanda_contrl = df_ncanda_contrl.drop(rows_to_remove)

        ###ADNI
        rows_to_remove = []
        for index, row in df_adni_contrl.iterrows():
            fn = 'ADNI/'+ row['PTID'] +'/'+ row['EXAMDATE'] + '/t1.nii.gz'
            path_here = path + fn
            if not os.path.exists(path_here):
                rows_to_remove.append(index)
        df_adni_contrl = df_adni_contrl.drop(rows_to_remove)

        unique_values_ADNI = df_adni_contrl['PTID'].unique()
        unique_values_SRI = df_sri_contrl['subject'].unique()
        unique_values_NCANDA = df_ncanda_contrl['subject'].unique()



        print(f'there are {df_adni.shape[0]} samples in ADNI , {df_adni_contrl.shape[0]} in control and {unique_values_ADNI.shape[0]} subjects in control.')
        print(f'there are {df_sri.shape[0]} samples in SRI, {df_sri_contrl.shape[0]} in control and {unique_values_SRI.shape[0]} subjects in control.')
        print(f'there are {df_ncanda.shape[0]} samples in NCANDA, {df_ncanda_contrl.shape[0]} in control and {df_ncanda_contrl["subject"].unique().shape[0]} subjects in control.')

        #### To get corresponding subject and metadata:
        ## adni: 'PTID'+ 'EXAMDATE', VISCODE is bl (baseline), m06 (half year visit),...m36(three year); Meta data
        ## ncanda: path shoudl be geting from 'subject' + 'visit', visit==1 mean file name == subject_followup_1y.nii.gz metdadata : visit_age, where is sex?
        ## sri: 'subject'+ 'visit', visit e.g. 20080609_3422_06092008, we only need 20080609 to get the path;
        # file_names_ADNI = ['ADNI/'+df_adni_contrl['PTID'][i]+'/'+ dateTrans(df_adni_contrl['EXAMDATE'][i])+ '/t1.nii.gz' for i in range(df_adni_contrl.shape[0])]

        

        ### make sure the data is at the disc
        file_names_SRI = ['Lab_data/img_orig_longitudinal/'+ df_sri_contrl['subject'][i]+'-'+df_sri_contrl['visit'][i].split('_')[0]+'.nii.gz' for i in df_sri_contrl['visit'].keys()]
        #'ADNI/'+ 'NCANDA/'+ 'SRI/'+
        file_names_NCANDA = ['NCANDA/'+ df_ncanda_contrl['subject'][i]+ f'_followup_{df_ncanda_contrl["visit"][i]}y.nii.gz' if df_ncanda_contrl['visit'][i]> 0 else 'NCANDA/'+df_ncanda_contrl['subject'][i]+'_baseline.nii.gz' for i in df_ncanda_contrl['visit'].keys()]
        file_names_ADNI = ['ADNI/'+ df_adni_contrl['PTID'][i]+'/'+ df_adni_contrl['EXAMDATE'][i]+ '/t1.nii.gz' for i in df_adni_contrl['EXAMDATE'].keys()]


        self.fnames = file_names_ADNI + file_names_NCANDA + file_names_SRI #+ + file_names_ADNI #
        # self.fnames = file_names_ADNI  #+ + file_names_ADNI #
        files_nt_exist = []
        for i in range(len(self.fnames)):
            path_here = path + self.fnames[i]
            # path_here = path + 'ADNI/' + self.fnames[i]
            # print(path_here)
            if not os.path.isfile(path_here):
                # print(f'data {self.fnames[i]} is not exsit')
                files_nt_exist.append(self.fnames[i])
        print(f'Missing {len(files_nt_exist)} MRI samples~')



        # file_names_NCANDA = [fname.replace('followup_0y', 'baseline') for fname in file_names_NCANDA]
        # file_names_NCANDA = [fname for fname in file_names_NCANDA if 'LAB_S00520_20080508.nii.gz' not in fname]
        #
        # file_names_ADNI = [fname for fname in file_names_ADNI if '002_S_0685/2011-02-02/t1.nii.gz' not in fname]
        # file_names_ADNI = [fname for fname in file_names_ADNI if '127_S_0260/2015-04-28/t1.nii.gz' not in fname]
        #
        # file_names_SRI = [fname for fname in file_names_SRI if 'LAB_S01654_20220921' not in fname]
        #

        self.fnames = [fname for fname in self.fnames if fname not in files_nt_exist]

        print(f'There are {len(self.fnames)} samples in the training!')

        # print(subject_names)


    def loaditem(self, path):
        img = nib.load(path)
        data = img.get_fdata()

        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)
        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = (data/max_value) * 2 - 1
        # data = data[5:5+128,:,5:5+128]

        # img = np.ones((144, 176, 144))*data.min()
        
        # img = np.zeros((138, 176, 138))
        img = data
        # # indx_z = torch.randint(img.shape[2]-48, (1,))
        # # # data = data[indx_x:indx_x+64, indx_y:indx_y+64, indx_z:indx_z+64]
        # # img = img[:, :, indx_z:indx_z+48]
        # nan_mask = np.isnan(img) # Remove NaN
        # img[nan_mask] = LOW_THRESHOLD
        # img = np.interp(img, [LOW_THRESHOLD, HIGH_THRESHOLD], [-1,1])
        # valid_plane_i = np.mean(img, (1,2)) != -1 # Remove blank axial planes
        # img = img[valid_plane_i,:,:]
        if self.resize:
            img = resize(img, (128, 128, 128), mode='constant', cval=-1)
        else:
            img2 = np.ones((144, 192, 144))*(-1)
            img2[3:3+138,8:8+176,3:3+138] = img
            img = img2
        data = th.from_numpy(img[None,:,:,:]).float()
        if self.resize:
            assert data.shape == (1, 128, 128, 128)
        else:
            assert data.shape == (1, 144, 192, 144)
        # print(f'this is shape {data.shape}')
        return data
    
    def loaditem2(self, path):
        img = nib.load(path)
        data = img.get_fdata()

        max_value = np.percentile(data, 98)
        data = np.where(data <= max_value, data, max_value)
        data = data/max_value

        # img = np.ones((144, 176, 138))*data.min()
        # img[3:3+138] = data

        ## Random crop a volume
        indx_x = torch.randint(data.shape[0]-64, (1,))
        indx_y = torch.randint(data.shape[1]-64, (1,))
        indx_z = torch.randint(data.shape[2]-128, (1,))
        data = data[indx_x:indx_x+64, indx_y:indx_y+64, indx_z:indx_z+64]
        # data = img[:, :, indx_z:indx_z+128]
        data = th.from_numpy(data[None,:,:,:]).float()
        return data

    def getitem_path(self, idx):
        return self.path / self.fnames[idx]

    def get_attributes(self, idx):
        # there are age sex etc information in the data
        # 5, 6 are For age and gender, 1 for label, label in range []
        return self.All_datas[idx][5], self.All_datas[idx][6], self.All_datas[idx][7], self.All_datas[idx][1]

    import torch.nn.functional as F
    def postprocess_video(self, video):
        o, h, w, t = video.shape
        images = []
        # This will be used when I directly generate a 3D volume
        img = F.interpolate(video[None], size= 128)
        return img.squeeze(0)
        # The following are for slices
        for i in range(t):
            img = F.interpolate(video[None, :,:,:,i], size= 128)
            images.append(img)
        return th.cat(images) #-1 + 2 * (video.permute(0, 3, 1, 2).float()/255)

    def __len__(self):
        return len(self.fnames)


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def get_default_dataset_paths():
    with open("datasets.yml") as yaml_file:
        read_data = yaml.load(yaml_file, Loader=yaml.FullLoader)

    paths = {}
    for i in range(len(read_data)):
        paths[read_data[i]["dataset"]] = read_data[i]["path"]

    return paths


def train_val_split(dataset, train_val_ratio):
    indices = list(range(len(dataset)))
    split_index = int(len(dataset) * train_val_ratio)
    train_indices, val_indices = indices[:split_index], indices[split_index:]
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(dataset, val_indices)
    return train_dataset, val_dataset


def get_data_loaders(
    dataset_name,
    img_size,
    batch_size,
    get_flipped=False,
    train_val_split_ratio=0.95,
    custom_dataset_path=None,
    num_workers=4,
    drop_last=True,
    shuffle=True,
    get_val_dataloader=False,
):

    train_dataset = create_dataset(dataset_name)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        sampler=None,
        shuffle=shuffle,
        batch_size=batch_size,
        drop_last=drop_last
    )
    if get_val_dataloader:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, num_workers=num_workers,
            sampler=None,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last
        )
    else:
        val_loader = None

    return train_loader, val_loader


    B, T, C, H, W = x.shape
    time_t = time_t.view(B, 1).expand(B, T)
    indicator_template = th.ones_like(x[:, :, :1, :, :])
    obs_indicator = indicator_template * condition_mask
    x = th.cat([x*(1-condition_mask) + x0*condition_mask,obs_indicator],dim=2)
if __name__ == "__main__":
    get_data_loaders("MRI", 176, 1)
