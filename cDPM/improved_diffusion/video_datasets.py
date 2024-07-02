import os
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from pathlib import Path
import shutil
from mpi4py import MPI
import nibabel as nib
import pandas as pd

from .test_util import Protect


video_data_paths_dict = {
    "minerl":       "datasets/minerl_navigate-torch",
    "mazes_cwvae":  "datasets/gqn_mazes-torch",
    "MRI": "/scratch/project_2001654/Wpeng/data/MRI_Three/MRI_High/",#/home/Nobias/data/hand_orig/adni/img_orig_longitudinal/
}

default_T_dict = {
    "minerl":       500,
    "mazes_cwvae":  300,
    "MRI": 139,
}

default_image_size_dict = {
    "minerl":       64,
    "mazes_cwvae":  64,
    "MRI": 144,
}


def load_data(dataset_name, batch_size, T=None, deterministic=False, num_workers=1, return_dataset=False):
    data_path = video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    shard = MPI.COMM_WORLD.Get_rank()
    num_shards = MPI.COMM_WORLD.Get_size()
    if dataset_name == "minerl":
        data_path = os.path.join(data_path, "train")
        dataset = MineRLDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "mazes_cwvae":
        data_path = os.path.join(data_path, "train")
        dataset = GQNMazesDataset(data_path, shard=shard, num_shards=num_shards, T=T)
    elif dataset_name == "MRI":
        dataset = MRIDataset(train=True, path=data_path, shard=shard, num_shards=num_shards, T=T)
    else:
        raise Exception("no dataset", dataset_name)
    if return_dataset:
        return dataset
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(not deterministic), num_workers=num_workers, drop_last=True
        )
        while True:
            yield from loader


def get_train_dataset(dataset_name, T=None):
    return load_data(
        dataset_name, return_dataset=False, T=T,
        batch_size=None, deterministic=None, num_workers=None
    )


def get_test_dataset(dataset_name, T=None):
    if dataset_name == "mazes":
        raise Exception('Deprecated dataset.')
    data_root = Path(os.environ["DATA_ROOT"]  if "DATA_ROOT" in os.environ and os.environ["DATA_ROOT"] != "" else ".")
    data_path = data_root / video_data_paths_dict[dataset_name]
    T = default_T_dict[dataset_name] if T is None else T
    if dataset_name == "minerl":
        data_path = os.path.join(data_path, "test")
        dataset = MineRLDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "mazes_cwvae":
        data_path = os.path.join(data_path, "test")
        dataset = GQNMazesDataset(data_path, shard=0, num_shards=1, T=T)
    elif dataset_name == "MRI":
        dataset = MRIDataset(train=False, path=data_path, shard=0, num_shards=1, T=T)
    else:
        raise Exception("no dataset", dataset_name)
    dataset.set_test()
    return dataset


class BaseDataset(Dataset):
    """ The base class for our video datasets. It is used for datasets where each video is stored under <dataset_root_path>/<split>
        as a single file. This class provides the ability of caching the dataset items in a temporary directory (if
        specified as an environment variable DATA_ROOT) as the items are read. In other words, every time an item is
        retrieved from the dataset, it will try to load it from the temporary directory first. If it is not found, it
        will be first copied from the original location.

        This class provides a default implementation for __len__ as the number of file in the dataset's original directory.
        It also provides the following two helper functions:
        - cache_file: Given a path to a dataset file, makes sure the file is copied to the temporary directory. Does
        nothing unless DATA_ROOT is set.
        - get_video_subsequence: Takes a video and a video length as input. If the video length is smaller than the
          input video's length, it returns a random subsequence of the video. Otherwise, it returns the whole video.
        A child class should implement the following methods:
        - getitem_path: Given an index, returns the path to the video file.
        - loaditem: Given a path to a video file, loads and returns the video.
        - postprocess_video: Given a video, performs any postprocessing on the video.

    Args:
        path (str): path to the dataset split
    """
    def __init__(self, path, T):
        super().__init__()
        self.T = T
        self.path = Path(path)
        self.is_test = False

    def __len__(self):
        path = self.get_src_path(self.path)
        return len(list(path.iterdir()))

    def __getitem__(self, idx):
        path = self.getitem_path(idx)
        self.cache_file(path)
        try:
            video = self.loaditem(path) # 1, 138, 176, 138
        except Exception as e:
            print(f"Failed on loading {path}")
            raise e
        # print(video.shape)
        video = th.permute(video, (3, 0, 2, 1)) # None, 138, 1, 176, 131
        print(video.shape)
        # print("Finishend get item, shape is ", video.shape)
        # assert video.shape == (131, 1, 176, 144)
        # video = self.postprocess_video(video)
        return video, {} #self.get_video_subsequence(video, self.T)

    def getitem_path(self, idx):
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

## For our MRI dataset
class MRIDataset(BaseDataset):
    def __init__(self, train, path, shard, num_shards, T):
        super().__init__(path=path, T=T)
        self.fnames = []
        # img_names = os.listdir(self.path)
        img_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(self.path) for f in fn]
        print(f'There are {len(img_names)} samples in this folder')
        img_names.sort()
        for i in range(len(img_names)):
            image_path = os.path.join(self.path, img_names[i])
            # print(image_path)
            if image_path[-3:] == '.gz' :
                self.fnames.append(image_path)

        self.fnames = self.fnames[shard::num_shards]
        print(f"Loading {len(self.fnames)} files (MRI dataset).")

    def loaditem(self, path):
        img = nib.load(path)
        data = img.get_fdata()

        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)
        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = (data/max_value) * 2 - 1
        # data = data / max_value
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

        img2 = np.ones((144, 176, 131))*(-1)
        # img2 = np.ones((144, 192, 144))*(-1)
        img2[3:3+138, :, 1:] = img[:, :, 4:134]
        # img = img2

        data = th.from_numpy(img2[None,:,:,:]).float()
        # print(data.shape)
        # assert data.shape == (1, 138, 176, 131)
        # print(f'this is shape {data.shape}')
        return data

    def getitem_path(self, idx):
        return self.path / self.fnames[idx]

    import torch.nn.functional as F
    def postprocess_video(self, video):
        o, t, h, w = video.shape
        images = []
        for i in range(t):
            # img = F.interpolate(video[None, :,:,:,i], size= 128)
            img = video[None, :, i, :, :]
            images.append(img)
        res = th.cat(images)
        print("Finished concat images")
        print("shape is ", res.shape)
        return res #-1 + 2 * (video.permute(0, 3, 1, 2).float()/255)

    def __len__(self):
        return len(self.fnames)