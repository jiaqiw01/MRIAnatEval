from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
import glob
from pathlib import Path
import nibabel as nib
import torch


class BrainMRIDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.is_test = False
        self.files = list(self.path.glob('*.nii.gz'))
        print("Total training files is: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        res = self.loaditem(path)
        return res
    
    def loaditem(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        assert data.shape == (138, 176, 138)

        max_value = np.percentile(data, 95)
        min_value = np.percentile(data, 5)
        data = np.where(data <= max_value, data, max_value)
        data = np.where(data <= min_value, 0., data)
        data = (data/max_value) * 2 - 1
        # data = data[5:5+128,:,5:5+128]

        # img = np.ones((144, 176, 144))*data.min()
        # img = np.zeros((144, 192, 144))
        # img[3:3+138,8:8+176,3:3+138] = data
        # indx_z = torch.randint(img.shape[2]-48, (1,))
        # # data = data[indx_x:indx_x+64, indx_y:indx_y+64, indx_z:indx_z+64]
        # img = img[:, :, indx_z:indx_z+48]
        data = torch.from_numpy(img[None,:,:,:]).float()
        # print(data.shape)
        return data, -1
    


class Volume_Dataset(Dataset):

    def __init__(self, data_dir, mode='train', fold=0, num_class=0):
        self.sid_list = []
        self.data_dir = data_dir
        self.num_class = num_class

        for item in glob.glob(self.data_dir+"*.npy"):
            self.sid_list.append(item.split('/')[-1])

        self.sid_list.sort()
        self.sid_list = np.asarray(self.sid_list)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, valid_index = list(kf.split(self.sid_list))[fold]
        print("Fold:", fold)
        if mode=="train":
            self.sid_list = self.sid_list[train_index]
        else:
            self.sid_list = self.sid_list[valid_index]
        print("Dataset size:", len(self))

        self.class_label_dict = dict()
        if self.num_class > 0: # conditional
            FILE = open("class_label.csv", "r")
            FILE.readline() # header
            for myline in FILE.readlines():
                mylist = myline.strip("\n").split(",")
                self.class_label_dict[mylist[0]] = int(mylist[1])
            FILE.close()

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = np.load(self.data_dir+self.sid_list[idx])
        class_label = self.class_label_dict.get(self.sid_list[idx], -1) # -1 if no class label
        return img[None,:,:,:], class_label