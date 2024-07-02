import math
from collections import OrderedDict
import json
import numpy as np
from skimage.transform import resize
from resnet3D import resnet50, resnet34, resnet101
import os
import torch
import pandas as pd
import torch.nn as nn
import nibabel as nib

# class MRITestDataset(torch.utils.data.Dataset):
#     def __init__(self, npy_file=None, path=None):
#         super().__init__()
#         test_datas = np.load(npy_file, allow_pickle=True)
#         print(f'There are {len(test_datas)} subjects in the test')
#         self.fnames = []
#         # Introduce 3 datasets: NCANDA, Lab, and ADNI datasets
#         ## Adding site label: (in metadata) 00 ucsf, 10, LAB, 01: adni
#         # main_path = "/scratch/users/wepeng/data/MRI_Train_DPM/"
#         file_adni = path + 'ADNIMERGE.csv'
#         # file_sri = path + 'sri.csv'
#         file_ncanda = path + 'ncanda.csv'

#         # df_adni = pd.read_csv(r"/home/groups/kpohl/t1_data/adni_all/ADNI_T1_3_16_2021.csv", header = 0)
#         df_adni = pd.read_csv(file_adni, header = 0)
#         # df_sri = pd.read_csv(file_sri, header = 0)
#         df_ncanda = pd.read_csv(file_ncanda, header = 0)

#         ### Choose the control
#         # df_adni_contrl = df_adni.loc[df_adni['DX_bl'] == 'CN']
#         df_adni_contrl1 = df_adni[df_adni['DX_bl'] == 'CN']
#         # df_sri_contrl1 = df_sri[df_sri['demo_diag'] == 'C']
#         df_ncanda_contrl1 = df_ncanda[(df_ncanda['cahalan'] == 'control') | (df_ncanda['cahalan'] == 'moderate')]

#         ###remove subjects which are in test
#         # df_sri_contrl = df_sri_contrl1[df_sri_contrl1['subject'].str.contains('|'.join(test_datas))]
#         df_ncanda_contrl = df_ncanda_contrl1[df_ncanda_contrl1['subject'].str.contains('|'.join(test_datas))]
#         df_adni_contrl = df_adni_contrl1[df_adni_contrl1['PTID'].str.contains('|'.join(test_datas))]
#         print(df_ncanda_contrl.shape[0], df_adni_contrl.shape[0])

#         # Iterate through the DataFrame and check if the file exists
#         # rows_to_remove = []
#         # for index, row in df_sri_contrl.iterrows():
#         #     fn = 'SRI/'+ row['subject']+'-'+row['visit'].split('_')[0]+'.nii.gz'
#         #     path_here = path + fn
#         #     if not os.path.exists(path_here):
#         #         rows_to_remove.append(index)
#         # df_sri_contrl = df_sri_contrl.drop(rows_to_remove)

#         ###ncanda
#         rows_to_remove = []
#         ncanda_files = []
#         for index, row in df_ncanda_contrl.iterrows():
#             for i in [1, 2]:
#                 fn = 'NCANDA/'+ row['subject']+ f"_followup_{i}y.nii.gz" # chosed the first
#                 path_here = path + fn
#                 if os.path.exists(path_here) and path_here not in ncanda_files:
#                     # rows_to_remove.append(index)
#                     ncanda_files.append(path_here)
#         # df_ncanda_contrl = df_ncanda_contrl.drop(rows_to_remove)
#         print(f"Found {len(ncanda_files)} ncanda file....")
#         ###ADNI
#         # rows_to_remove = []
#         adni_files = []
#         counts = {} # choose maximum 2 each subject
#         for index, row in df_adni_contrl.iterrows():
#             fn = 'ADNI/'+ row['PTID'] +'/'+ row['EXAMDATE'] + '/t1.nii.gz'
#             path_here = path + fn
#             if row["PTID"] not in counts:
#                 counts[row["PTID"]]  = 0
#             elif counts[row["PTID"]] == 2:
#                 continue
#             if os.path.exists(path_here) and path_here not in adni_files:
#                 # rows_to_remove.append(index)
#                 adni_files.append(path_here)
#                 counts[row["PTID"]] += 1
#         # df_adni_contrl = df_adni_contrl.drop(rows_to_remove)
#         print(f"Found {len(adni_files)} adni file....")
        
#         unique_values_ADNI = df_adni_contrl['PTID'].unique()
#         # unique_values_SRI = df_sri_contrl['subject'].unique()
#         unique_values_NCANDA = df_ncanda_contrl['subject'].unique()
#         # print(unique_values_NCANDA)
#         print(f'there are {df_adni.shape[0]} samples in ADNI , {df_adni_contrl.shape[0]} in control and {unique_values_ADNI.shape[0]} subjects in control.')
#         # print(f'there are {df_sri.shape[0]} samples in SRI, {df_sri_contrl.shape[0]} in control and {unique_values_SRI.shape[0]} subjects in control.')
#         print(f'there are {df_ncanda.shape[0]} samples in NCANDA, {df_ncanda_contrl.shape[0]} in control and {df_ncanda_contrl["subject"].unique().shape[0]} subjects in control.')
#         self.fnames = adni_files + ncanda_files
#         print(f"Found {len(self.fnames)} test files!!!")
#         #### To get corresponding subject and metadata:
#         ## adni: 'PTID'+ 'EXAMDATE', VISCODE is bl (baseline), m06 (half year visit),...m36(three year); Meta data
#         ## ncanda: path shoudl be geting from 'subject' + 'visit', visit==1 mean file name == subject_followup_1y.nii.gz metdadata : visit_age, where is sex?
#         ## sri: 'subject'+ 'visit', visit e.g. 20080609_3422_06092008, we only need 20080609 to get the path;
#         # file_names_ADNI = ['ADNI/'+df_adni_contrl['PTID'][i]+'/'+ dateTrans(df_adni_contrl['EXAMDATE'][i])+ '/t1.nii.gz' for i in range(df_adni_contrl.shape[0])]

        

#         ### make sure the data is at the disc
#         # file_names_SRI = ['SRI/'+ df_sri_contrl['subject'][i]+'-'+df_sri_contrl['visit'][i].split('_')[0]+'.nii.gz' for i in df_sri_contrl['visit'].keys()]
#         #'ADNI/'+ 'NCANDA/'+ 'SRI/'+
#         # file_names_NCANDA = ['NCANDA/'+ df_ncanda_contrl['subject'][i]+ f'_followup_1y.nii.gz' for i in unique_values_NCANDA]
#         # file_names_ADNI = ['ADNI/'+ df_adni_contrl['PTID'][i]+'/'+ df_adni_contrl['EXAMDATE'][i]+ '/t1.nii.gz' for i in df_adni_contrl['EXAMDATE'].keys()]
        
#         # # print(len(file_names_SRI))
#         # print(len(file_names_NCANDA))
#         # print(file_names_NCANDA[:5])
#         # print(len(file_names_ADNI))

#         # self.fnames = file_names_ADNI + file_names_NCANDA #+ + file_names_ADNI #
#         # # self.fnames = file_names_ADNI  #+ + file_names_ADNI #
#         # files_nt_exist = []
#         # for i in range(len(self.fnames)):
#         #     path_here = path + self.fnames[i]
#         #     # path_here = path + 'ADNI/' + self.fnames[i]
#         #     # print(path_here)
#         #     if not os.path.isfile(path_here):
#         #         # print(f'data {self.fnames[i]} is not exsit')
#         #         files_nt_exist.append(self.fnames[i])
#         # print(f'A total of {len(self.fnames)} mri test data!!!!!')

#     def __getitem__(self, idx):
#         path = self.fnames[idx]
#         img = nib.load(path)
#         data = img.get_fdata()

#         max_value = np.percentile(data, 95)
#         min_value = np.percentile(data, 5)
#         data = np.where(data <= max_value, data, max_value)
#         data = np.where(data <= min_value, 0., data)
#         # data = (data/max_value) * 2 - 1
#         data = data / max_value
#         # data = data[5:5+128,:,5:5+128]

#         # img = np.ones((144, 176, 144))*data.min()
        
#         # img = np.zeros((138, 176, 138))
#         img = data
#         if self.resize:
#             img = resize(img, (128, 128, 128), mode='constant', cval=-1)
#         else:
#             img2 = np.ones((144, 192, 144))*(-1)
#             # img2 = np.ones((144, 192, 144))*(-1)
#             img2[3:3+138,8:8+176,3:3+138] = img
#             img = img2
#         data = torch.from_numpy(img[None,:,:,:]).float()
#         if self.resize:
#             assert data.shape == (1, 128, 128, 128)
#         else:
#             assert data.shape == (1, 144, 192, 144)
#         # print(f'this is shape {data.shape}')
#         return data, -1

#     def __len__(self):
#         return len(self.fnames)





KEYS = ['checkpoint_root', "exp_name", "ckpt_fname", 'latent_dim', 'step', 'save_dir', "feature_extractor_ckpt"]

class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)

def load_config(fname: str):
    with open(fname, 'r') as fr:
        info = json.load(fr)
    return info

def check_config(conf: dict) -> bool:
    keys = list(conf.keys())
    for k in KEYS:
        if k not in keys or conf[k] is None:
            return False, f'Key {k} not found'
    dest = os.path.join(conf['checkpoint_root'], conf['exp_name'], conf['ckpt'].replace("STEP", str(conf['step'])))
    if not os.path.exists(dest) or not os.path.exists(conf['feature_extractor_ckpt']):
        return False, "Checkpoint file not found"
    return True, ''

def trim_state_dict_name(state_dict):
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            # remove prefix
            state_dict[k[len("module."):]] = state_dict[k]
            del state_dict[k]
    return state_dict

def load_model(model, ckpt):
    print(f"Trying to load {ckpt}")
    train_on_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if train_on_gpu:
        model= nn.DataParallel(model)
    model.to(device)
    if train_on_gpu:
        content = torch.load(ckpt)
        if type(content) is dict and 'model' in content:
            model.load_state_dict(torch.load(ckpt)['model'])
        elif 'state_dict' in content:
            model.load_state_dict(torch.load(ckpt)['state_dict'])
        else:
            model.load_state_dict(torch.load(ckpt))
    else: # TODO: check on cpu
        state_dict = torch.load(ckpt, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v
        model.load_state_dict(new_state_dict)
    return model



def load_samples(sample_dir):
    f = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if 'nii' in f]
    return f

# def get_feature_extractor101(ckpt_path):
#     model = resnet101(shortcut_type='B')
#     model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
#                                    Flatten()) # (N, 512)
#     # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
#     ckpt = torch.load("./pretrain/resnet_101.pth")
#     ckpt = trim_state_dict_name(ckpt["state_dict"])
#     model.load_state_dict(ckpt) # No conv_seg module in ckpt
#     model = nn.DataParallel(model).cuda()
#     model.eval()
#     print("Feature extractor weights loaded")
#     return model

def get_feature_extractor(resnet_version, ckpt_path):
    '''
        Please add your own feature extractors
    '''
    if resnet_version == 50:
        model = resnet50(shortcut_type='B')
    elif resnet_version == 101:
        model = resnet101(shortcut_type='B')
    else:
        raise NotImplementedError
    
    assert os.path.exists(ckpt_path), "Ops, feature extractor ckpt not found!"

    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                   Flatten()) # (N, 512)
    # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
    ckpt = torch.load(ckpt_path)
    ckpt = trim_state_dict_name(ckpt["state_dict"])
    model.load_state_dict(ckpt) # No conv_seg module in ckpt
    model = nn.DataParallel(model).cuda()
    model.eval()
    print("Feature extractor weights loaded")
    return model


# def get_feature_extractor50(ckpt_path):
#     model = resnet50(shortcut_type='B')
#     model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
#                                    Flatten()) # (N, 512)
#     # ckpt from https://drive.google.com/file/d/1399AsrYpQDi1vq6ciKRQkfknLsQQyigM/view?usp=sharing
#     ckpt = torch.load(ckpt_path)
#     ckpt = trim_state_dict_name(ckpt["state_dict"])
#     model.load_state_dict(ckpt) # No conv_seg module in ckpt
#     model = nn.DataParallel(model).cuda()
#     model.eval()
#     print("Feature extractor weights loaded")
#     return model



if __name__ == "__main__":
    ds = MRITestDataset()