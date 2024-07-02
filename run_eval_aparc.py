# run the stats evaluation of synthseg of sythetic data
# use cohen's d 
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

# importing libraries
import pandas as pd
import glob
import os

from scipy.stats import ttest_ind

import statsmodels.api as sm



volume_dist=[
    'total intracranial', #0
    'left cerebral white matter', #1
    'left cerebral cortex', #2
    'left lateral ventricle', #3
    'left inferior lateral ventricle',#4
    'left cerebellum white matter',
    'left cerebellum cortex',
    'left thalamus',
    'left caudate',
    'left putamen',
    'left pallidum', #10
    '3rd ventricle',#11
    '4th ventricle',#12
    'brain-stem', #13
    'left hippocampus', #14
    'left amygdala', #15
    'csf', #16
    'left accumbens area',
    'left ventral DC',#18
    'right cerebral white matter',
    'right cerebral cortex',
    'right lateral ventricle',
    'right inferior lateral ventricle',
    'right cerebellum white matter',
    'right cerebellum cortex',
    'right thalamus',
    'right caudate',
    'right putamen',
    'right pallidum',
    'right hippocampus',
    'right amygdala',
    'right accumbens area',
    'right ventral DC',#32
    'ctx-lh-bankssts',#33
    'ctx-lh-caudalanteriorcingulate',#34
    'ctx-lh-caudalmiddlefrontal',
    'ctx-lh-cuneus',
    'ctx-lh-entorhinal',
    'ctx-lh-fusiform',
    'ctx-lh-inferiorparietal',
    'ctx-lh-inferiortemporal',
    'ctx-lh-isthmuscingulate',
    'ctx-lh-lateraloccipital',
    'ctx-lh-lateralorbitofrontal',
    'ctx-lh-lingual',
    'ctx-lh-medialorbitofrontal',
    'ctx-lh-middletemporal',
    'ctx-lh-parahippocampal',
    'ctx-lh-paracentral',
    'ctx-lh-parsopercularis',
    'ctx-lh-parsorbitalis',
    'ctx-lh-parstriangularis',
    'ctx-lh-pericalcarine',
    'ctx-lh-postcentral',
    'ctx-lh-posteriorcingulate',
    'ctx-lh-precentral',
    'ctx-lh-precuneus',
    'ctx-lh-rostralanteriorcingulate',
    'ctx-lh-rostralmiddlefrontal',
    'ctx-lh-superiorfrontal',
    'ctx-lh-superiorparietal',
    'ctx-lh-superiortemporal',
    'ctx-lh-supramarginal',
    'ctx-lh-frontalpole',
    'ctx-lh-temporalpole',
    'ctx-lh-transversetemporal',
    'ctx-lh-insula',#66
    'ctx-rh-bankssts',#67
    'ctx-rh-caudalanteriorcingulate',#68
    'ctx-rh-caudalmiddlefrontal',
    'ctx-rh-cuneus',
    'ctx-rh-entorhinal',
    'ctx-rh-fusiform',
    'ctx-rh-inferiorparietal',
    'ctx-rh-inferiortemporal',
    'ctx-rh-isthmuscingulate',
    'ctx-rh-lateraloccipital',
    'ctx-rh-lateralorbitofrontal',
    'ctx-rh-lingual',
    'ctx-rh-medialorbitofrontal',
    'ctx-rh-middletemporal',
    'ctx-rh-parahippocampal',
    'ctx-rh-paracentral',
    'ctx-rh-parsopercularis',
    'ctx-rh-parsorbitalis',
    'ctx-rh-parstriangularis',
    'ctx-rh-pericalcarine',
    'ctx-rh-postcentral',
    'ctx-rh-posteriorcingulate',
    'ctx-rh-precentral',
    'ctx-rh-precuneus',
    'ctx-rh-rostralanteriorcingulate',
    'ctx-rh-rostralmiddlefrontal',
    'ctx-rh-superiorfrontal',
    'ctx-rh-superiorparietal',
    'ctx-rh-superiortemporal',
    'ctx-rh-supramarginal',
    'ctx-rh-frontalpole',
    'ctx-rh-temporalpole',
    'ctx-rh-transversetemporal',
    'ctx-rh-insula',#100
]

volume_name = [
    'total intracranial', #0
    'cerebral white matter', #1
    'cerebral cortex' ,#2
    'lateral ventricle', #3
    'inferior lateral ventricle',#4
    'cerebellum white matter',
    'cerebellum cortex',
    'thalamus',
    'caudate',
    'putamen',
    'pallidum', #10
    '3rd ventricle' ,#11
    '4th ventricle' ,#12
    'brain-stem' ,#13
    'hippocampus' ,#14
    'amygdala', #15
    'csf', #16
    'accumbens area',
    'ventral DC',#18
    'ctx-bankssts',#33
    'ctx-caudalanteriorcingulate',#34
    'ctx-caudalmiddlefrontal',
    'ctx-cuneus',
    'ctx-entorhinal',
    'ctx-fusiform',
    'ctx-inferiorparietal',
    'ctx-inferiortemporal',
    'ctx-isthmuscingulate',
    'ctx-lateraloccipital',
    'ctx-lateralorbitofrontal',
    'ctx-lingual',
    'ctx-medialorbitofrontal',
    'ctx-middletemporal',
    'ctx-parahippocampal',
    'ctx-paracentral',
    'ctx-parsopercularis',
    'ctx-parsorbitalis',
    'ctx-parstriangularis',
    'ctx-pericalcarine',
    'ctx-postcentral',
    'ctx-posteriorcingulate',
    'ctx-precentral',
    'ctx-precuneus',
    'ctx-rostralanteriorcingulate',
    'ctx-rostralmiddlefrontal',
    'ctx-superiorfrontal',
    'ctx-superiorparietal',
    'ctx-superiortemporal',
    'ctx-supramarginal',
    'ctx-frontalpole',
    'ctx-temporalpole',
    'ctx-transversetemporal',
    'ctx-insula',
]

volume_name1 = [
    # 'total intracranial', #0
    'cerebral white matter', #1
    'cerebral cortex' ,#2
    'lateral ventricle', #3
    'inferior lateral ventricle',#4
    'cerebellum white matter',
    'cerebellum cortex',
    'thalamus',
    'caudate',
    'putamen',
    'pallidum', #10
    '3rd ventricle' ,#11
    '4th ventricle' ,#12
    'brain-stem' ,#13
    'hippocampus' ,#14
    'amygdala', #15
    'csf', #16
    'accumbens area',
    'ventral DC',#18
    'ctx-bankssts',#33
    'ctx-caudalanteriorcingulate',#34
    'ctx-caudalmiddlefrontal',
    'ctx-cuneus',
    'ctx-entorhinal',
    'ctx-fusiform',
    'ctx-inferiorparietal',
    'ctx-inferiortemporal',
    'ctx-isthmuscingulate',
    'ctx-lateraloccipital',
    'ctx-lateralorbitofrontal',
    'ctx-lingual',
    'ctx-medialorbitofrontal',
    'ctx-middletemporal',
    'ctx-parahippocampal',
    'ctx-paracentral',
    'ctx-parsopercularis',
    'ctx-parsorbitalis',
    'ctx-parstriangularis',
    'ctx-pericalcarine',
    'ctx-postcentral',
    'ctx-posteriorcingulate',
    'ctx-precentral',
    'ctx-precuneus',
    'ctx-rostralanteriorcingulate',
    'ctx-rostralmiddlefrontal',
    'ctx-superiorfrontal',
    'ctx-superiorparietal',
    'ctx-superiortemporal',
    'ctx-supramarginal',
    'ctx-frontalpole',
    'ctx-temporalpole',
    'ctx-transversetemporal',
    'ctx-insula',
]
volume_name_aseg = [
    # 'total intracranial', #0
    'cerebral white matter', #1
    'cerebral cortex' ,#2
    'lateral ventricle', #3
    'inferior lateral ventricle',#4
    'cerebellum white matter',
    'cerebellum cortex',
    'thalamus',
    'caudate',
    'putamen',
    'pallidum', #10
    '3rd ventricle' ,#11
    '4th ventricle' ,#12
    'brain-stem' ,#13
    'hippocampus' ,#14
    'amygdala', #15
    'csf', #16
    'accumbens area',
    'ventral DC',#18
]
volume_name_aprac =[
    'ctx-bankssts',#33
    'ctx-caudalanteriorcingulate',#34
    'ctx-caudalmiddlefrontal',
    'ctx-cuneus',
    'ctx-entorhinal',
    'ctx-fusiform',
    'ctx-inferiorparietal',
    'ctx-inferiortemporal',
    'ctx-isthmuscingulate',
    'ctx-lateraloccipital',
    'ctx-lateralorbitofrontal',
    'ctx-lingual',
    'ctx-medialorbitofrontal',
    'ctx-middletemporal',
    'ctx-parahippocampal',
    'ctx-paracentral',
    'ctx-parsopercularis',
    'ctx-parsorbitalis',
    'ctx-parstriangularis',
    'ctx-pericalcarine',
    'ctx-postcentral',
    'ctx-posteriorcingulate',
    'ctx-precentral',
    'ctx-precuneus',
    'ctx-rostralanteriorcingulate',
    'ctx-rostralmiddlefrontal',
    'ctx-superiorfrontal',
    'ctx-superiorparietal',
    'ctx-superiortemporal',
    'ctx-supramarginal',
    'ctx-frontalpole',
    'ctx-temporalpole',
    'ctx-transversetemporal',
    'ctx-insula',
]

def reg_headsize_t(data1, data2, name, rhead, ghead):
    y = data1[name]
    y2 = data2[name]
    # y = np.concatenate(y,y2)
    # y = pd.cat(y,y2)
    y = pd.concat([y, y2])  
    headsize = pd.concat([rhead,ghead])
    model = sm.OLS(y.to_numpy(), headsize.to_numpy())
    results = model.fit()
    residuals = y - results.fittedvalues
    # data[name] = residuals
    return residuals


def reg_headsize(y, headsize, size):
# Fit the ordinary least squares (OLS) regression model
    # model = sm.OLS(y, sm.add_constant(headsize))
    model = sm.OLS(y,  sm.add_constant(headsize.to_numpy()))
    results = model.fit()

    # Access the estimated coefficients (weights) and intercept
    weights = results.params[1:]  # Exclude the intercept
    intercept = results.params[0]  # Intercept
    print(results.params)
    # print(intercept)

    # Manually calculate the predicted values using weights and intercept
    headsize_mean = headsize.mean()
    headsize = headsize - headsize_mean
    # Calculate the residuals (regressed-out values)
    print(len(y))
    print(len(headsize))
    residuals = y - headsize*weights
    return residuals[:size], residuals[size:]

def get_measure(volume_dist, real_df):
    # label 1
    measure=[]
    measure.append(real_df[volume_dist[0]])
    measure.append((real_df[volume_dist[1]]+real_df[volume_dist[19]])/2)
    measure.append((real_df[volume_dist[2]]+real_df[volume_dist[20]])/2)
    measure.append((real_df[volume_dist[3]]+real_df[volume_dist[21]])/2)
    measure.append((real_df[volume_dist[4]]+real_df[volume_dist[22]])/2)
    measure.append((real_df[volume_dist[5]]+real_df[volume_dist[23]])/2)
    measure.append((real_df[volume_dist[6]]+real_df[volume_dist[24]])/2)
    measure.append((real_df[volume_dist[7]]+real_df[volume_dist[25]])/2)
    measure.append((real_df[volume_dist[8]]+real_df[volume_dist[26]])/2)
    measure.append((real_df[volume_dist[9]]+real_df[volume_dist[27]])/2)
    measure.append((real_df[volume_dist[10]]+real_df[volume_dist[28]])/2)
    measure.append(real_df[volume_dist[11]])
    measure.append(real_df[volume_dist[12]])
    measure.append(real_df[volume_dist[13]])
    measure.append((real_df[volume_dist[14]]+real_df[volume_dist[29]])/2)
    measure.append((real_df[volume_dist[15]]+real_df[volume_dist[30]])/2)
    measure.append(real_df[volume_dist[16]])
    measure.append((real_df[volume_dist[17]]+real_df[volume_dist[31]])/2)
    measure.append((real_df[volume_dist[18]]+real_df[volume_dist[32]])/2)
    # label 2

    for i in range(33,67):
            measure.append((real_df[volume_dist[i]]+real_df[volume_dist[i+34]])/2)
    
    return measure

def get_measure_aseg(volume_dist, real_df):
    # label 1
    measure=[]
    measure.append(real_df[volume_dist[0]])
    measure.append((real_df[volume_dist[1]]+real_df[volume_dist[19]])/2)
    measure.append((real_df[volume_dist[2]]+real_df[volume_dist[20]])/2)
    measure.append((real_df[volume_dist[3]]+real_df[volume_dist[21]])/2)
    measure.append((real_df[volume_dist[4]]+real_df[volume_dist[22]])/2)
    measure.append((real_df[volume_dist[5]]+real_df[volume_dist[23]])/2)
    measure.append((real_df[volume_dist[6]]+real_df[volume_dist[24]])/2)
    measure.append((real_df[volume_dist[7]]+real_df[volume_dist[25]])/2)
    measure.append((real_df[volume_dist[8]]+real_df[volume_dist[26]])/2)
    measure.append((real_df[volume_dist[9]]+real_df[volume_dist[27]])/2)
    measure.append((real_df[volume_dist[10]]+real_df[volume_dist[28]])/2)
    measure.append(real_df[volume_dist[11]])
    measure.append(real_df[volume_dist[12]])
    measure.append(real_df[volume_dist[13]])
    measure.append((real_df[volume_dist[14]]+real_df[volume_dist[29]])/2)
    measure.append((real_df[volume_dist[15]]+real_df[volume_dist[30]])/2)
    measure.append(real_df[volume_dist[16]])
    measure.append((real_df[volume_dist[17]]+real_df[volume_dist[31]])/2)
    measure.append((real_df[volume_dist[18]]+real_df[volume_dist[32]])/2)
    # label 2
    return measure

def get_measure_aprac(volume_dist, real_df):
    # label 1
    measure=[]
    # label 2
    
    for i in range(33,67):
        measure.append((real_df[volume_dist[i]]+real_df[volume_dist[i+34]])/2)
    return measure


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['medians'], color=color)
    plt.setp(bp['fliers'], color=color, marker='+')
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


model ='cdpm'

File_paths = './synsegdata/realvol.csv'
File_paths_G = f'./synsegdata/{model}vol.csv'


real_df = pd.read_csv(File_paths)
gen_df = pd.read_csv(File_paths_G)


r_head = real_df['total intracranial']
g_head = gen_df['total intracranial']
Head_size = pd.concat((r_head, g_head))


data_real = get_measure_aprac(volume_dist, real_df)
data_gen = get_measure_aprac(volume_dist, gen_df)

data =[]
data_g = []


for i in range(len(data_real)):
    dat = np.concatenate((data_real[i], data_gen[i]))
    dat_r, dat_g = reg_headsize(dat, Head_size, len(data_real[i]))
    data.append(dat_r)
    data_g.append(dat_g)

data_real = data
data_gen = data_g

plt.figure(figsize=(12, 6))
positions = np.arange(0, (len(volume_name_aprac)))*6

bp_1 = plt.boxplot(data_real, notch=True, sym='+', vert=True, whis=1.5, positions=positions-0.6, widths=1.)
bp_2 = plt.boxplot(data_gen, notch=True, sym='+', vert=True, whis=1.5, positions=positions+0.6, widths=1.)

set_box_color(bp_1, '#D7191C')
set_box_color(bp_2, '#2C7BB6')

plt.plot([], c='#D7191C', label='Real')
plt.plot([], c='#2C7BB6', label='Synthetic')
plt.legend()
volume_name_aprac = [name[len('ctx-'):] if name.startswith('ctx-') else name for name in volume_name_aprac]
plt.xticks(positions, volume_name_aprac, rotation=45, fontsize=4)#, rotation=40,
plt.ylabel('Volume (mm3)')
plt.ylim(0, 32000)
plt.title(f"Cortical Parcellation ROI Measurement of cDPM")

left, right = plt.xlim()  # return the current xlim 
plt.xlim(left-2, right+2) # set the xlim to left, right
plt.tight_layout()

###
df_rois = pd.DataFrame(columns=["ROI", "effect_size"])
### For p-avlue and effect side, and Draw
for i in range(len(data_gen)):
    group1 = data_real[i]
    d_group_m = data_gen[i]
    _, p_value = ttest_ind(group1, d_group_m)
    # _, p_value,_ = ttest_ind(group1, d_group_m)

    # Calculate Cohen's d
    mean_diff = np.mean(group1) - np.mean(d_group_m)
    pooled_std = np.sqrt((np.var(group1) + np.var(d_group_m)) / 2)
    cohen_d = mean_diff / pooled_std

    # plt.text(positions[i]-0.98, data[i].max() + 300, f'{p_value:.2f}', ha='center', c='blue', fontsize=6)
    plt.text(positions[i]-0.98, data_real[i].max() + 500, f'{cohen_d:.2f}', ha='center', fontsize=8)

    ### Create a dictionary with your data
    row = {"ROI": volume_name_aprac[i], "effect_size": f'{cohen_d:.4f}'}
    
    # Append the row to the DataFrame
    df_rois = df_rois._append(row, ignore_index=True)

# Save the DataFrame to a CSV file
df_rois.to_csv("effective_size_aparc.csv", index=False)


plt.savefig(f'{model}_aparc.pdf')