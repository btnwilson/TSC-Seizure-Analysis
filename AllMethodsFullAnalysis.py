#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 08:39:55 2023

@author: wilsobe
"""
import numpy as np
import os 
import pysindy as ps
from mlxtend.preprocessing import MeanCenterer
import functions as f
import pandas as pd
import scipy.signal


path = '/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips'
os.chdir(path)
# create list of seizure clip names
files = []
for file_name in os.listdir():
    files.append(file_name)
files.remove('.DS_Store')
# set number of columns for hankel matrix
columns = 128
# create a summed HtH matrix incorperating all seizure clips
HtH = np.zeros((columns,columns))
hankels = {}
num_welch = 6000
# create a summed HtH matrix incorperating all seizure clips
PSD_vecs = np.zeros((len(files), int(num_welch / 2 + 1)))

for index, file in enumerate(files):
    file_info = f.read_file(file, path)
    raw_signal = file_info['signals'][0]
    
    signal = (raw_signal - np.mean(raw_signal)) /  np.std(raw_signal)
    
    signal_length = file_info['num_measurements']
    time_interval = file_info['frequency']
    
    freq, S = scipy.signal.welch(signal, time_interval, nperseg= num_welch)
    PSD_vecs[index, :]= S / np.sum(S)
    
    hankel = f.hankel(signal, columns)
    hankels[f'{file}'] = hankel
    hankelt = hankel.transpose()
    thisHtH = np.matmul(hankelt, hankel) / signal_length
    HtH = np.add(HtH, thisHtH)
    
eigen_values, V = np.linalg.eig(HtH)

f.heatmap(PSD_vecs, 'PSD Vectors')

PSD_centered = MeanCenterer().fit_transform(PSD_vecs)

PSD_df = pd.DataFrame(PSD_centered, index=files)

PSD_corr_df = (PSD_df.transpose()).corr()

num_groups = 4
PSD_groups = f.clustering(num_groups, PSD_corr_df, 'PSD')

group_list = ['BXD_9', 'BXD_6', 'BXD_17']

f.scatter(PSD_corr_df, group_list, "PSD")

f.heatmap(V, 'Universal V')

x = np.zeros((len(files), columns))

for num, matrix in enumerate(hankels):
    hankel = hankels[matrix]
    US = np.matmul(hankel, V)
    std = np.std(US, axis=0)
    US_centered =  MeanCenterer().fit_transform(US ** 2) / std
    US_avg = np.mean(US_centered, axis=0)
    x[num, :] = US_avg
f.heatmap(x, 'Stacked X matrix baseline method')

x_df = pd.DataFrame(x, index=files)

corr_df = (x_df.transpose()).corr()

groups = f.clustering(num_groups, corr_df, 'Baseline')

f.scatter(corr_df, group_list, 'Baseline')
# %%
num_us_retained = 16
# x is a matrix comprised of flattened model coefficent matrices from a linear SINDy model
x_new = np.zeros((len(files), num_us_retained ** 2))
# Calculate US based on H*V = U*S and then feeding specified number of columns into 
# SINDy package set to develop a linear model. 

for num, matrix in enumerate(hankels):
    hankel = hankels[matrix]
    US = np.matmul(hankel, V)
    identity_lib = ps.IdentityLibrary()
    threshold = .01
    model = ps.SINDy(feature_library=identity_lib, optimizer= ps.STLSQ(threshold= threshold))
    model.fit(US[:,:num_us_retained], time_interval)
    score = model.score(US[:, :num_us_retained], time_interval)
    """
    print('Fit Score:')
    print(score)
    """
    A_sindy = model.coefficients()
    x_new[num, :] = A_sindy.flatten()
x_new_centered = MeanCenterer().fit_transform(x_new)

x_new_df = pd.DataFrame(x_new_centered, index=files)

corr_new_df = (x_new_df.transpose()).corr()

new_groups = f.clustering(num_groups, corr_new_df, "Dynamics Method")

f.scatter(corr_new_df, group_list, "Dynamics Method")
