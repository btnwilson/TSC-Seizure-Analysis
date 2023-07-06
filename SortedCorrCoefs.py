#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:04:50 2023

@author: wilsobe
"""

import matplotlib.pyplot as plt
import numpy as np
import pyedflib as edf 
import os 
import pysindy as ps
from mlxtend.preprocessing import MeanCenterer
import scipy.cluster.hierarchy as ch 


os.chdir('/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips')
files = []
for file_name in os.listdir():
    files.append(file_name)
files.remove('.DS_Store')
rowlength = 128
HtH = np.zeros((rowlength,rowlength))

for file in files:
    signals, headers, main_header = edf.highlevel.read_edf(file)
    channel1 = signals[0,:]
    channel2 = signals[1,:]
    channel3 = signals[2,:]
    time_interval =  1 / headers[0]['sample_rate']
    time_s = time_interval * len(channel1)
    time_sec = np.linspace(0, time_s, num= len(channel1))
    start = 0
    len_window = len(channel1)
    
    
    #create a list of integers used for making a hankel index array
    index_nums = np.array(list(range(0,len(channel1)+1)))
    # hind stands for hankel index needed to have initial values in order to append
    hankel_index = np.zeros((len_window - rowlength, rowlength), dtype = int)
    rows_hankel = len(channel1)-rowlength
    # creates the index array 
    for i in range(0, rows_hankel): 
        hankel_index[i,:] = index_nums[i: i+rowlength]
    # creates the actual hankel matrix with signal values 
    hankel = channel1[hankel_index]
    hankelt = hankel.transpose()
    thisHtH = np.matmul(hankelt, hankel) / len(channel1)
    HtH = np.add(HtH, thisHtH)
eigen_values, V = np.linalg.eig(HtH)

plt.figure()
plt.title('V matrix')
plt.imshow(V, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

num_us_retained = 15

x = np.zeros((len(files), num_us_retained ** 2))
num = 0
for file in files:
    signals, headers, main_header = edf.highlevel.read_edf(file)
    channel1 = signals[0,:]
    channel2 = signals[1,:]
    channel3 = signals[2,:]
    time_interval =  1 / headers[0]['sample_rate']
    time_s = time_interval * len(channel1)
    time_sec = np.linspace(0, time_s, num= len(channel1))
    start = 0
    len_window = len(channel1)
    
    #create a list of integers used for making a hankel index array
    index_nums = np.array(list(range(0,len(channel1)+1)))
    # hind stands for hankel index needed to have initial values in order to append
    hankel_index = np.zeros((len_window - rowlength, rowlength), dtype = int)
    rows_hankel = len(channel1)-rowlength
    # creates the index array 
    for i in range(0, rows_hankel): 
        hankel_index[i,:] = index_nums[i: i+rowlength]
    # creates the actual hankel matrix with signal values 
    hankel = channel1[hankel_index]
    US = np.matmul(hankel, V)
    """
    name = file + '_US_matrix'
    path = '/Users/wilsobe/Desktop/US matrices/'
    plt.figure()
    plt.title(file + ' US matrix')
    plt.imshow(US, cmap= 'hot', interpolation= 'nearest', aspect= 'auto')
    plt.colorbar()
    plt.savefig(path + name + '.png')
    """
    identity_lib = ps.IdentityLibrary()
    threshold = .01
    model = ps.SINDy(feature_library=identity_lib, optimizer= ps.STLSQ(threshold= threshold))
    model.fit(US[:,:num_us_retained], time_interval)
    score = model.score(US[:, :num_us_retained], time_interval)
    print('Fit Score:')
    print(score)
    A_sindy = model.coefficients()
    x[num, :] = A_sindy.flatten()
    num += 1
x_centered = MeanCenterer().fit_transform(x)
covar_mat = np.matmul(x_centered, x_centered.transpose())

plt.figure()
plt.title('Covariance matrix X')
plt.xticks(np.arange(0,len(files), step=5))
plt.yticks(np.arange(0,len(files), step=5))
plt.imshow(covar_mat, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

#from mlxtend.evaluate import scoring
signals, headers, main_header = edf.highlevel.read_edf("BXD_6_Clip_1.edf")
channel1 = signals[0,:]
channel2 = signals[1,:]
channel3 = signals[2,:]
time_interval =  1 / headers[0]['sample_rate']
time_s = time_interval * len(channel1)
time_sec = np.linspace(0, time_s, num= len(channel1))
start = 0
len_window = len(channel1)

rowlength = 128
#create a list of integers used for making a hankel index array
index_nums = np.array(list(range(0,len(channel1)+1)))
# hind stands for hankel index needed to have initial values in order to append
hankel_index = np.zeros((len_window - rowlength, rowlength), dtype = int)
rows_hankel = len(channel1)-rowlength
# creates the index array 
for i in range(0, rows_hankel): 
    hankel_index[i,:] = index_nums[i: i+rowlength]
# creates the actual hankel matrix with signal values 
hankel = channel1[hankel_index]
US = np.matmul(hankel, V) 
Vt = V.transpose()

est_hankel = np.matmul(US[:,:num_us_retained], Vt[0:num_us_retained, :])

b = np.fliplr(est_hankel)
signal = []
diag_num = np.shape(b)[1] - 1
last_num = - np.shape(b)[0] + 1

while diag_num >= last_num:
    avg = b.trace(diag_num) / len(b.diagonal(diag_num))
    signal.append(avg)
    diag_num -= 1
    
    
identity_lib = ps.IdentityLibrary()
threshold = .01
model = ps.SINDy(feature_library=identity_lib, optimizer= ps.STLSQ(threshold= threshold))
model.fit(US[:,:num_us_retained], time_interval)
score = model.score(US[:, :num_us_retained], time_interval)
print('Fit Score:')
print(score)
A_sindy = model.coefficients()
 
# %%
plt.figure(figsize=(100,10))
plt.plot(time_sec[:len(signal)], signal, linestyle = '--', color= 'red', linewidth= 5, label= 'Model')
plt.plot(time_sec[:len(signal)], channel1[:len(signal)], color= 'blue', label= "Original Signal")
plt.legend()
#model_score =  scoring(channel1[:len(signal)],signal)

# %%

cc = np.corrcoef(x_centered)
plt.figure()
linkage_data = ch.linkage(cc, method='ward', metric='euclidean')
dend = ch.dendrogram(linkage_data)
plt.show()
plt.figure()
plt.title('Unsorted Corr Coef Matrix')
plt.imshow(cc, cmap= 'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()
cc_eig_val, cc_eig_vec = np.linalg.eig(cc)

order_eig =  np.argsort(cc_eig_vec[:,0])

cc_int_eig = cc[order_eig, :]
cc_aranged_eig = cc_int_eig[:,order_eig]
plt.figure()
plt.title('Correlation Coef Matrix Sorted By 1st Eigen Vector')
#plt.imshow(np.sort(cc), cmap = 'hot', interpolation= 'nearest', aspect = 'auto')
plt.imshow(cc_aranged_eig, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

# %%
cc_int = cc[dend['leaves'][:]]
cc_aranged = cc_int[:,dend['leaves'][:]]
plt.figure()
plt.title('Correlation Coef Matrix Sorted By Dend Groups')
plt.imshow(cc_aranged, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

tree_pieces = ch.cut_tree(linkage_data, n_clusters=4)