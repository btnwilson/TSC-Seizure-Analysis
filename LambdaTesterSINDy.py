#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:37:25 2023

@author: wilsobe
"""
import matplotlib.pyplot as plt
import numpy as np
import pyedflib as edf 
import os 
import pysindy as ps
from mlxtend.preprocessing import MeanCenterer
import scipy.cluster.hierarchy as ch 
from sklearn.preprocessing import StandardScaler

os.chdir('/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips')
# create list of seizure clip names
files = []
for file_name in os.listdir():
    files.append(file_name)
files.remove('.DS_Store')
# set number of columns for hankel matrix
rowlength = 128
# create a summed HtH matrix incorperating all seizure clips
HtH = np.zeros((rowlength,rowlength))
for file in files:
    # import data from edf file and gather info needed from headers
    signals, headers, main_header = edf.highlevel.read_edf(file)
    channel1 = signals[0,:]
    channel2 = signals[1,:]
    channel3 = signals[2,:]
    time_interval =  1 / headers[0]['sample_rate']
    time_s = time_interval * len(channel1)
    time_sec = np.linspace(0, time_s, num= len(channel1))
    start = 0
    len_window = len(channel1)
    
    channel1 = np.reshape(channel1, (len(channel1), 1))
    channel1 = StandardScaler().fit_transform(channel1)
    channel1 = np.reshape(channel1, (len(channel2),))
    
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
# creation and visualization of eigen vectors of summed HtH matrix to be used as a univeral basis
eigen_values, V = np.linalg.eig(HtH)

plt.figure()
plt.title('V matrix')
plt.imshow(V, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

# dictates the number of columns from the calculated US matrix are used in SINDy
num_us_retained = 15
# x is a matrix comprised of flattened model coefficent matrices from a linear SINDy model
x = np.zeros((len(files), num_us_retained ** 2))
# Calculate US based on H*V = U*S and then feeding specified number of columns into 
# SINDy package set to develop a linear model. 
num_file = 0
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
    
    channel1 = np.reshape(channel1, (len(channel1), 1))
    channel1 = StandardScaler().fit_transform(channel1)
    channel1 = np.reshape(channel1, (len(channel2),))
    
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
    L_test = np.logspace(-5, 1, num=35)
    scores = []
    avg_sq_list = []
    avg_list = []
    dxdt = []
    for i in L_test:
        threshold = i
        model_test = ps.SINDy(feature_library= identity_lib, optimizer= ps.STLSQ(threshold= threshold))
        model_test.fit(US[:,:num_us_retained], time_interval)
        scores.append(model_test.score(US[:,:num_us_retained], time_interval))
        A_sindy_test = model_test.coefficients()
        sum_vars = 0
        for row in range(np.shape(A_sindy_test)[0]):
            for col in range(np.shape(A_sindy_test)[1]):
                if A_sindy_test[row,col] != 0:
                    sum_vars += A_sindy_test[row,col] ** 2
        num = np.count_nonzero(A_sindy_test)
        if num == 0:
            avg_sq_list.append(0)
            avg_list.append(0)
        else:
            avg_sq_list.append(sum_vars/num)
            avg_list.append(np.sum(A_sindy_test)/num)
    
    for i in range(len(avg_sq_list)-1):
        dxdt.append((avg_sq_list[i+1] - avg_sq_list[i]) / (L_test[i+1] - L_test[i]))
    dxdt_short = [i for i in dxdt if i != 0]


    for i in range(len(dxdt_short)-2):
        if dxdt_short[i+2]/dxdt_short[i] < .20:
            threshold_index = dxdt.index(dxdt_short[i])
    print(threshold_index)
    fig, ax1 = plt.subplots(figsize= (8, 8))
    ax2 = ax1.twinx() 
    ax1.grid()
    line1 = ax1.plot(L_test, scores, color= 'red', label = "Fit Score")
    line2 = ax2.plot(L_test, avg_sq_list, label=  "Number of Coefficients" )
    ax1.set_xlabel("Lambda")
    ax1.set_xscale('log')
    ax1.set_ylabel('Fit Score')
    ax2.set_ylabel('Coefficent Squared Avg')
    plt.xticks(np.logspace(-5, 1, num=35))
    ax1.legend(handles=line1)
    ax2.legend(handles=line2, loc='lower right')
    plt.title(f'{file}')
    plt.show()
    threshold = L_test[threshold_index]
    model = ps.SINDy(feature_library=identity_lib, optimizer= ps.STLSQ(threshold= threshold))
    model.fit(US[:,:num_us_retained], time_interval)
    score = model.score(US[:, :num_us_retained], time_interval)
    print('Fit Score:')
    print(score)
    A_sindy = model.coefficients()
    x[num_file, :] = A_sindy.flatten()
    print(num_file)
    num_file += 1
# center the columns by mean because output of SINDy A matrices were found to have almost
# the same values for all clips. 
x_centered = MeanCenterer().fit_transform(x)
#np.savetxt('A_From_SINDy.csv', x, delimiter=',')
# %%
# import a single clip to test the fit of model using universal V matrix
signals, headers, main_header = edf.highlevel.read_edf("BXD_6_Clip_1.edf")
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
Vt = V.transpose()

# create estimated reconstruction of this clips hankel matrix
est_hankel = np.matmul(US[:,:num_us_retained], Vt[0:num_us_retained, :])

# Recreate signal using diagonal averaging of hankel
b = np.fliplr(est_hankel)
signal = []
diag_num = np.shape(b)[1] - 1
last_num = - np.shape(b)[0] + 1
while diag_num >= last_num:
    avg = b.trace(diag_num) / len(b.diagonal(diag_num))
    signal.append(avg)
    diag_num -= 1
    
 
identity_lib = ps.IdentityLibrary()
L_test = np.logspace(-5, 1, num=30)
scores = []
avg_sq_list = []
avg_list = []
dxdt = []
for i in L_test:
    threshold = i
    model_test = ps.SINDy(feature_library= identity_lib, optimizer= ps.STLSQ(threshold= threshold))
    model_test.fit(US[:,:num_us_retained], time_interval)
    scores.append(model_test.score(US[:,:num_us_retained], time_interval))
    A_sindy_test = model_test.coefficients()
    sum_vars = 0
    for row in range(np.shape(A_sindy_test)[0]):
        for col in range(np.shape(A_sindy_test)[1]):
            if A_sindy_test[row,col] != 0:
                sum_vars += A_sindy_test[row,col] ** 2
    num = np.count_nonzero(A_sindy_test)
    if num == 0:
        avg_sq_list.append(0)
        avg_list.append(0)
    else:
        avg_sq_list.append(sum_vars/num)
        avg_list.append(np.sum(A_sindy_test)/num)

for i in range(len(avg_sq_list)-1):
    dxdt.append((avg_sq_list[i+1] - avg_sq_list[i]) / (L_test[i+1] - L_test[i]))
dxdt_short = [i for i in dxdt if i != 0]


for i in range(len(dxdt_short)-2):
    if dxdt_short[i+2]/dxdt_short[i] < .25:
        threshold_index = dxdt.index(dxdt_short[i])
fig, ax1 = plt.subplots(figsize= (8, 8))
ax2 = ax1.twinx() 
ax1.grid()
line1 = ax1.plot(L_test, scores, color= 'red', label = "Fit Score")
line2 = ax2.plot(L_test, avg_sq_list, label=  "Number of Coefficients" )
ax1.set_xlabel("Lambda")
ax1.set_xscale('log')
ax1.set_ylabel('Fit Score')
ax2.set_ylabel('Coefficent Squared Avg')
plt.xticks(np.logspace(-5, 1, num=35))
ax1.legend(handles=line1)
ax2.legend(handles=line2, loc='lower right')
plt.title(f'{file}')
plt.show()
threshold = L_test[threshold_index]
print(threshold)
model = ps.SINDy(feature_library=identity_lib, optimizer= ps.STLSQ(threshold= threshold))
model.fit(US[:,:num_us_retained], time_interval)
score = model.score(US[:, :num_us_retained], time_interval)
print('Fit Score:')
print(score)
model.print()
A_sindy = model.coefficients()
 
# %%
# graphing the model
plt.figure(figsize=(100,10))
plt.plot(time_sec[:len(signal)], signal, linestyle = '--', color= 'red', linewidth= 5, label= 'Model')
plt.plot(time_sec[:len(signal)], channel1[:len(signal)], color= 'blue', label= "Original Signal")
plt.legend()


# %%
# getting a correlation matrix of the A matrices from SINDy
cc = np.corrcoef(x_centered)

# making a dendrogram 
plt.figure()
linkage_data = ch.linkage(cc, method='ward', metric='euclidean')
dend = ch.dendrogram(linkage_data)
plt.show()

plt.figure()
plt.title('Unsorted Corr Coef Matrix')
plt.imshow(cc, cmap= 'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

# Sorting the heatmap to visualize groups using the first eigenvector of the correlation coefficent matrix
cc_eig_val, cc_eig_vec = np.linalg.eig(cc)

order_eig =  np.argsort(cc_eig_vec[:,0])

cc_int_eig = cc[order_eig, :]
cc_aranged_eig = cc_int_eig[:,order_eig]
plt.figure()
plt.title('Correlation Coef Matrix Sorted By 1st Eigen Vector')
plt.imshow(cc_aranged_eig, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

# %%
# Sorting the heatmap to visualize clusters extracted from dendrogram
cc_int = cc[dend['leaves'][:]]
cc_aranged = cc_int[:,dend['leaves'][:]]
plt.figure()
plt.title('Correlation Coef Matrix Sorted By Dend Groups')
plt.imshow(cc_aranged, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
plt.colorbar()

tree_pieces = ch.cut_tree(linkage_data, n_clusters=4)
# creating lists of file names for each group identified
group_1 = []
group_2 = []
group_3 = []
group_4 = []
index = 0
for i in tree_pieces:
    if i == 0:
        group_1.append(files[index])
    elif i == 1:
        group_2.append(files[index])
    elif i == 2:
        group_3.append(files[index])
    elif i == 3:
        group_4.append(files[index])
    index += 1
    
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)