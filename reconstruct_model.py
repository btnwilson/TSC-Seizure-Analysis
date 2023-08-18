#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 10:41:08 2023

@author: wilsobe
"""

import os
import numpy as np
import pickle
from scipy.integrate import odeint
from autoencoder import full_network
from training import create_feed_dictionary
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.preprocessing import PolynomialFeatures
import copy
import pysindy as ps
from mlxtend.preprocessing import MeanCenterer
from matplotlib.lines import Line2D
import pandas as pd
import functions as f
tf.disable_v2_behavior()

split_data = pickle.load(open('/Users/wilsobe/Desktop/Python Files/Real Data/Autoencoder Groups 1.pkl', 'rb'))
training_data = {}
seizure_range = pickle.load(open('/Users/wilsobe/Desktop/Python Files/Real Data/Seizure Range Dict.pkl', 'rb'))
colors = []
x_auto = np.array([])
hankel_leng = {}
for file in split_data['training']['group_1']:
    print(file)
    US = np.loadtxt('/Users/wilsobe/Desktop/Python Files/Real Data/Standardized US CSV/' + file + '.csv', delimiter= ',')
    print(len(US))
    x_auto = np.append(x_auto, US)
    hankel_leng[f'{file}'] = len(US)
    if file in seizure_range.keys():
        for i in range(0, len(US)):
            if i > seizure_range[file][0] and i < seizure_range[file][1]:
                colors.append('red')
            else:
                colors.append('blue')
    else:
        for i in range(0, len(US)):
            colors.append('blue')
colors.pop(-1)
x_auto = x_auto.reshape(int(len(x_auto) / 128), 128)
dx_auto = (x_auto[1:,:] - x_auto[:-1,:]) / .00390625
ddx_auto = dx_auto[1:,:] - dx_auto[:-1,:] / .00390625
training_data['x'] = x_auto[:len(dx_auto), :15]
training_data['dx'] = dx_auto[:len(dx_auto), :15]
training_data['ddx'] = ddx_auto[:, :15]
# %%
data_path = '/Users/wilsobe/Desktop/Python Files/Real Data/SindyAutoencoders-master/src/500 Epoch 2nd order (7 vars)/'
save_name = 'TSC_Seizures'
params = pickle.load(open('/Users/wilsobe/Desktop/Python Files/Real Data/SindyAutoencoders-master/src/500 Epoch 2nd order (7 vars)/TSC_Seizures_2023_08_16_15_24_40_179965_params.pkl', 'rb'))
params['save_name'] = data_path + save_name

autoencoder_network = full_network(params)
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

tensorflow_run_tuple = ()
for key in autoencoder_network.keys():
    tensorflow_run_tuple += (autoencoder_network[key],)


# %%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, data_path + save_name)
    test_dictionary = create_feed_dictionary(training_data, params)
    tf_results = sess.run(tensorflow_run_tuple, feed_dict=test_dictionary)

test_set_results = {}
for i,key in enumerate(autoencoder_network.keys()):
    test_set_results[key] = tf_results[i]

# %%    
sample_list = random.sample(range(0, len(test_set_results['z'])), k=10)
def syst_ode(y, t, A):
    x0, x1, x2 = y
    vec = np.array([[x0], [x1], [x2]]).reshape((1,3))
    poly = PolynomialFeatures(2)
    feature_lib = poly.fit_transform(vec)
    
    dydt = np.matmul(A.transpose(), feature_lib.reshape(10,))
    return dydt

for lst in sample_list:
    A = test_set_results["sindy_coefficients"]
    for m in range(0, A.shape[0]):
        for n in range(0, A.shape[1]):
            if abs(A[m,n]) < .01:
                A[m,n] = 0
    y0 = test_set_results['z'][lst]
    t = np.linspace(0, 40, 20000)
    sol_ode = odeint(syst_ode, y0, t, args=(A,))
    
    plt.figure(figsize=(100,10))
    plt.title(str(y0))
    plt.plot(sol_ode[:,0])
    plt.xlabel('Time')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(str(y0))
    ax.plot3D(sol_ode[:,0], sol_ode[:,1], sol_ode[:,2])
    ax.set_xlabel('Sol 1')
    ax.set_ylabel('Sol 2')
    ax.set_zlabel('Sol 3')
    fig.show()
    
reduced_dim_data = copy.deepcopy(test_set_results['z'])    

legend_elements = [Line2D([0], [0], marker='o', color= 'w', label= 'Seizure', markerfacecolor= 'red'),
                 Line2D([0], [0], marker='o', color= 'w', label= 'No Seizure', markerfacecolor= 'blue')]
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Reduced Dim Scatter')
ax.scatter3D(reduced_dim_data[:,0], reduced_dim_data[:,1], reduced_dim_data[:,2], alpha= .25, s= .5, c=colors)
fig.legend(handles = legend_elements)
fig.show()


centered_reduced_dim_data = MeanCenterer().fit_transform(reduced_dim_data)

identity_lib = ps.PolynomialLibrary(2)
threshold = .1
model = ps.SINDy(feature_library=identity_lib, optimizer= ps.STLSQ(threshold= threshold))
model.fit(centered_reduced_dim_data, 1/256)
score = model.score(centered_reduced_dim_data, 1/256)
A_centered = model.coefficients()
# %%
def syst_ode_cent(y, t, A_centered):
    x0, x1, x2 = y
    vec = np.array([[x0], [x1], [x2]]).reshape((1,3))
    poly = PolynomialFeatures(2)
    feature_lib = poly.fit_transform(vec)
    dydt = np.matmul(A_centered, feature_lib.reshape(10,))
    return dydt

#for i in range(0,10):
for lst in sample_list:
    #y0 = [random.randrange(-2,2),random.randrange(-2,2),random.randrange(-2,2)]
    y0 = centered_reduced_dim_data[lst]
    t = np.linspace(0, 100, 30000)
    sol_ode = odeint(syst_ode_cent, y0, t, args=(A_centered,))
    
    plt.figure(figsize=(100,10))
    plt.title('ODEint' + str(y0))
    plt.plot(sol_ode[:,0])
    plt.xlabel('Time')
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_title(str(y0))
    ax.plot3D(sol_ode[:,0], sol_ode[:,1], sol_ode[:,2], linewidth= .1)
    ax.set_xlabel('Sol 1')
    ax.set_ylabel('Sol 2')
    ax.set_zlabel('Sol 3')
    fig.show()
# %%
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Reduced Dim Centered Scatter')
ax.scatter3D(centered_reduced_dim_data[:,0], centered_reduced_dim_data[:,1], centered_reduced_dim_data[:,2], alpha= .25, s= .5, c=colors)
fig.legend(handles = legend_elements)
fig.show()

z_scatter = pd.DataFrame(centered_reduced_dim_data)
z_scatter['color'] = colors
seizure_scatter = z_scatter[z_scatter['color'].isin(['red'])]
non_seizure_scatter = z_scatter[z_scatter['color'].isin(['blue'])]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Reduced Dim Centered Seizure Scatter')
ax.scatter3D(seizure_scatter[0], seizure_scatter[1], seizure_scatter[2], alpha= .25, s= .5, c= seizure_scatter['color'], label= 'Seizure')
fig.legend()
fig.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title('Reduced Dim Centered Non Seizure Scatter')
ax.scatter3D(non_seizure_scatter[0], non_seizure_scatter[1], non_seizure_scatter[2], alpha= .25, s= .5, c= non_seizure_scatter['color'], label= 'Non Seizure')
fig.legend()
fig.show()
# %%
model_equations = model.equations()
for i in range(0,3):
    model_equations[i] = model_equations[i].replace(' x',' * x')
    model_equations[i] = model_equations[i].replace(' 1 ',' * 1')
    model_equations[i] = model_equations[i].replace('^', '**')
    equation = f"x_dot_{i}_lam"
    exec(equation + " = lambda x0,x1,x2: " + model_equations[i])

x0, x1, x2 = np.meshgrid(np.arange(-3, 4, 1),np.arange(-3, 4, 1), np.arange(-3, 4, 1))
fig = plt.figure()
ax = plt.axes(projection='3d')
field = ax.quiver(x0, x1, x2, x_dot_0_lam(x0,x1,x2), x_dot_1_lam(x0,x1,x2), x_dot_2_lam(x0,x1,x2), length=.005)
plt.show()

































x_decode = test_set_results['x_decode']

path = '/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips'
files = []
os.chdir('/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips')
for file_name in os.listdir():
    files.append(file_name)
files.remove('.DS_Store')
# set number of columns for hankel matrix
columns = 128
# create a summed HtH matrix incorperating all seizure clips
HtH = np.zeros((columns,columns))

# create a summed HtH matrix incorperating all seizure clips

for index, file in enumerate(files):
    file_info = f.read_file(file, path)
    raw_signal = file_info['signals'][0]
    
    signal = (raw_signal - np.mean(raw_signal)) /  np.std(raw_signal)
    
    signal_length = file_info['num_measurements']
    time_interval = file_info['frequency']
    
    hankel = f.hankel(signal, columns)
    hankelt = hankel.transpose()
    thisHtH = np.matmul(hankelt, hankel) / signal_length
    HtH = np.add(HtH, thisHtH)
    
eigen_values, V = np.linalg.eig(HtH)
pos = 0
dec_reconst_sig = {}
for i, file in enumerate(split_data['training']['group_1']):
    if i + 1 == len(split_data['training']['group_1']):
        reconst_us = x_decode[pos: (pos + hankel_leng[file] - 1)]
    else:
        reconst_us = x_decode[pos: (pos + hankel_leng[file])]
    pos += hankel_leng[file]
    
    reconst_hankel = np.matmul(reconst_us, V[:,:15].transpose())
    b = np.fliplr(reconst_hankel)
    signal = []
    diag_num = np.shape(b)[1] - 1
    last_num = - np.shape(b)[0] + 1

    while diag_num >= last_num:
        avg = b.trace(diag_num) / len(b.diagonal(diag_num))
        signal.append(avg)
        diag_num -= 1
    
    dec_reconst_sig[file] = signal
    
    
for key in dec_reconst_sig.keys():
    file_info = f.read_file(key, path)
    raw_signal = file_info['signals'][0]
    true_signal = (raw_signal - np.mean(raw_signal)) /  np.std(raw_signal)
    
    plt.figure(figsize=(100,10))
    plt.title(key + ' True Standardized signal')
    plt.plot(true_signal)
    plt.show()
    
    plt.figure(figsize=(100,10))
    plt.title(key + ' Autoencoder Reconstruction')
    plt.plot(dec_reconst_sig[key])
    plt.show()





































