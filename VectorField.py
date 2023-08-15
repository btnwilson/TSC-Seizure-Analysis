#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:25:56 2023

@author: wilsobe
"""

import sksfa
import matplotlib.pyplot as plt
import numpy as np
import pyedflib as edf 
import os 
from sklearn.preprocessing import PolynomialFeatures
import pysindy as ps
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import axes3d
from matplotlib.widgets import Slider
from itertools import product

file = input('Enter file name: ')
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
location = find(file, "Tsc1_EEG")
print(location)
signals, headers, main_header = edf.highlevel.read_edf(location)
channel1 = signals[0,:]
channel2 = signals[1,:]
channel3 = signals[2,:]
locationsplit = location.split('/')
time_interval =  1 / headers[0]['sample_rate']
time_s = time_interval * len(channel1)
time_sec = np.linspace(0, time_s, num= len(channel1))
start = 0
len_window = len(channel1)


plt.figure(figsize=(100, 10))
plt.title(locationsplit[3] + ",    " + file)
plt.xlabel('Time (s)')
plt.ylabel('microvolts')
plt.plot(time_sec[start:len_window], channel1[start:len_window])
plt.show()


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

U, S, Vt = np.linalg.svd(hankel, full_matrices=False)

US = U @ np.diag(S)
pf = PolynomialFeatures(degree=2)
expanded_us = pf.fit_transform(US[:, :10])
num_features = 3
sfa = sksfa.SFA(n_components= num_features)
output_features = sfa.fit_transform(expanded_us)

"""
for col in range(num_features):
    plt.figure(figsize= (100,10))
    plt.title(f"SFA Feature {col + 1}") 
    plt.plot(time_sec[:len(output_features)], output_features[:, col])
"""    

variables = []
for i in range(num_features):
    string = f"x{i}"
    variables.append(string)
var_string = ""
for i, var in enumerate(variables):
    if i == 0:
        var_string += var
    else:
        var_string = var_string + ", " + var
identity_lib_squared = ps.IdentityLibrary() + ps.IdentityLibrary() * ps.IdentityLibrary()
threshold = .25
psmodel = ps.SINDy(feature_library= identity_lib_squared, optimizer= ps.STLSQ(threshold= threshold))
#psmodel = ps.SINDy(optimizer= ps.STLSQ(threshold= threshold))
sindy_model = psmodel.fit(output_features[:, 0:num_features], time_interval)
score = psmodel.score(output_features[:, 0:num_features], time_interval)
print(f"Score: {score}")
#psmodel.print()
A = psmodel.coefficients()

model_equations = psmodel.equations()

for i in range(num_features):
    equation = f"x_dot_{i}_lam"
    exec(equation + " = lambda " + var_string +": " + model_equations[i].replace(' x',' * x'))

x1, x2 = np.meshgrid(np.arange(-20, 20, 1),np.arange(-20, 20, 1))
x0 = 5

fig = plt.figure()

field = plt.quiver(x1, x2, x_dot_1_lam(x0,x1,x2), x_dot_2_lam(x0,x1,x2))
slider_color = 'White'

axis_position = plt.axes([0.075, 0.025, 0.65, 0.05],
                         facecolor = slider_color)
slider_position = Slider(axis_position,'x0 value', -20, 20) 

def update(val):
    x0 = slider_position.val
    field.set_(plt.quiver(x1, x2, x_dot_1_lam(x0,x1,x2), x_dot_2_lam(x0,x1,x2)))
    fig.canvas.draw()
slider_position.on_changed(update)
plt.show()











