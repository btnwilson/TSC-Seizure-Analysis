#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 19:19:02 2023

@author: wilsobe
"""

import functions as f
import os
import matplotlib.pyplot as plt
import pickle
path = '/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips'
os.chdir(path)
# create list of seizure clip names
files = []
for file_name in os.listdir():
    files.append(file_name)
files.remove('.DS_Store')

for file in files:
    info = f.read_file(file, path)
    plt.figure(figsize=(100,10))
    plt.title(file)
    plt.plot(info['signals'][0,:])
    plt.show()
seizure_range = {'BXD_6_Clip18.edf': [8000, 24000], 'BXD_6_Clip_24.edf': [0, 67000], 'BXD_17_Clip_8.edf': [7500, 27500],
                 'BXD_17_Clip_9.edf': [20000, 40000], 'BXD_9_Clip_9.edf': [5000, 15000], 'BXD_9_Clip_8.edf': [25000, 35000],
                 'BXD_6_Clip_22.edf': [10000, 60000], 'BXD_6_Clip_21.edf': [2500, 25000], 'BXD_6_Clip_20.edf': [5000, 15000],
                 'BXD_6_Clip_8.edf': [18000, 85000], 'BXD_6_Clip_9.edf': [15000, 40000], 'BXD_17_Clip_18.edf': [6000, 17500],
                 'BXD_17_Clip_19.edf': [4000, 17500], 'BXD_17_Clip_21.edf': [5000, 22500], 'BXD_17_Clip_20.edf': [17500, 30000],
                 'BXD_17_Clip_22.edf': [8000, 22000], 'BXD_17_Clip_12.edf': [7500, 20000], 'BXD_6_Clip_1.edf': [7500, 11000],
                 'BXD_17_Clip_13.edf': [7500, 18000], 'BXD_17_Clip_11.edf': [11000, 22500], 'BXD_6_Clip_6.edf': [25000, 38000],
                 'BXD_9_Clip_10.edf': [22500, 31000], 'BXD_6_Clip_3.edf': [5300, 26000], 'BXD_17_Clip_10.edf': [7000, 19500],
                 'BXD_17_Clip_14.edf': [5000, 8000], 'BXD_6_Clip_7.edf': [32500, 86000], 'BXD_17_Clip_15.edf': [13500, 20000],
                 'BXD_17_Clip_17.edf': [7000, 13500], 'BXD_6_Clip_5.edf': [10000, 50000], 'BXD_17_Clip_16.edf': [13500, 22500],
                 'BXD_17_Clip_1.edf': [12000, 20000], 'BXD_9_Clip_3.edf': [7750, 15500], 'BXD_9_Clip_2.edf': [4000, 13000],
                 'BXD_6_Clip_10.edf': [13000, 26000], 'BXD_6_Clip_12.edf': [3000, 7200], 'BXD_17_Clip_2.edf': [8500, 17500],
                 'BXD_9_Clip_1.edf': [5500, 12500], 'BXD_17_Clip_3.edf': [5000, 15000], 'BXD_6_Clip_13.edf': [5000, 14500],
                 'BXD_17_Clip_7.edf': [14000, 24500], 'BXD_9_Clip_5.edf': [7750, 16500], 'BXD_9_Clip_4.edf': [7000, 14000],
                 'BXD_17_Clip_6.edf': [7750, 18500], 'BXD_6_Clip_16.edf': [8800, 23000], 'BXD_6_Clip_14.edf': [600, 13200],
                 'BXD_17_Clip_4.edf': [37000, 49000], 'BXD_9_Clip_6.edf': [20000, 28500], 'BXD_9_Clip_7.edf': [10000, 19000],
                 'BXD_17_Clip_5.edf': [13500, 24500], 'BXD_6_Clip_15.edf': [1000, 12000]}
pickle.dump(seizure_range, open('Seizure Range Dict.pkl', 'wb'), protocol= pickle.HIGHEST_PROTOCOL)