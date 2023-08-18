#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 10:05:51 2023

@author: wilsobe
"""

import random
import os
import pickle
import copy 
path = '/Users/wilsobe/Desktop/Python Files/Real Data/SeizureClips'
os.chdir(path)
# create list of seizure clip names
files = []
for file_name in os.listdir():
    files.append(file_name)
files.remove('.DS_Store')
copy_files = copy.deepcopy(files)
group_list = {}
while len(files) > 6:
    bxd6 = False
    bxd17 = False
    bxd9 = False
    num_bxd9 = 0
    while bxd6 != True or bxd17 != True or bxd9 != True or num_bxd9 > 1:
        num_bxd9 = 0
        bxd6 = False
        bxd17 = False
        bxd9 = False
        sample = random.sample(files, k=6)
        print(sample)
        for clip in sample:
            if clip[0:5] == 'BXD_6':
                bxd6 = True
            if clip[0:6] == 'BXD_17':
                bxd17 = True
            if clip[0:5] == 'BXD_9':
                bxd9 = True
                num_bxd9 += 1
    group_list[f'group_{len(group_list) + 1}'] = sample
    for clip in sample:
        files.remove(clip)
group_list['remainder'] = files     
validation = {}
train = {}
for group in group_list:
    all_files = copy.deepcopy(copy_files)
    if len(group_list[group]) == 6:
        for clip in group_list[group]:
            all_files.remove(clip)
        val = random.sample(all_files, k= 12)
        validation[f'group_{len(validation) + 1}'] = val
        for clip in val:
            all_files.remove(clip)
        train[f'group_{len(train) + 1}'] = all_files
    else:
        for clip in group_list[group]:
            all_files.remove(clip)
        add_ons = random.sample(all_files, k= 3)
        for clip in add_ons:
            all_files.remove(clip)
            group_list[group].append(clip)
        val = random.sample(all_files, k= 12)
        validation[f'group_{len(validation) + 1}'] = val
        for clip in val:
            all_files.remove(clip)
        train[f'group_{len(train) + 1}'] = all_files
split_data = {"training": train, 'validation': validation, 'test': group_list}

# %%
pickle.dump(split_data, open('Autoencoder Groups 1.pkl', 'wb'), protocol= pickle.HIGHEST_PROTOCOL)

# %%
pickled_file = pickle.load(open('/Users/wilsobe/Desktop/Python Files/Real Data/Autoencoder Groups 1.pkl', 'rb'))

