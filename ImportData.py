
"""
Created on Wed Jun 21 11:42:44 2023

@author: wilsobe
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import pyedflib as edf 
import os 

file = input('Enter file name: ')
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
location = find(file, "Tsc1_EEG")
print(location)
#signals, headers, main_header = edf.highlevel.read_edf('./Tsc1_EEG/BXD87/HET/TBXD1_F3_HET/Default_0001_2022-05-16_17_56_53_export.edf')
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
# %%
"""
# in order to cut specified sections of the seizure and create new shorter edf files (time in seconds entered)
start = int(55320/ time_interval)
end =   int(55440/ time_interval)
edf.highlevel.write_edf_quick('Control_Clip_1.edf', [channel1[start:end], channel2[start:end], channel3[start:end]], 256)
"""


"""
# plots the EEG and incorperates a slider to scan through clips

fig, Axis = plt.subplots()
eeg = plt.plot(channel1[:len_window])
slider_color = 'White'
axis_position = plt.axes([0.2, 0.1, 0.65, 0.03],
                         facecolor = slider_color)
slider_position = Slider(axis_position,
                         'Pos', 0.01, 50000)
def update(val):
    pos = slider_position.val
    Axis.axis([pos, pos+1000, -250, 250])
    fig.canvas.draw_idle()
slider_position.on_changed(update)
plt.show()
"""


