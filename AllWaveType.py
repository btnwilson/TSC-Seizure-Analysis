#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 07:57:15 2023

@author: wilsobe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy as sp
import pyedflib as edf 
import os

slow_wave = {"1": np.array([[25, -15, -10], [35, 0, 0], [10,0,0]]),
             "2": np.array([[4],[-5],[-3]]),
             "3": np.array([[.0225], [.03], [.12]])}

spike_wave =  {"1": np.array([[38, -29, -10], [40, 0, 0], [15,0,0]]),
               "2": np.array([[5],[-2],[0]]),
               "3": np.array([[.017], [.017], [.25]])}

spike_trains = {"1": np.array([[23, -15, -10], [35, 0, 0], [10,0,0]]),
                "2": np.array([[.05],[-5],[-5]]),
                "3": np.array([[.015],[.013],[.267]]) }

sinusoidal_wave = {"1": np.array([[24, -20, -15], [40, 0, 0], [7,0,0]]),
                   "2": np.array([[3],[-2],[0]]),
                   "3": np.array([[.013],[.013],[.267]])}
user_input = int(input("1, 2, 3, 4: "))
if user_input == 1:
    C = slow_wave["1"]
    I = slow_wave["2"]
    T = slow_wave["3"]
elif user_input == 2:
    C = spike_wave["1"]
    I = spike_wave["2"]
    T = spike_wave["3"]
elif user_input == 3:
    C = spike_trains["1"]
    I = spike_trains["2"]
    T = spike_trains["3"]
elif user_input == 4:
    C = sinusoidal_wave["1"]
    I = sinusoidal_wave["2"]
    T = sinusoidal_wave["3"]

def dxdt(x, t, C, I, T):
    
    # Coerce to np array
    x_np = np.array(x).reshape((3,1))
    
    # Define sigmoid
    def S(x, a, theta):
        S_x = 1 / (1 + np.exp(-a * (x - theta)))
        return(S_x)
    
    # Linear part
    lin_part = np.matmul(C,x_np) + I
    
    # Sigmoid part
    sig_part = -x_np + S(lin_part, 1, 4)
    
    # Scale by time constants
    x_dot = sig_part / T
    
    final = x_dot.T.flatten()
    
    return(final)

    
x0 = [0,1,0]
p = (C, I, T)
t = np.linspace(0, 31, 31744)
sol = odeint(dxdt, x0, t, p)

#x1_sol = sol[:, 1]
#x2_sol = sol[:, 2]
#x0_sol = sol[:, 0]
signal =  sol[1024:, 0]
# plot the x solution vs time 
for i in range(10,17, 1):
    noise = np.random.normal(0, .05, size= signal.shape)
    noisy_signal = signal + noise
    os.chdir('/Users/wilsobe/Desktop/Python Files/Real Data/Theoretical')
    edf.highlevel.write_edf_quick(f'SlowWave_{i+1}.edf', noisy_signal, 1024)
