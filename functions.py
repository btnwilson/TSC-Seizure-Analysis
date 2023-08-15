#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 09:23:50 2023

@author: wilsobe
"""
import pyedflib as edf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.cluster.hierarchy as ch 
from statsmodels.multivariate.manova import MANOVA
import pandas as pd

def read_file(file_name, path= '', file_type= 'edf'):
    """
    Reads either edf (European Data Format) or csv files and stores information
    in a dictionary. 

    Parameters
    ----------
    file_name : str
        Name of the file intended for loading..
    path : str, optional
        Path to file location.. The default is ''.
    file_type : str, optional
        Either 'edf' or 'csv' depending on file type. The default is 'edf'.

    Returns
    -------
    file_info : Dict
        Dictionary containing all gatherable information from edf or csv file.

    """
    file_info = {}
    if file_type == 'edf':
        signals, headers, main_header = edf.highlevel.read_edf(path + "/" + file_name)
        file_info['signals'] = signals
        file_info['headers'] = headers
        file_info['main_header'] = main_header
        file_info['frequency'] =  1 / headers[0]['sample_rate']
        time_s = file_info['frequency'] * len(signals[0,:])
        file_info['time_sec'] = np.linspace(0, time_s, num= len(signals[0,:]))
        file_info['num_measurements'] = len(signals[0,:])
        return file_info
    
    
    elif file_type == 'csv':
        file_info['signals'] =  np.loadtxt(path + '/' + file_name, delimiter= ",", dtype= float)
        file_info['num_measurements'] = len(file_info['signals'][:,0])
        file_info['frequency'] = float(input("Enter Sampling Frequency:"))
        return file_info

 
def hankel(signal, num_columns):
    """
    Creates a hankel matrix from a given signal. Hankel matrices have the form 
            [a0  a1  ...  an]
        H = [a1  a2  ... an+1]
            [a2  a3  ... an+2]
            
    Parameters
    ----------
    signal : ndarray or list
        1D array or list of indivdual signal.
    num_columns : int
        Determines the number of columns in the hankel matrix
        (n in example matrix above).

    Returns
    -------
    hankel : ndarray
        A 2D array (length of the signal - n x n) with the structure of the
        example above.

    """
    index_nums = np.array(list(range(0,len(signal)+1)))
    
    hankel_index = np.zeros((len(signal) - num_columns, num_columns), dtype = int)
    rows_hankel = len(signal)-num_columns
    
    for i in range(0, rows_hankel): 
        hankel_index[i, :] = index_nums[i: i + num_columns]
    
    hankel = signal[hankel_index]
    
    return hankel

    

def heatmap(matrix, name):
    """
    Creates a basic heatmap particularly for the visualization of matrices.
    

    Parameters
    ----------
    matrix : ndarray
        2D array for which the visual will be created.
    name : str
        Title for the heatmap when it is displayed.

    Returns
    -------
    None.

    """
    plt.figure()
    plt.title(name)
    plt.imshow(matrix, cmap=  'hot', interpolation = 'nearest', aspect= 'auto')
    plt.colorbar()
    plt.show()

    
def scatter(corr_df, group_list, title, ptype= "3d"):
    """
    Creates scatter plots of the the first three PCs of a correlation 
    coefficient matrix. Scatter plot is color coded for the different types
    of catagories. 

    Parameters
    ----------
    corr_df : pandas.DataFrame
        DataFrame of the correlation coefficient matrix including index values
        specifying the files .
    group_list : lst
        List of strings that determine the color coding for the plot.
    title : str
        Title for the scatter plot.
    ptype : str, optional
        Either '2d' or '3d'. If 2d then prints three 2d scatter plots of all
        combinations of the first three PCs.
        The default is "3d".

    Returns
    -------
    None.

    """
    pos_colors = ['red', 'green', 'blue', 'black', 'cyan', 'purple', 'yellow', 'magenta', 'cyan']
    plot_colors = []
    type_list = []
    for file in corr_df.columns.values:
        for i, cat in enumerate(group_list):
            string_len = len(group_list[i])
            if file[0:string_len] == cat:
                plot_colors.append(pos_colors[i])
                type_list.append(cat)
    legend_elements = []
    for i, cat in enumerate(group_list):
        legend_elements.append(Line2D([0], [0], marker='o', color= 'w', label= group_list[i], markerfacecolor= pos_colors[i]))

    cc_eig_val, cc_eig_vec = np.linalg.eig(corr_df)
    
    
    eig_1_to_3_df = pd.DataFrame({'Type' : type_list, 'Vec1' :  cc_eig_vec[:,0], 'Vec2' : cc_eig_vec[:,1], 'Vec3' :  cc_eig_vec[:,2]})
    
    if ptype == '3d':
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.set_title(title)
        ax.scatter3D(cc_eig_vec[:,0], cc_eig_vec[:,1], cc_eig_vec[:,2], c = plot_colors)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        fig.legend(handles = legend_elements)
        fig.show()
        
        fit = MANOVA.from_formula('Vec1 + Vec2 + Vec3 ~ Type', data= eig_1_to_3_df)
        print('\n\t\t\t\t\t\t\t' + title)
        print(fit.mv_test())
    if ptype == '2d':
        for tup in [(0,1), (0, 2), (1, 2)]:
            plt.figure()
            plt.title(title)
            plt.axhline(0, color='black')
            plt.axvline(0, color='black')
            plt.scatter(cc_eig_vec[:,tup[0]], cc_eig_vec[:,tup[1]], c=plot_colors)
            plt.xlabel(f'PC{tup[0] + 1}')
            plt.ylabel(f'PC{tup[1] + 1}')
            plt.legend(handles=legend_elements)
            plt.show()
  
def deconstruct_hankel(hankel):
    """
    Deconstructs the hankel matrix to return the original signal. 

    Parameters
    ----------
    hankel : ndarray
        Hankel matrix to deconstruct.

    Returns
    -------
    signal : lst
        A list of the original signal as long as input was not a rank
        approximation.

    """
    signal = []
    signal.append(hankel[0,:-1])
    signal.append(hankel[1:, -1])
    return signal


def clustering(num_groups, corr_df, title):
    """
    Generates a dendrogram and heatmaps from a correlation DataFrame.
    One heatmap is for sorting by the first eigenvector of the correlation 
    matrix, the other is sorted by the dendrogram groups. 

    Parameters
    ----------
    num_groups : int
        Number of groups the dendrogram divides the data into.
    corr_df : pandas.DataFrame
        Correlation coefficient matrix of vectors from each file.
    title : str
        Goes in title of each figure, used to display the method for creating
        the vector for each file in this case.

    Returns
    -------
    groups : dict
        Dictionary lists containing the sorted index values from the DataFrame
        as sorted by the dendrogram program.

    """
    plt.figure()
    plt.title(title)
    linkage_data = ch.linkage(corr_df, method='ward', metric='euclidean')
    dend = ch.dendrogram(linkage_data)
    plt.show()
    groups = {}
    tree_pieces = ch.cut_tree(linkage_data, n_clusters=num_groups)
    for i in range(0, num_groups):
        groups[f'group_{i+1}'] = []
        for num, piece in enumerate(tree_pieces):
            if piece == i:
                groups[f'group_{i+1}'].append(corr_df.index[num])
    cc = corr_df.to_numpy()
    cc_int = cc[dend['leaves'][:]]
    cc_aranged = cc_int[:,dend['leaves'][:]]
    heatmap(cc_aranged, title + ' Correlation Coef Matrix Sorted By Dend Groups')
    
    cc_eig_val, cc_eig_vec = np.linalg.eig(corr_df)
    
    order_eig =  np.argsort(cc_eig_vec[:,0])

    cc_int_eig = cc[order_eig, :]
    cc_aranged_eig = cc_int_eig[:,order_eig]
    heatmap(cc_aranged_eig, title + ' Correlation Coef Matrix Sorted By 1st Eigen Vector')
    return groups
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    