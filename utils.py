# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:30:26 2020

@author: Daniel
"""

import numpy as np
import scipy.io

from pathlib import Path


#Do I still need this?
def load_pos_and_template(filepath,templates_dir):
    """
    filepath must include at least 2 parent directories - they contain the 
    info about which template was used
    """
    filepath = Path(filepath)
    templates_dir = Path(templates_dir)
    
    template_idx = int(filepath.parts[-2])
    templates_fname = f'{filepath.parts[-3]}.mat'
    
    templates_path = Path(templates_dir,templates_fname)
    templates_mat = scipy.io.loadmat(templates_path)
    templates = templates_mat['new_shapes'][0]

    template = templates[template_idx]
    pos_arr = np.load(filepath)
    
    
    return pos_arr,template


def scale(array,minmax=None): # add in a way to save the scaling factors
    """
    for scaling a 2d array between 0 and 1
    """    
    arr=array.copy()
    
    # pretty sure this can be done with a single matrix mult, which would be easy to save
    for i in range(2):
        if minmax == None: # rename minmax. poor choice
            arr[:,i] -= np.min(arr[:,i])
            arr[:,i] /= np.max(arr[:,i])
        else:
            arr[:,i] -= np.min(minmax[:,i])
            arr[:,i] /= np.max(minmax[:,i])
        
    return arr



def scale_to_norm_units(array,scaling_matrix=None):
    """ 
    for scaling matrix to monitor norm units: scales input array between 0 and 1
    and then translates to origin in 1,-1 coord system
    
    """
    temp_array = np.ones([array.shape[0],array.shape[1]+1])
    temp_array[:,0] = array[:,0]
    temp_array[:,1] = array[:,1]
    
    if scaling_matrix == None:
        scaling_matrix = np.identity(3)
        
        #scaling
        S_x = 1/(np.max(array[:,0]) - np.min(array[:,0]))
        S_y = 1/(np.max(array[:,1]) - np.min(array[:,1]))
        
        #pre scaling translation
        T_x = -np.min(array[:,0])
        T_y = -np.min(array[:,1])
        
        #post scaling translation
        t_x = -0.5
        t_y = -0.5
        
        scaling_matrix[0,0] = S_x
        scaling_matrix[1,1] = S_y
        
        scaling_matrix[0,2] = T_x * S_x + t_x
        scaling_matrix[1,2] = T_y * S_y + t_y
        
    return np.matmul(scaling_matrix,temp_array.T).T,scaling_matrix
        
        
    
#dont need this anymore
# def load_pixel_data(path_to_mouse_pos_pix):
#     """ 
#     helper func for loading data
#     """
    
#     path_to_mouse_pos_pix = Path(path_to_mouse_pos_pix)
#     path_to_template_pix = Path(path_to_mouse_pos_pix.parent,'template_pix.npy')
#     mouse_pos_pix = np.load(path_to_mouse_pos_pix)
#     template_pix = np.load(path_to_template_pix)
#     return mouse_pos_pix, template_pix


def simple_plot(arr, kind='plot'):
    """ given a 2d array plots it """
    assert arr.shape[1] == 2
    
    from matplotlib.pyplot import plot,scatter,figure
    
    #figure()
    types = {'plot':plot,'scatter':scatter}
    types[kind](arr[:,0],arr[:,1])