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


def scale(arr,minmax=None): # add in a way to save the scaling factors
    """
    for scaling a 2d array between 0 and 1
    """    
    # pretty sure this can be done with a single matrix mult, which would be easy to save
    for i in range(2):
        if minmax != None: 
            arr[:,i] -= np.min(arr[:,i])
            arr[:,i] /= np.max(arr[:,i])
        else:
            arr[:,i] -= np.min(minmax[:,i])
            arr[:,i] /= np.max(minmax[:,i])
        
    return arr

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