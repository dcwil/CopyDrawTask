# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:30:26 2020

@author: Daniel
"""
import pandas as pd
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


### Will need to apply inverse to convert from norm to og units again right?
### Need to check if scaling matrix is invertible (probably is tho)
### should this be a static method or a separate func?
### add in option to flip vertically or however
def scale_to_norm_units(array,scaling_matrix=None):
    """ 
    for scaling matrix to monitor norm units: scales input array between 0 and 1
    and then translates to origin in 1,-1 coord system
    
    """
    
    #translation require an extra col of 1s
    temp_array = np.ones([array.shape[0],array.shape[1]+1])
    
    #is this copying or referencing? Check it!
    temp_array[:,0] = array[:,0]
    temp_array[:,1] = array[:,1]
    
    if scaling_matrix is None:
        scaling_matrix = np.identity(3)
        
        
        ##minmax scaling
        #scaling
        S_x = 1/(np.max(array[:,0]) - np.min(array[:,0]))
        S_y = 1/(np.max(array[:,1]) - np.min(array[:,1]))
        
        #pre scaling translation
        T_x = -np.min(array[:,0])
        T_y = -np.min(array[:,1])
        
        
        ##translate to origin in norm units
        #post scaling translation
        t_x = -0.5
        t_y = -0.5
        
        scaling_matrix[0,0] = S_x
        scaling_matrix[1,1] = S_y
        
        scaling_matrix[0,2] = T_x * S_x + t_x
        scaling_matrix[1,2] = T_y * S_y + t_y
        
    scaled_matrix = np.matmul(scaling_matrix,temp_array.T).T[:,:-1]
    return scaled_matrix,scaling_matrix
 

def smooth(shape,return_df=False):       
        
    # create img
    df = pd.DataFrame(shape, columns=['x', 'y'])
    df['dx'] = df.x.diff()
    df['dy'] = df.y.diff()
    df['dxma'] = df.dx.rolling(2).mean()
    df['dyma'] = df.dy.rolling(2).mean()
    df['dxs'] = 0
    df['dys'] = 0
    df['dxs'][:-1] = (df.dx - df.dxma)[1:]
    df['dys'][:-1] = (df.dy - df.dyma)[1:]
    df['dxm'] = df.x + df.dxs
    df['dym'] = df.y + df.dys
    
    #first row will be nans, need to replace with og starting points?
    df['dxm'][0] = df['x'][0]
    df['dym'][0] = df['y'][0]
    
    
    return df if return_df else df[['dxm','dym']].to_numpy()
    
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