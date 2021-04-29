# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 14:30:26 2020

@author: Daniel
"""
import pandas as pd
import pyglet as pg
import numpy as np
import scipy.io
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pathlib import Path
from numpy.linalg import norm
# Keep all utils in here, or have separate utils files for different bits?


def select_display():
    """ Select the screen to display the task to

    NOTE: Looking at screens[0].display.x_screen and
    Looking at screens[0].display.x_screen is seems
    like a setup with an extended screen has only
    one number (screen index) with shifted coords,
    as can be accessed by screen[0].y or .x

    """
    screens = pg.canvas.Display().get_screens()

    if len(screens) > 1:
        print(f"Found {len(screens)} screens with the following settings:"
              ''.join([f"\nScreen {i}: \n {s}" for i, s in enumerate(screens)]))
        resp = -1
        while resp not in list(range(len(screens))):
            inp = input("\nPlease select one of "
                        + str(list(range(len(screens))))
                        + " : ")
            resp = int(inp)

        ix_scr = resp
    else:
        ix_scr = 0

    screen_conf = {'screen': ix_scr}
    # check if the internal indeces agree, if yes -> shift if necessary
    # to the correct screen
    if all([s.display.x_screen == screens[0].display.x_screen
            for s in screens]):
        # now assume that .x and .y of the selected screen correspond
        # to the offsets to the main display
        screen_conf['pos'] = [screens[ix_scr].x, screens[ix_scr].y]

    return screen_conf


def remove_nans_2d(arr):
    """ Removes rows where nans are present. For use with trace data etc """
    # https://stackoverflow.com/questions/11453141/how-to-remove-all-rows-in-a-numpy-ndarray-that-contain-non-numeric-values

    # the tilde flips the bools
    return arr[~np.isnan(arr).any(axis=1)]


# Do I still need this?
def load_pos_and_template(filepath, templates_dir):
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
    
    
    return pos_arr, template


def scale(array, minmax=None): # add in a way to save the scaling factors
    # is this func needed anymore?
    """
    for scaling a 2d array between 0 and 1
    """    
    arr=array.copy()
    
    # pretty sure this can be done with a single matrix mult, which would be easy to save
    for i in range(2):
        if minmax is None:  # rename minmax. poor choice
            arr[:, i] -= np.min(arr[:, i])
            arr[:, i] /= np.max(arr[:, i])
        else:
            arr[:, i] -= np.min(minmax[:, i])
            arr[:, i] /= np.max(minmax[:, i])
        
    return arr


# Will need to apply inverse to convert from norm to og units again right?
# Need to check if scaling matrix is invertible (probably is tho)
def scale_to_norm_units(array,scaling_matrix=None):
    """ 
    for scaling matrix to monitor norm units: scales input array between 0 and 1
    and then translates to origin in 1,-1 coord system
    
    """
    
    # translation require an extra col of 1s
    temp_array = np.ones([array.shape[0],array.shape[1]+1])
    
    # is this copying or referencing? Check it!
    temp_array[:, 0] = array[:, 0]
    temp_array[:, 1] = array[:, 1]
    
    if scaling_matrix is None:
        scaling_matrix = np.identity(3)

        # minmax scaling
        # scaling
        S_x = 1/(np.max(array[:, 0]) - np.min(array[:, 0]))
        S_y = 1/(np.max(array[:, 1]) - np.min(array[:, 1]))
        
        # pre scaling translation
        T_x = -np.min(array[:, 0])
        T_y = -np.min(array[:, 1])
        
        # translate to origin in norm units
        # post scaling translation
        t_x = -0.5
        t_y = -0.5
        
        scaling_matrix[0, 0] = S_x
        scaling_matrix[1, 1] = S_y
        
        scaling_matrix[0, 2] = T_x * S_x + t_x
        scaling_matrix[1, 2] = T_y * S_y + t_y
        
    scaled_matrix = np.matmul(scaling_matrix, temp_array.T).T[:, :-1]
    return scaled_matrix,scaling_matrix


def scale_trace_to_template(trace, template, return_sf=False):
    """ NOTE, assumes the trace and template have the same starting position
    (just in different coord frames) which should be ensured via the code,
    but there have been bugs in the past changing this"""
    sf = template[0]/trace[0]
    return (trace*sf, sf) if return_sf else trace*sf


def smooth(shape, return_df=False):
        
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
    
    # first row will be nans, need to replace with og starting points?
    df['dxm'][0] = df['x'][0]
    df['dym'][0] = df['y'][0]

    return df if return_df else df[['dxm','dym']].to_numpy()


def simple_plot(arr, kind='plot', label=None):
    """ given a 2d array plots it """
    assert arr.shape[1] == 2
    
    from matplotlib.pyplot import plot,scatter,figure
    
    #figure()
    types = {'plot':plot,'scatter':scatter}
    types[kind](arr[:,0],arr[:,1], label=label)


def movingmean(arr, w_size):
    """ This is trying to mimic some of the functionality from:
    https://uk.mathworks.com/matlabcentral/fileexchange/41859-moving-average-function
    which (I think) is the function used in compute_scoreSingleTrial.m
    (not in matlab by default). Returns an array of the same size by shrinking
    the window for the start and end points. """

    # round down even window sizes
    if w_size%2 == 0:
        w_size -= 1
    
    w_tail = np.floor(w_size/2)
    
    arr_sub = np.zeros_like(arr)
    
    for j, col in enumerate(arr.T):  # easier to work with columns like this
        for i, val in enumerate(col):
            
            # truncate window if needed
            start = i - w_tail if i > w_tail else 0
            stop = i + w_tail + 1 if i + w_tail < len(col) else len(col)
            s = slice(int(start),int(stop))
            
            # idxs reversed bc .T
            arr_sub[i,j] = np.mean(col[s])
            
            # could probably find a way to do this both cols at the same time
    
    return arr_sub


def deriv_and_norm(var, delta_t):
    """
    Given an array (var) and timestep (delta_t), computes the derivative 
    for each timepoint and returns it (along with the magnitudes)
    
    """
    # This is not the same as the kinematic scores in the matlab code!
    deriv_var = np.diff(var, axis=0)/delta_t
    deriv_var_norm = norm(deriv_var, axis=1)
    return deriv_var, deriv_var_norm


def check_with_tol(x, y, tol=0.0001):
    return np.abs(x - y) < tol


def template_to_image(template, fname, path, **kwargs):

    # if template images dir doesn't exists make it
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)

    fullpath = path.joinpath(f'{fname}.png')

    if not fullpath.exists():
        plt.figure(figsize=(16, 10))
        plt.plot(template.T[0], template.T[1],
                 **kwargs)  # how can i remove this from being shown?
        ax = plt.axes()
        ax.xaxis.set_major_locator(ticker.NullLocator())
        ax.yaxis.set_major_locator(ticker.NullLocator())
        plt.tight_layout(pad=0)
        plt.savefig(fullpath, format='png', bbox_inches='tight',
                    transparent=True, dpi=300)

    return fullpath


def create_template_order(stimuli_dict, block_settings_dict):
    # requires stimuli to be loaded & n_trials to have been defined
    if block_settings_dict['n_trials'] % stimuli_dict['n_templates'] != 0:
        # change to a proper warning message?
        print(f'WARNING: {block_settings_dict["n_trials"]} trials means that '
              f'there will be an uneven number of templates')

    order = [(i % stimuli_dict['n_templates'])
                             for i in range(block_settings_dict['n_trials'])]

    if block_settings_dict['shuffle']:
        random.shuffle(order)

        # reshuffle to remove repeated trials showing
        while 0 in np.diff(np.array(order)):
            random.shuffle(order)

    return order


def unnest(ls: list, final=None):
    """ For flattening nested lists"""
    if final is None:
        final = []
    if isinstance(ls, list):
        for i in ls:
            final = unnest(i, final=final)
    else:
        final.append(ls)
    return final