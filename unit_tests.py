# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:09:25 2021

@author: Daniel
"""

#is unit tests the appropriate name for this?

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pandas as pd

from dtw import *

scores = sio.loadmat('sample_data/scores_copyDraw_block01.mat',simplify_cells=True)
templates = sio.loadmat('templates/Size_20.mat',simplify_cells=True)

test_trial = scores['block_perf'][4]

test_trace = test_trial['pos_t']
test_template = templates['new_shapes'][1]

### check that we have the right template ###
### cant find any indicator of the template in the sample data scores ###
plt.figure()
plt.plot(test_trace[:,0],test_trace[:,1],label='trace')
plt.plot(test_template[:,0],test_template[:,1],label='template')
plt.legend()
plt.show()
### visually inspect! ###
#trial 4 goes with template 1
#trial 5 goes with template 4
#trial 6 goes with template 0

#https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
def search_sequence_numpy(arr,seq):
    """ Find sequence in an array using NumPy only.

    Parameters
    ----------    
    arr    : input 1D array
    seq    : input 1D array

    Output
    ------    
    Output : 1D Array of indices in the input array that satisfy the 
    matching of input sequence in the input array.
    In case of no match, an empty list is returned.
    """

    # Store sizes of input array and sequence
    Na, Nseq = arr.size, seq.size

    # Range of sequence
    r_seq = np.arange(Nseq)

    # Create a 2D array of sliding indices across the entire length of input array.
    # Match up with the input sequence & get the matching starting indices.
    M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)

    # Get the range of those indices as final output
    if M.any() >0:
        return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
    else:
        return []         # No match found

# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    #array = np.nan_to_num(array, copy=True, nan=dt_l/2)
    idx = (np.abs(array - value)).argmin()
    return array.flatten()[idx]

#rabinerJuangStepPattern(type,slope_weighting="d",smoothed=False)

step_patterns = ['symmetric1','symmetric2','asymmetric','symmetricP0','asymmetricP0',
                 'symmetricP05','asymmetricP05','symmetricP1', 'asymmetricP1',
                 'symmetricP2','asymmetricP2','typeIa','typeIb','typeIc','typeId',
                 'typeIas','typeIbs','typeIcs','typeIds','typeIIa','typeIIb','typeIIc','typeIId',
                 'typeIIIc','typeIVc','mori2006']




#i think w/optim_path is alignment.index1 and 2
# pathlen/idx_min is probably jmin? (adapt for ptyhon counting though)
# also pathlen is sometimes "off" by a small amount (<10)
# d_l is indexed from costmatrix, correct pattern should contain d_l or something close?


print('Testing')
w = test_trial['w'] #also called optim path
w_ = {
    'w_x' : w[:,0]-1,
    'w_y' : w[:,1]-1
    }
pathlen = test_trial['pathlen'] # also called idx min
dt = test_trial['dt']
dt_l = test_trial['dt_l']

rows_list = []

for sp in step_patterns:
    #try:
        

    alignment = dtw(test_template,test_trace,step_pattern=sp,keep_internals=True)
    new_w = np.stack([alignment.index1,alignment.index2],axis=1)
    new_w_s = np.stack([alignment.index1s,alignment.index2s],axis=1)
    new_pathlen = min([alignment.jmin,test_template.shape[0]])
    new_d = alignment.distance
    new_dl = alignment.costMatrix[-1,alignment.jmin-1]
    
    row = {
        'w_nrows_difference':w.shape[0] - new_w.shape[0],
        'w_nrows_difference *':w.shape[0] - new_w_s.shape[0],
        'pathlen_difference':pathlen - new_pathlen,
        'query':'template',
        'reference':'trace',
        'dt_difference':dt - new_d,
        'dt_difference %':100*np.abs((dt - new_d))/dt,
        'dt in costmatrix':True if dt in alignment.costMatrix else find_nearest(alignment.costMatrix,dt)/dt,
        'dt_l in costmatrix':True if dt_l in alignment.costMatrix else find_nearest(alignment.costMatrix,dt_l)/dt_l,
        'stepPatten':sp,
        #look for index sequence in w
        # 'w[:,0] len match index1': len(search_sequence_numpy(w[:,0]-1,alignment.index1)),
        # 'w[:,0] len match index1 rev': len(search_sequence_numpy(seq=w[:,0]-1,arr=alignment.index1)),
        # 'w[:,0] len match index2': len(search_sequence_numpy(w[:,0]-1,alignment.index2)),
        # 'w[:,0] len match index1s': len(search_sequence_numpy(w[:,0]-1,alignment.index1s)),
        # 'w[:,0] len match index2s': len(search_sequence_numpy(w[:,0]-1,alignment.index2s)),
        # 'w[:,1] len match index1': len(search_sequence_numpy(w[:,1]-1,alignment.index1)),
        # 'w[:,1] len match index2': len(search_sequence_numpy(w[:,1]-1,alignment.index2)),
        # 'w[:,1] len match index1s': len(search_sequence_numpy(w[:,1]-1,alignment.index1s)),
        # 'w[:,1] len match index2s': len(search_sequence_numpy(w[:,1]-1,alignment.index2s)),

        }
    

    
    alignment_idxs = {
        'idx1':alignment.index1,
        'idx1s':alignment.index1s,
        'idx2':alignment.index2,
        'idx2s':alignment.index2s
        }
    
    for al_key,al_index in alignment_idxs.items():
        for w_key,w_vals in w_.items():
            if len(w_vals) > len(al_index):
                n_match = len(search_sequence_numpy(arr=w_vals, seq=al_index))
            else:
                n_match = len(search_sequence_numpy(arr=al_index, seq=w_vals))
                
            row[f'n_matches with {w_key} & {al_key}'] = n_match
            
            
    rows_list.append(row)
    
    

    alignment = dtw(test_trace,test_template,step_pattern=sp,keep_internals=True)
    new_w = np.stack([alignment.index1,alignment.index2],axis=1)
    new_w_s = np.stack([alignment.index1s,alignment.index2s],axis=1)
    new_pathlen = min([alignment.jmin,test_template.shape[0]])
    new_d = alignment.distance
    new_dl = alignment.costMatrix[-1,alignment.jmin-1]

    row = {
        'w_nrows_difference':w.shape[0] - new_w.shape[0],
        'w_nrows_difference *':w.shape[0] - new_w_s.shape[0],
        'pathlen_difference':pathlen - new_pathlen,
        'query':'trace',
        'reference':'template',
        'dt_difference':dt - new_d,
        'dt_difference %':100*np.abs((dt - new_d))/dt,
        'dt in costmatrix':True if dt in alignment.costMatrix else find_nearest(alignment.costMatrix,dt)/dt,
        'dt_l in costmatrix':True if dt_l in alignment.costMatrix else find_nearest(alignment.costMatrix,dt_l)/dt_l,
        'stepPatten':sp,
        # 'w[:,0] len match index1': len(search_sequence_numpy(w[:,0]-1,alignment.index1)),
        # 'w[:,0] len match index2': len(search_sequence_numpy(w[:,0]-1,alignment.index2)),
        # 'w[:,0] len match index1s': len(search_sequence_numpy(w[:,0]-1,alignment.index1s)),
        # 'w[:,0] len match index2s': len(search_sequence_numpy(w[:,0]-1,alignment.index2s)),
        # 'w[:,1] len match index1': len(search_sequence_numpy(w[:,1]-1,alignment.index1)),
        # 'w[:,1] len match index2': len(search_sequence_numpy(w[:,1]-1,alignment.index2)),
        # 'w[:,1] len match index1s': len(search_sequence_numpy(w[:,1]-1,alignment.index1s)),
        # 'w[:,1] len match index2s': len(search_sequence_numpy(w[:,1]-1,alignment.index2s)),
        }
    
    
    
    alignment_idxs = {
        'idx1':alignment.index1,
        'idx1s':alignment.index1s,
        'idx2':alignment.index2,
        'idx2s':alignment.index2s
        }
    
    for al_key,al_index in alignment_idxs.items():
        for w_key,w_vals in w_.items():
            if len(w_vals) > len(al_index):
                n_match = len(search_sequence_numpy(w_vals, al_index))
            else:
                n_match = len(search_sequence_numpy(al_index, w_vals))
                
                
            row[f'n_matches with {w_key} & {al_key}'] = n_match
    rows_list.append(row)
        
    # except:
    #     print(f'failed with {sp}')
        
df = pd.DataFrame(rows_list)

# inspecting df suggests none of the step patterns correctly get w (but symmetric1 gets a lot of the other stuff)
# buuuut plottting thigns like this might bemore informative:
    
fig, axs = plt.subplots(nrows=9,ncols=3,figsize=(8,16))

axs = axs.flatten()

for i,sp in enumerate(step_patterns):
    alignment = dtw(test_template,test_trace,step_pattern=sp,keep_internals=True)
    axs[i].plot(w[:,0]-1,w[:,1]-1,label='w',alpha=0.5)
    axs[i].plot(alignment.index1,alignment.index2,label='new',alpha=0.5)
    axs[i].set_title(sp)
    axs[i].legend()
plt.tight_layout()
plt.savefig('step_patterns.pdf', format='pdf')
