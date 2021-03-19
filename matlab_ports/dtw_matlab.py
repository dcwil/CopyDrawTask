# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 14:48:09 2021

@author: Daniel
"""

#python'd version of dtw.m

import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt

def dtw_matlab(s,t,*args,w=float('inf')):#s is trace, t is template
    
    if s.dtype != 'float64' or t.dtype != 'float64':
        print('Check dtype!')
    
    # this is in the matlab code but I don't really get it
    if len(args) == 0:
        w = float('inf')
        
    ns = s.shape[0] #n_samples
    nt = t.shape[0]
    
    if s.shape[1] != t.shape[1]:
        raise ValueError('Error in dtw(): the dimensions of the two input signals do not match.')
        
    # adapt window size
    w = max([w, np.abs(ns-nt)])
    
    # cache matrix
    D = np.zeros([ns+1, nt+1]) + float('inf')  # array of infs
        
    C = np.zeros([ns+1, nt+1])
    
    if nt > ns:
        D = D.T
        C = C.T

    D[0, 0] = 0
    
    # begin dynamic processing (i must be higher than j)
    for i in range(D.shape[0]-1):
        # print(f'range: [{max([i-w,1])},{min([i+w,D.shape[1]-1])}]')
        for j in range(max([i-w, 0]), min([i+w, D.shape[1]-1])):

            oost = norm(s[i, :] - t[j, :]) \
                if ns >= nt else norm(s[j, :] - t[i, :])
        
            C[i+1, j+1] = oost
        
            D[i+1, j+1] = oost + min([D[i, j], D[i+1, j], D[i, j+1]])


    # print(i,j,D.shape,s.shape,t.shape)
    idx_min = np.argmin(C[-1,1:]) if ns >= nt else np.argmin(C[1:, -1])
    
    d = D[-1, -1]
    
    d_l = D[-1, idx_min] if ns >= nt else D[idx_min, -1]

    # compute optimal path
    D = D[1:, 1:]
    
    optim_path = np.zeros([max([s.shape[0],t.shape[0]]),2])
    
    for i in range(optim_path.shape[0]):
        optim_path[i,0] = i
        i_min = np.argmin(D[i,:])
        optim_path[i,1] = i_min
        
    #optimpath returns -1 indices, add one to make exact the same as matlab
    #same goes for idx min
        
    return d, d_l, optim_path, idx_min
    

### verify ###
# checked in matlab, my func produces the same results in python as the matlab one does
# still cant recreate sample data
if __name__ == '__main__':
    
    import scipy.io as sio
    pro_scores = sio.loadmat(
        '../sample_data/processed_scores_trial_01_block01.mat', simplify_cells=True)
    #i think those are block scores
    
    scores = sio.loadmat('../sample_data/scores_copyDraw_block01.mat', simplify_cells=True)
    templates = sio.loadmat('../templates/Size_20.mat', simplify_cells=True)
    #old_templates = sio.loadmat('../CopyDraw_mat_repo/CopyDrawTask-master/templates/shapes/Size_2/132.mat',simplify_cells=True)
    
    
    test_trial = scores['block_perf'][4]
    
    #converting dtype is vital - put a warning in the func?
    test_trace = test_trial['pos_t'].astype(float)
    test_template = templates['new_shapes'][1].astype(float)
    #test_old_template = old_templates['templateLet'] 
    #old template gives massive pathlen, can't be right
    ## Can't verify with pathlen - gets potentially changed later in compute_scoreSingleTrial.m
    #it also gives wrong w shape (bc old template is huge)
    
    ### check that we have the right template ###
    ### cant find any indicator of the template in the sample data scores ###
    plt.figure()
    plt.plot(test_trace[:,0],test_trace[:,1],label='trace')
    plt.plot(test_template[:,0],test_template[:,1],label='template')
    #plt.scatter(test_old_template[:,0],test_old_template[:,1],label='old',color='k',alpha=0.3)
    plt.legend()
    plt.show()
    ### visually inspect! ###
    #trial 4 goes with template 1 d:[360,518]
    #trial 5 goes with template 4 d_l:[360,515]
    #trial 6 goes with template 0
    
    dt, dt_l, w, pathlen = dtw_matlab(test_trace,test_template)
    
    
    # w and pathlen will need +1 when checking with matlab since bc of the index difference
    print(f'dt correct: {dt==test_trial["dt"]}')
    print(f'dt_l correct: {dt_l == test_trial["dt_l"]}')
    print(f'pathlen correct: {pathlen+1 == test_trial["pathlen"]}')
    print(f'w shape correct: {w.shape == test_trial["w"].shape}')
    print(f'w correct: {(w+1 == test_trial["w"]).all()}')
    
    