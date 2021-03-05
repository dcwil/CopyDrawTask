# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:00:00 2021

@author: Daniel
"""
import numpy as np
from pixToIm import pixToIm


### not yet written in the statistics on individual trials thats in the mat file

def getScore(gaus_template,gaus_trace):
    
    youSuck = np.sum(gaus_template)
    score = np.sum(np.abs(gaus_template-gaus_trace))
    score = score/youSuck 
    
    if score > 1: score = 1
    
    score = 1-score
    score = score*100
    return score


#wrap pixtioim and getscore into one
def get_score_from_trace(trace,template,thebox,winsize,shift=True):
    
    gaus_template,_,_ = pixToIm(template,thebox,winsize,shift=shift)
    gaus_trace,_,_ = pixToIm(trace,thebox,winsize,shift=shift)
    
    return getScore(gaus_template,gaus_trace)

if __name__ == '__main__':
    #import matplotlib.pyplot as plt
    #import pandas as pd
    
    # df = pd.read_pickle('results/TEST_SESSION/TEST_BLOCK/scores_copyDraw_block1.pkl')
    
    # trial_no = 0
    
    # pixLet = df.loc['pos_t'][trial_no]
    # the_box = df.loc['theBoxPix'][trial_no]
    # winSize = df.loc['winSize'][trial_no]
    # target = df.loc['templatePix'][trial_no]
    
    # gaus1,im1,blur1 = pixToIm(target,the_box,winSize)
    # gaus2,im2,blur2 = pixToIm(pixLet,the_box,winSize) #trace
    
    
    
    # #sanity check
    # fig, ax = plt.subplots(nrows=2,ncols=2)
    
    # ax[0,0].plot(pixLet.T[0],pixLet.T[1])
    # ax[1,0].plot(target.T[0],target.T[1])
    
    # ax[0,1].imshow(gaus1)
    # ax[1,1].imshow(gaus2)
    
    
    # youSuck = np.sum(gaus1)
    # score = np.sum(np.abs(gaus1-gaus2))
    # score = score/youSuck
    
    # print(score)
    
    # #cut off the score if its too high (high =bad?)
    # if score > 1:
    #     score = 1
        
    # score = 1-score
    # score = score*100
    # print(score)
    
    
    #verify with matlab
    from pathlib import Path
    import scipy.io as sio
    
    
    #matlab results
    for s in [1,2,3]:
        p = Path(rf'F:\F_WSL\AG Tangermann\CopyDraw\CopyDraw_mat_repo\CopyDrawTask-master\tmp\sample_run_5f52b547-2a93-47e4-914b-08d1f72d1e7f\copyDraw_block01\tscore_{s}copyDraw_block01.mat')
        
        matlab_res = sio.loadmat(p,simplify_cells = True)
        
        theBox_mat = matlab_res['the_box']['boxPos'].T.astype(float)
        winSize_mat = np.array([3440,1440]) # matlab just goes fullscreen - adapt to other monitors if needed
        target_mat = matlab_res['templateLet'].astype(float)
        pixLet_mat = matlab_res['traceLet'].astype(float) # should be int  but there are nans in there that we "need"
        
        pyscore = get_score_from_trace(pixLet_mat, target_mat, theBox_mat, winSize_mat, shift=False)
        matscore = matlab_res['score']
        
        print(f'MATLAB: {matscore}\nPYTHON: {pyscore}')
        
        #output: 
            
            # MATLAB: 8.004574446468194
            # PYTHON: 9.112219009818578
            # MATLAB: 9.757498670798515
            # PYTHON: 12.177279084773918
            # MATLAB: 8.435895381163338
            # PYTHON: 10.640118055976622
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    