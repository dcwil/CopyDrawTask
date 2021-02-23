# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:00:00 2021

@author: Daniel
"""
import numpy as np
from pixToIm import pixToIm



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_pickle('results/TEST_SESSION/TEST_BLOCK/scores_copyDraw_block1.pkl')
    
    pixLet = df.loc['pos_t'][0]
    theBox = df.loc['theBoxPix'][0]
    winSize = df.loc['winSize'][0]
    target = df.loc['templatePix'][0]
    
    gaus1,im1,blur1 = pixToIm(pixLet,theBox,winSize)
    gaus2,im2,blur2 = pixToIm(target,theBox,winSize)
    
    fig, ax = plt.subplots(nrows=2,ncols=2)
    
    ax[0,0].plot(pixLet.T[0],pixLet.T[1])
    ax[1,0].plot(target.T[0],target.T[1])
    
    ax[0,1].imshow(gaus1)
    ax[1,1].imshow(gaus2)