# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 12:04:11 2021

@author: Daniel
"""

import numpy as np
from scipy.signal import convolve2d


### THIS IS UNFINISHED, technically, should work tho ####
### not yet verified with matlab either ###

def in_box(point,theBox,leeway=100):
    min_x = np.min(theBox.T[0]) - leeway
    max_x = np.max(theBox.T[0]) + leeway
    
    min_y = np.min(theBox.T[1]) - leeway
    max_y = np.max(theBox.T[1]) + leeway
    
    if point[0] >= min_x and point[0] <= max_x and point[1] >= min_y and point[1] <= max_y:
        return True
    else:
        return False
    

    
    
    
#https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h



#https://stackoverflow.com/questions/3731093/is-there-a-python-equivalent-of-matlabs-conv2-function
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)



def pixToIm(pixLett,theBoxPix,winSize,sz=50,reach=12):
    #values are relative to 0,0 at center of screen, will need to shift
    shift_x = winSize[0]/2
    shift_y = winSize[1]/2
    
    #if you dont copy, everything gets all messed up
    theBox = theBoxPix.copy()
    pixLet = pixLett.copy()
    
    pixLet[:,0] += shift_x
    pixLet[:,1] += shift_y
    theBox[:,0] += shift_x
    theBox[:,1] += shift_y
    
    im = np.zeros(winSize)
    for i,(px,py) in enumerate(pixLet):
        px = int(np.round(px))
        py= int(np.round(py))
        
        #need to ignore jumps outside of the Box
        if in_box((px,py),theBox):
            im[px,py] = 1
            
    blur = matlab_style_gauss2D(shape=(sz,sz),sigma=reach)
    gausIm = conv2(im,blur)

    return gausIm, im, blur     
    

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    import pandas as pd
    df = pd.read_pickle('results/TEST_SESSION/TEST_BLOCK/scores_copyDraw_block1.pkl')
    
    pixLet = df.loc['pos_t'][0]
    theBox = df.loc['theBoxPix'][0]
    winSize = df.loc['winSize'][0]
    target = df.loc['templatePix'][0]
    
    print(theBox)
    
    plt.figure()
    plt.plot(pixLet.T[0],pixLet.T[1],label='og')
    plt.plot(theBox.T[0],theBox.T[1],label='og')
    plt.plot(target.T[0],target.T[1],label='og')
    
    sz=50
    reach = 12
    
    #values are relative to 0,0 at center of screen, will need to shift
    shift_x = winSize[0]/2#np.abs(min(theBox.T[0]))
    shift_y = winSize[1]/2#np.abs(min(theBox.T[1]))
    
    pixLet[:,0] += shift_x
    pixLet[:,1] += shift_y
    theBox[:,0] += shift_x
    theBox[:,1] += shift_y
    target[:,0] += shift_x
    target[:,1] += shift_y
    
    
    plt.plot(pixLet.T[0],pixLet.T[1],label='shifted')
    plt.plot(theBox.T[0],theBox.T[1],label='shifted')
    plt.plot(target.T[0],target.T[1],label='shifted')
    
    win = np.array([[0,winSize[1]],
                    [0,0],
                    [winSize[0],0],
                    [winSize[0],winSize[1]]])
    plt.plot(win.T[0],win.T[1],label= 'Window')
    plt.legend()
    
    
    im = np.zeros(winSize)
    
    #detecting linebreaks should go here, currently code doesnt "handle" it though
    ####
    
    for i,(px,py) in enumerate(pixLet):
        px = int(np.round(px))
        py= int(np.round(py))
        
        #need to ignore jumps outside of the Box
        if in_box((px,py),theBox):
            im[px,py] = 1 

    
    blur = matlab_style_gauss2D(shape=(sz,sz),sigma=reach)
    gausIm = conv2(im,blur)
    
    
    plt.figure(figsize=(16,10))
    plt.imshow(gausIm)
    
    
    im = np.zeros(winSize)
    
    for i,(px,py) in enumerate(target):
        px = int(np.round(px))
        py = int(np.round(py))
        
        #need to ignore jumps outside of the Box
        if in_box((px,py),theBox):
            im[px,py] = 1 
            
        
    blur = matlab_style_gauss2D(shape=(sz,sz),sigma=reach)
    gausIm = conv2(im,blur)
    
    
    plt.figure(figsize=(16,10))
    plt.imshow(gausIm)

    gausIm, _,_ = pixToIm(target,theBox,winSize)
    plt.figure(figsize=(16,10))
    plt.imshow(gausIm)