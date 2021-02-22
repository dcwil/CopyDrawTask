# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:57:07 2021

@author: Daniel
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import scipy.io as sio

from pathlib import Path

templates = sio.loadmat('templates/Size_20.mat',simplify_cells=True)
test_template = templates['new_shapes'][4]

plt.figure()
plt.plot(test_template.T[0],test_template.T[1],lineWidth=5)
ax =plt.axes()
#ax.set_facecolor('black')
ax.xaxis.set_major_locator(ticker.NullLocator()) 
ax.yaxis.set_major_locator(ticker.NullLocator()) 
plt.savefig('test.png',format='png',bbox_inches='tight',transparent=True)


#move to utils?
def template_to_image(template,fname,path,**kwargs):
    plt.figure(figsize=(16,10))
    plt.plot(template.T[0],template.T[1],**kwargs)
    ax = plt.axes()
    ax.xaxis.set_major_locator(ticker.NullLocator()) 
    ax.yaxis.set_major_locator(ticker.NullLocator()) 
    
    fullpath = Path(path,f'{fname}.png' )
    plt.savefig(fullpath,format='png',bbox_inches='tight',transparent=True,dpi=300)
    return fullpath