# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:14:23 2020

@author: Daniel
"""

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pandas as pd

from copydraw import CopyDraw


scores = sio.loadmat('sample_data/scores_copyDraw_block01.mat',simplify_cells=True)
templates = sio.loadmat('templates/Size_20.mat',simplify_cells=True)

test_trial = scores['block_perf'][5]

test_trace = test_trial['pos_t']
test_template = templates['new_shapes'][4]

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

CD = CopyDraw(None,'./',n_trials=2,finishWhenRaised=True)

for t in [2.2,2.7,3,6.00555]:
    print(t)
    CD.check_results(test_trial,test_template,trial_time=t)