# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:55:44 2021

@author: Daniel
"""

from dtw import *
from matlab_ports.dtw_matlab import dtw_matlab

import numpy as np

def dtw_features(trace,template,step_pattern='MATLAB'):
    res = {}
    if step_pattern != 'MATLAB':
        alignment = dtw(trace,template,step_pattern=step_pattern, keep_internals=True)
        
        idx_min = np.argmin(alignment.localCostMatrix[-1,1:]) #called pathlen briefly in .m files
        
        res['w'] = np.stack([alignment.index1,alignment.index2], axis=1)
        res['pathlen'] = min([idx_min,template.shape[0]]) #bug in matlab code here, chooses wrong template axis
        res['dt'] = alignment.distance
        res['dt_l'] = alignment.costMatrix[-1,idx_min]
    else:
        res['dt'],res['dt_l'],res['w'],pathlen = dtw_matlab(trace, template)
        res['pathlen'] = min([pathlen,template.shape[0]]) #bug in matlab code here, chooses wrong template axis
        
    return res