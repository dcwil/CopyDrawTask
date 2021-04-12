# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:55:44 2021

@author: Daniel
"""

from dtw import *
from matlab_ports.dtw_matlab import dtw_matlab

import numpy as np


def dtw_features(trace, template, step_pattern='MATLAB'):
    res = {}
    if step_pattern != 'MATLAB':
        alignment = dtw(trace, template, step_pattern=step_pattern, keep_internals=True)

        ns = trace.shape[0]  # n_samples
        nt = template.shape[0]
        C = alignment.localCostMatrix
        D = alignment.costMatrix
        idx_min = np.argmin(C[-1, 1:]) if ns >= nt else np.argmin(C[1:, -1])  # called pathlen briefly in .m files

        res['w'] = np.stack([alignment.index2, alignment.index1], axis=1)  # not sure why these need to be stacked backwards, but they do
        res['pathlen'] = min([idx_min, template.shape[0]])  # bug in matlab code here, chooses wrong template axis
        res['dt'] = alignment.distance

        d_l = D[-1, idx_min] if ns >= nt else D[idx_min, -1]
        res['dt_l'] = d_l
    else:
        res['dt'], res['dt_l'], res['w'], pathlen = dtw_matlab(trace, template)
        res['pathlen'] = min([pathlen, template.shape[0]])  # bug in matlab code here, chooses wrong template axis
        
    return res
