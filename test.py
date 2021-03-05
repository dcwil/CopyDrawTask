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

# # move this outta here
# def check_results(self, sample_data, template, trial_time=2.7*3):
#     trial_results = {}
#
#     pos_t = sample_data['pos_t'].copy().astype(float)
#
#     delta_t = trial_time / len(pos_t)
#
#     ##### Kinematic scores #####
#     kin_scores = self.kin_scores(pos_t, delta_t)
#     trial_results = {**trial_results, **kin_scores}
#
#     ## sub sample ##
#     mouse_pos_pix_sub = self.movingmean(pos_t, 5)
#     mouse_pos_pix_sub = mouse_pos_pix_sub[::3, :]  # take every third point
#     kin_scores_sub = self.kin_scores(mouse_pos_pix_sub, delta_t * 3, sub_sampled=True)
#     trial_results = {**trial_results, **kin_scores_sub}
#
#     ##### dtw #####
#     dtw_res = self.dtw_features(pos_t, template)
#     trial_results = {**trial_results, **dtw_res}
#
#     ##### misc #####
#     # +1 on the pathlens bc matlab indexing
#     trial_results['dist_t'] = np.sqrt(np.sum((template[
#                                               trial_results['w'].astype(int)[:trial_results['pathlen'] + 1, 0], :] -
#                                               pos_t[trial_results['w'].astype(int)[:trial_results['pathlen'] + 1,
#                                                     1]]) ** 2, axis=1))
#
#     # normalize distance dt by length of copied template (in samples)
#     trial_results['dt_norm'] = trial_results['dt_l'] / (trial_results['pathlen'] + 1)
#
#     # get length of copied part of the template (in samples)
#     trial_results['len'] = (trial_results['pathlen'] + 1) / len(template)
#
#     ### no way to calculate these ###
#     # its the time between touching the cyan square and starting drawing (i think)
#     trial_results['ptt'] = sample_data['ptt']
#     trial_results['ix_block'] = sample_data['ix_block']
#     trial_results['ix_trial'] = sample_data['ix_trial']
#     trial_results['start_t_stamp'] = sample_data['start_t_stamp']
#     trial_results['stim'] = sample_data['stim']
#
#     ### weird stuff and index changes ###
#     # trial_results['pos_t'] = trial_results['pos_t'].astype('<u2')
#     trial_results['pathlen'] += 1
#     trial_results['w'] += 1
#     # process delta t error
#     trial_results['acceleration'] *= delta_t
#     trial_results['acceleration_x'] *= delta_t
#     trial_results['acceleration_y'] *= delta_t
#     trial_results['acceleration_sub'] *= delta_t * 3
#     trial_results['acceleration_x_sub'] *= delta_t * 3
#     trial_results['acceleration_y_sub'] *= delta_t * 3
#
#     def check_with_tol(x, y, tol=0.0001):
#         return np.abs(x - y) < tol
#
#     print('calculated results - beginning checks')
#     for key, data in trial_results.items():
#         try:
#             if isinstance(data, np.ndarray):
#                 check = check_with_tol(data, sample_data[key]).all()
#
#                 if not check:
#                     print(f'{key} FAILED')
#             else:
#
#                 check = check_with_tol(data, sample_data[key])
#
#                 if not check:
#                     print(f'{key} FAILED: {data} should be {sample_data[key]}')
#         except:
#             print(f'encountered error with {key}')
#
#     return trial_results
