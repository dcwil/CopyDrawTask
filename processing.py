# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:48:38 2021

@author: Daniel
"""

### this is where everything that is done after the experiment should go

import utils
import numpy as np
import dtw_py


def kin_scores(var_pos, delta_t,sub_sampled=False): 
    
    kin_res = {}
    
    kin_res['pos_t'] = var_pos
    
    velocity, velocity_mag = utils.deriv_and_norm(var_pos, delta_t)
    accel, accel_mag = utils.deriv_and_norm(velocity, delta_t)
    jerk, jerk_mag = utils.deriv_and_norm(accel, delta_t)
    
    N = len(var_pos)
    
    #average
    ## divided by number of timepoints, because delta t was used to calc instead of total t
    kin_res['speed'] = np.sum(velocity_mag) / N
    kin_res['acceleration'] = np.sum(accel_mag) / N
    kin_res['velocity_x'] = np.sum(np.abs(velocity[:,0])) / N
    kin_res['velocity_y'] = np.sum(np.abs(velocity[:,1])) / N
    kin_res['acceleration_x'] = np.sum(np.abs(accel[:,0])) / N
    kin_res['acceleration_y'] = np.sum(np.abs(accel[:,1])) / N #matlab code does not compute y values, incorrect indexing
    
    #isj
    # in matlab this variable is overwritten
    isj_ = np.sum((jerk * (delta_t)**3)**2,axis=0)
    kin_res['isj_x'],kin_res['isj_y'] = isj_[0],isj_[1]
    kin_res['isj'] = np.mean(isj_)
    
    kin_res['speed_t'] = velocity * (delta_t)
    kin_res['accel_t']= accel * (delta_t)**2
    kin_res['jerk_t'] = jerk * (delta_t)**3
    
    if sub_sampled:
        kin_res = {f'{k}_sub':v for k,v in kin_res.items()}
    
    return kin_res



def computeScoreSingleTrial(traceLet,template,trialTime):
    
    trial_results = {}
    
    #compute avg delta_t
    delta_t = trialTime/traceLet.shape[0]
    
    ##### Kinematic scores #####
    kin_res = kin_scores(traceLet,delta_t)
    trial_results = {**trial_results, **kin_res }
    
    ## sub sample ##
    traceLet_sub = utils.movingmean(traceLet,5)
    traceLet_sub = traceLet_sub[::3,:] # take every third point
    kin_res_sub = kin_scores(traceLet_sub,(delta_t)*3,sub_sampled=True)
    trial_results = {**trial_results, **kin_res_sub}
    
    ##### dtw #####
    # print(f'template has size: {self.frame_elements["template"].verticesPix.shape}')
    # print(f'stim has shape: {self.current_stimulus.shape}')
    # print(f'trace has shape: {traceLet.shape}')
    
    # think about units here bound to run into issues!
    dtw_res = dtw_py.dtw_features(traceLet, template)
    trial_results = {**trial_results, **dtw_res}
    
    
    #misc
    # +1 on the pathlens bc matlab indexing
    trial_results['dist_t'] = np.sqrt(np.sum((template[trial_results['w'].astype(int)[:trial_results['pathlen']+1,0],:] - trial_results['pos_t'][trial_results['w'].astype(int)[:trial_results['pathlen']+1,1]])**2,axis=1))
    
    # normalize distance dt by length of copied template (in samples)
    trial_results['dt_norm'] = trial_results['dt_l'] / (trial_results['pathlen']+1)
    
    # get length of copied part of the template (in samples)
    trial_results['len'] = (trial_results['pathlen']+1) / len(template)
    
    
    return trial_results


