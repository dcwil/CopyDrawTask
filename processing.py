# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:48:38 2021

@author: Daniel
"""

# this is where everything that is done after the experiment should go
# TODO: varying screen size etc can scale the recorded template so it no longer
# aligns with the original. Write a function ( align()? ) that lines them up,
# using the fact that the start point in both should always be the same
import utils
import numpy as np
import dtw_py
import scipy.io as sio
import pandas as pd

from matlab_ports.getScore import get_score_from_trace
from pathlib import Path


def kin_scores(var_pos, delta_t, sub_sampled=False):
    
    kin_res = {}
    
    kin_res['pos_t'] = var_pos
    
    velocity, velocity_mag = utils.deriv_and_norm(var_pos, delta_t)
    accel, accel_mag = utils.deriv_and_norm(velocity, delta_t)
    jerk, jerk_mag = utils.deriv_and_norm(accel, delta_t)
    
    N = len(var_pos)
    
    # average
    # divided by number of timepoints,
    # because delta t was used to calc instead of total t
    kin_res['speed'] = np.sum(velocity_mag) / N
    kin_res['acceleration'] = np.sum(accel_mag) / N
    kin_res['velocity_x'] = np.sum(np.abs(velocity[:, 0])) / N
    kin_res['velocity_y'] = np.sum(np.abs(velocity[:, 1])) / N
    kin_res['acceleration_x'] = np.sum(np.abs(accel[:, 0])) / N
    # matlab code does not compute y values, incorrect indexing
    kin_res['acceleration_y'] = np.sum(np.abs(accel[:, 1])) / N
    
    # isj
    # in matlab this variable is overwritten
    isj_ = np.sum((jerk * delta_t ** 3) ** 2, axis=0)
    kin_res['isj_x'], kin_res['isj_y'] = isj_[0], isj_[1]
    kin_res['isj'] = np.mean(isj_)
    
    kin_res['speed_t'] = velocity * delta_t
    kin_res['accel_t'] = accel * delta_t ** 2
    kin_res['jerk_t'] = jerk * delta_t ** 3
    
    if sub_sampled:
        kin_res = {f'{k}_sub': v for k, v in kin_res.items()}
    
    return kin_res


def computeScoreSingleTrial(traceLet, template, trialTime, step_pattern='MATLAB'):
    
    trial_results = {}
    
    # compute avg delta_t
    delta_t = trialTime/traceLet.shape[0]
    # trial_results['delta_t'] = delta_t

    # Kinematic scores
    kin_res = kin_scores(traceLet, delta_t)
    trial_results = {**trial_results, **kin_res}
    
    # sub sample
    traceLet_sub = utils.movingmean(traceLet, 5)
    traceLet_sub = traceLet_sub[::3, :]  # take every third point
    kin_res_sub = kin_scores(traceLet_sub, delta_t*3, sub_sampled=True)
    trial_results = {**trial_results, **kin_res_sub}
    
    # dtw
    dtw_res = dtw_py.dtw_features(traceLet, template, step_pattern=step_pattern)
    dtw_res['pathlen'] = min([dtw_res['pathlen'], template.shape[0]])
    trial_results = {**trial_results, **dtw_res}

    # misc
    # +1 on the pathlens bc matlab indexing
    # this is a horrible one-liner, it needs breaking up

    w = trial_results['w'].astype(int)
    pos_t = trial_results['pos_t']  # the trace
    pathlen = trial_results['pathlen']
    tmp1 = template[w[:pathlen + 1, 0], :]
    tmp2 = pos_t[w[:pathlen + 1, 1]]
    trial_results['dist_t'] = np.sqrt(np.sum((tmp1 - tmp2) ** 2, axis=1))
    
    # normalize distance dt by length of copied template (in samples)
    trial_results['dt_norm'] = trial_results['dt_l'] / (trial_results['pathlen']+1)
    
    # get length of copied part of the template (in samples)
    trial_results['len'] = (trial_results['pathlen']+1) / len(template)
    
    return trial_results

# processing sessions/blocks/trials


def save_processed_trial(res, trial_path, verbose=True):
    """ Save a single trial as a pickle """
    block_dir = trial_path.parents[0]
    fname = f"processed_scores_trial_{res['ix_trial']}" \
            f"_block{res['ix_block']}.pkl"

    df = pd.Series(res)
    if verbose:  # eventually this should be put into a log file?
        print(f"Saving trial {res['ix_trial']} block {res['ix_block']}")
    df.to_pickle(block_dir / fname)


def process_trial(trial_path, legacy=False, sf=None, save=True):
    """ Actual trial post processing takes place here. Set legacy=True when
    processing old matlab data for verification etc. sf is scaling factor,
     calculated from data if not given (not needed for legacy data, for current data
     assumes that startpoint of template & trace are the same)

     set save=False to return pandas series of trial, True to save series to pkl.



     When processing old matlab data be aware that template size is currently
      hardcoded (change this?)."""
    res = {}

    if legacy:
        mat = sio.loadmat(trial_path, simplify_cells=True)
        res['ptt'] = mat['preTrialTime']
        res['startTStamp'] = mat['trialStart']  # are these the same thing..?

        # get idxs from fname
        fname = Path(trial_path).name
        res['ix_block'] = fname.split('_')[-1].strip('block.mat')
        res['ix_trial'] = fname.split('_')[1].strip('copyDraw')

        # data
        res['trace_let'] = mat['traceLet'].astype(float)
        res['template'] = mat['templateLet'].astype(float)
        res['trial_time'] = mat['trialTime']
        score = mat['score']  # this will be zero

        # new smaller template was used for the results I have
        from itertools import permutations
        word_to_id = {''.join(w): i
                      for i, w in enumerate(permutations(['1', '2', '3']))}

        tmps = sio.loadmat('templates/Size_20.mat', simplify_cells=True)
        tmp_late = tmps['new_shapes'][word_to_id[mat['theWord']]].astype(float)

        # TODO: remove nans from trace (this should be moved into the overall pipeline)
        res['trace_let'] = utils.remove_nans_2d(res['trace_let'])
    else:

        df = pd.read_pickle(trial_path)

        res['ptt'] = df.ptt
        res['startTStamp'] = df.start_t_stamp
        res['ix_trial'] = df.ix_trial
        res['ix_block'] = df.ix_block
        # both trace and template should have the same starting point
        # use this to scale things
        sf = sf or (df.trace_let[0] / df.template[0])
        scaled_template = df.template * sf
        res['scaled_template'] = scaled_template
        res['sf'] = sf
        res['trace_let'] = df.trace_let
        res['trial_time'] = df.trial_time

        # compute old style scoring (now set to 0 in matlab)
        score = get_score_from_trace(res['trace_let'], scaled_template,
                                     df.theBoxPix, df.winSize)

        tmp_late = scaled_template

    # do dtw etc
    scores = computeScoreSingleTrial(res['trace_let'],
                                     tmp_late,
                                     res['trial_time'])

    res = {**res, **scores, 'score': score}

    if save:
        save_processed_trial(res, trial_path)

    return pd.Series(res)


def process_session(path_to_session, ignore=None, ftype='pkl', verbose: bool = True,
                    legacy: bool = False, save: bool = False, compile: bool = True):
    """ Helper for looping over every trial in a block and every block in a
     session. Assumes any folders not passed in ignore are blocks (but it
     knows to ignore 'info_runs').

     Set legacy=True when processing matlab data for verification.
     """

    save_individual = True if save and not compile else False

    session_dir = Path(path_to_session)

    # should this be written differently?
    ignore = ignore or []
    ignore.append('info_runs')
    ignore.append('processed')

    print(f'processing {session_dir.name}')

    # list blocks (excluding any from ignore)
    all_blocks = [f for f in session_dir.rglob('*/**')
                  if not any([d in f.name for d in ignore])]

    assert len(all_blocks) > 0, f'No blocks found in {session_dir}'

    # loop over blocks
    session = []
    for block_path in all_blocks:

        # scoring each block and saving
        session.append(process_block(block_path, ftype=ftype, legacy=legacy, verbose=verbose, save=save_individual))

    # compile and return compiled session
    if not save and compile:
        session = utils.unnest(session)
        return pd.DataFrame(session)
    # compile and save compiled session
    elif save and compile:
        session = utils.unnest(session)
        df = pd.DataFrame(session)
        df.to_pickle(f'{session_dir.name}_new_scores_raw.p')
    # individual and save individual trials
    elif save and not compile:
        pass
    elif not save and not compile:
        return session


def process_block(block_path, ftype='pkl', legacy: bool = False, verbose: bool = True, save: bool = True):
    """ Helper for running process trial on each trial in a given block """

    if verbose:
        print(f'Processing {block_path}')

    all_trials = [f for f in block_path.rglob(f'tscore*.{ftype}')]

    block = []
    for trial_path in all_trials:
        block.append(process_trial(trial_path, legacy=legacy, save=save))

    return block


def compile_session(path_to_session=None, output_name=None, process=False, **kwargs):
    """ for compiling a session into a single data structure (dataframe), processes trials if process is True,
     otherwise it assumes they have been processed. Remember to pass important args for process session: legacy,
      ignore, ftype etc"""

    if not isinstance(path_to_session, Path):
        path_to_session = Path(path_to_session)

    if process:
        process_session(path_to_session=path_to_session, **kwargs)

    series_list = []

    for block_path in path_to_session.glob('copyDraw*block*'):
        for proc_trial in block_path.glob('processed*pkl'):
            se = pd.read_pickle(proc_trial)
            series_list.append(se)

    if output_name is None:
        output_name = f'{path_to_session.name}_scores_raw.pkl'

    proc_df = pd.DataFrame(series_list)  # will need to separate the optimpath/w
    proc_df.to_pickle(output_name)
