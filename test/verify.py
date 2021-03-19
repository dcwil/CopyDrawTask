# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:09:25 2021

@author: Daniel
"""
# rewrite into a function?
import pandas as pd
import numpy as np
import pickle5 as pkl5
import processing as pro

from pathlib import Path
from utils import check_with_tol

results_dir = Path('../results')
sample_session = 'S7'
session_dir = results_dir.joinpath(sample_session)
scores_path = session_dir.joinpath('processed', 'scores_raw.p')
opti_paths_path = session_dir.joinpath('processed', 'dtw_opti_paths_df.p')
assert scores_path.exists()
assert opti_paths_path.exists()

# Get matlab results
scores_df = pd.read_pickle(scores_path)
with open(opti_paths_path, 'rb') as h:
    opti_paths = pkl5.load(h)

# Set to None and processed data will be collected and saved
proc_scores_fname = 'new_scores_raw.pkl'

# set to true if processing hasn't been done or needs to be overwritten
process = False

# do the processing (this is slowish, be aware)
if process:
    pro.process_session(path_to_session=session_dir,
                        ignore=['processed'],
                        legacy=True, ftype='mat')

# Collect processed results
if proc_scores_fname is None:
    series_list = []

    for block_path in session_dir.glob('copyDraw*block*'):
        for proc_trial in block_path.glob('processed*pkl'):
            se = pd.read_pickle(proc_trial)
            series_list.append(se)

    proc_df = pd.DataFrame(series_list)  # will need to separate the optimpath/w
    proc_df.to_pickle('new_scores_raw.pkl')
else:
    proc_df = pd.read_pickle(Path('..', proc_scores_fname))

# compare results - verification
# go trial by trial
rows_list = []
# for ix_block in proc_df['ix_block']:
#     for ix_trial in proc_df[proc_df['ix_block'] == ix_block]['ix_trial']:
for i, df_row in proc_df.iterrows():

    ix_block = df_row['ix_block']
    ix_trial = df_row['ix_trial']

    new_trial = proc_df[(proc_df['ix_block'] == ix_block) &
                        (proc_df['ix_trial'] == ix_trial)]

    old_trial = scores_df[(scores_df['ix_block'] == int(ix_block)) &
                          (scores_df['ix_trial'] == int(ix_trial))]

    old_trial_opti = opti_paths[(opti_paths['block'] == ix_block) &
                                (opti_paths['trial'] == ix_trial.zfill(2)) &
                                (opti_paths['session'] == sample_session)]

    metadata = ['ptt', 'pos_t']

    kindata = ['speed', 'speed_sub', 'velocity_x', 'velocity_x_sub',
               'velocity_y', 'velocity_y_sub', 'isj', 'isj_sub', 'isj_x',
               'isj_x_sub', 'isj_y', 'isj_y_sub', 'acceleration',
               'acceleration_sub', 'acceleration_x', 'acceleration_x_sub',
               'acceleration_y', 'acceleration_y_sub', 'pos_t_sub',
               'speed_t', 'speed_t_sub', 'accel_t', 'accel_t_sub', 'jerk_t',
               'jerk_t_sub']

    dtwdata = ['dt', 'dt_l', 'w', 'pathlen', 'len', 'dt_norm', 'dist_t']

    # each trial will be a row in a df
    row = {'ix_block': ix_block, 'ix_trial': ix_trial}

    # check kinematic scores
    for d in kindata:
        try:
            # still need to handle the accel issue
            old_val = old_trial[d].values
            new_val = new_trial[d].values
            check = check_with_tol(old_val, new_val).all()
        except ValueError:
            old_val = old_trial[d].values[0]
            new_val = new_trial[d].values[0]
            check = check_with_tol(old_val, new_val).all()

        if check:
            row[d] = 'Matched!'
        else:
            row[f'old_{d}'] = old_val
            row[f'new_{d}'] = new_val

    for d in dtwdata:
        try:
            if d == 'dist_t':
                old_val = np.squeeze(old_trial[d].values[0])
                new_val = new_trial[d].values[0]

            elif d == 'pathlen':  # matlab indexing
                old_val = old_trial[d].values[0]
                new_val = new_trial[d].values[0] + 1

            elif d == 'w':  # saved something wrong, this is optimal path
                # Is there a better way to handle these extra arrays?
                old_val = old_trial_opti['optimal_path'].values[0][0][0][0]
                new_val = new_trial[d].values[0][0]+1

            else:
                old_val = old_trial[d].values
                new_val = new_trial[d].values

            check = check_with_tol(old_val, new_val).all()

        except ValueError:
            old_val = old_trial[d].values[0]
            new_val = new_trial[d].values[0]
            check = check_with_tol(old_val, new_val).all()

        if check:
            row[d] = 'Matched!'
        else:
            row[f'old_{d}'] = old_val
            row[f'new_{d}'] = new_val

    rows_list.append(row)

checked_df = pd.DataFrame(rows_list)
checked_df.to_pickle('checked_scores.pkl')

# p = results_dir / 'SAMPLE_SESSION' / 'SAMPLE_BLOCK'
#
# # turn matlab into pickles
# if not p.exists():
#
#     p.mkdir(parents=True)
#     # paths
#     sample_scores_path = sample_data_dir / 'scores_copyDraw_block01.mat'
#
#     # this is for one block
#     scores = sio.loadmat(sample_scores_path, simplify_cells=True)
#
#     # get first trial from scores
#     trial = scores['block_perf'][0]
#
#     #sort out the renaming
#     trial['start_t_stamp'] = trial.pop('startTStamp')
#     trial['trace_let'] = trial.pop('pos_t')
#
#     # turn to df and save
#     df = pd.Series(trial)
#
#     trial_fname = 'tscore_1copyDraw_block1.pkl'
#     df.to_pickle(p / trial_fname)
#
#
# sample_proc_scores_path = sample_data_dir / 'processed_scores_trial_01_block01.mat'
# proc_scores = sio.loadmat(sample_proc_scores_path, simplify_cells=True)
# # get processed trial
# proc_trial = proc_scores['scores']
#
# # process sample data
# cd = CopyDraw('./')
# cd.process_session(session_name='SAMPLE_SESSION')




### Pretty sure none of this code is needed anymore! ###

# scores = sio.loadmat('sample_data/scores_copyDraw_block01.mat',simplify_cells=True)
# templates = sio.loadmat('templates/Size_20.mat',simplify_cells=True)
#
# test_trial = scores['block_perf'][5]
#
# test_trace = test_trial['pos_t']
# test_template = templates['new_shapes'][4]
#
# ### check that we have the right template ###
# ### cant find any indicator of the template in the sample data scores ###
# plt.figure()
# plt.plot(test_trace[:,0],test_trace[:,1],label='trace')
# plt.plot(test_template[:,0],test_template[:,1],label='template')
# plt.legend()
# plt.show()
# ### visually inspect! ###
# #trial 4 goes with template 1 d:[360,518]
# #trial 5 goes with template 4 d_l:[360,515]
# #trial 6 goes with template 0
#
# #https://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
# def search_sequence_numpy(arr,seq):
#     """ Find sequence in an array using NumPy only.
#
#     Parameters
#     ----------
#     arr    : input 1D array
#     seq    : input 1D array
#
#     Output
#     ------
#     Output : 1D Array of indices in the input array that satisfy the
#     matching of input sequence in the input array.
#     In case of no match, an empty list is returned.
#     """
#
#     # Store sizes of input array and sequence
#     Na, Nseq = arr.size, seq.size
#
#     # Range of sequence
#     r_seq = np.arange(Nseq)
#
#     # Create a 2D array of sliding indices across the entire length of input array.
#     # Match up with the input sequence & get the matching starting indices.
#     M = (arr[np.arange(Na-Nseq+1)[:,None] + r_seq] == seq).all(1)
#
#     # Get the range of those indices as final output
#     if M.any() >0:
#         return np.where(np.convolve(M,np.ones((Nseq),dtype=int))>0)[0]
#     else:
#         return []         # No match found
#
# # https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
# def find_nearest(array, value):
#     #array = np.nan_to_num(array, copy=True, nan=dt_l/2)
#     idx = (np.abs(array - value)).argmin()
#     return array.flatten()[idx]
#
# #rabinerJuangStepPattern(type,slope_weighting="d",smoothed=False)
#
# step_patterns = ['symmetric1','symmetric2','asymmetric','symmetricP0','asymmetricP0',
#                  'symmetricP05','asymmetricP05','symmetricP1', 'asymmetricP1',
#                  'symmetricP2','asymmetricP2','typeIa','typeIb','typeIc','typeId',
#                  'typeIas','typeIbs','typeIcs','typeIds','typeIIa','typeIIb','typeIIc','typeIId',
#                  'typeIIIc','typeIVc','mori2006']
#
#
#
#
# #i think w/optim_path is alignment.index1 and 2
# # pathlen/idx_min is probably jmin? (adapt for ptyhon counting though)
# # also pathlen is sometimes "off" by a small amount (<10)
# # d_l is indexed from costmatrix, correct pattern should contain d_l or something close?
#
#
# print('Testing')
# w = test_trial['w'] #also called optim path
# w_ = {
#     'w_x' : w[:,0]-1,
#     'w_y' : w[:,1]-1
#     }
# pathlen = test_trial['pathlen'] # also called idx min
# dt = test_trial['dt']
# dt_l = test_trial['dt_l']
#
# rows_list = []
#
# for sp in step_patterns:
#     #try:
#
#
#     alignment = dtw(test_template,test_trace,step_pattern=sp,keep_internals=True)
#     new_w = np.stack([alignment.index1,alignment.index2],axis=1)
#     new_w_s = np.stack([alignment.index1s,alignment.index2s],axis=1)
#     new_pathlen = min([alignment.jmin,test_template.shape[0]])
#     new_d = alignment.distance
#     new_dl = alignment.costMatrix[-1,alignment.jmin-1]
#
#     row = {
#         'w_nrows_difference':w.shape[0] - new_w.shape[0],
#         'w_nrows_difference *':w.shape[0] - new_w_s.shape[0],
#         'pathlen_difference':pathlen - new_pathlen,
#         'query':'template',
#         'reference':'trace',
#         'dt_difference':dt - new_d,
#         'dt_difference %':100*np.abs((dt - new_d))/dt,
#         'dt in costmatrix':True if dt in alignment.costMatrix else find_nearest(alignment.costMatrix,dt)/dt,
#         'dt_l in costmatrix':True if dt_l in alignment.costMatrix else find_nearest(alignment.costMatrix,dt_l)/dt_l,
#         'stepPatten':sp,
#         #look for index sequence in w
#         # 'w[:,0] len match index1': len(search_sequence_numpy(w[:,0]-1,alignment.index1)),
#         # 'w[:,0] len match index1 rev': len(search_sequence_numpy(seq=w[:,0]-1,arr=alignment.index1)),
#         # 'w[:,0] len match index2': len(search_sequence_numpy(w[:,0]-1,alignment.index2)),
#         # 'w[:,0] len match index1s': len(search_sequence_numpy(w[:,0]-1,alignment.index1s)),
#         # 'w[:,0] len match index2s': len(search_sequence_numpy(w[:,0]-1,alignment.index2s)),
#         # 'w[:,1] len match index1': len(search_sequence_numpy(w[:,1]-1,alignment.index1)),
#         # 'w[:,1] len match index2': len(search_sequence_numpy(w[:,1]-1,alignment.index2)),
#         # 'w[:,1] len match index1s': len(search_sequence_numpy(w[:,1]-1,alignment.index1s)),
#         # 'w[:,1] len match index2s': len(search_sequence_numpy(w[:,1]-1,alignment.index2s)),
#
#         }
#
#
#
#     alignment_idxs = {
#         'idx1':alignment.index1,
#         'idx1s':alignment.index1s,
#         'idx2':alignment.index2,
#         'idx2s':alignment.index2s
#         }
#
#     for al_key,al_index in alignment_idxs.items():
#         for w_key,w_vals in w_.items():
#             if len(w_vals) > len(al_index):
#                 n_match = len(search_sequence_numpy(arr=w_vals, seq=al_index))
#             else:
#                 n_match = len(search_sequence_numpy(arr=al_index, seq=w_vals))
#
#             row[f'n_matches with {w_key} & {al_key}'] = n_match
#
#
#     rows_list.append(row)
#
#
#
#     alignment = dtw(test_trace,test_template,step_pattern=sp,keep_internals=True)
#     new_w = np.stack([alignment.index1,alignment.index2],axis=1)
#     new_w_s = np.stack([alignment.index1s,alignment.index2s],axis=1)
#     new_pathlen = min([alignment.jmin,test_template.shape[0]])
#     new_d = alignment.distance
#     new_dl = alignment.costMatrix[-1,alignment.jmin-1]
#
#     row = {
#         'w_nrows_difference':w.shape[0] - new_w.shape[0],
#         'w_nrows_difference *':w.shape[0] - new_w_s.shape[0],
#         'pathlen_difference':pathlen - new_pathlen,
#         'query':'trace',
#         'reference':'template',
#         'dt_difference':dt - new_d,
#         'dt_difference %':100*np.abs((dt - new_d))/dt,
#         'dt in costmatrix':True if dt in alignment.costMatrix else find_nearest(alignment.costMatrix,dt)/dt,
#         'dt_l in costmatrix':True if dt_l in alignment.costMatrix else find_nearest(alignment.costMatrix,dt_l)/dt_l,
#         'stepPatten':sp,
#         # 'w[:,0] len match index1': len(search_sequence_numpy(w[:,0]-1,alignment.index1)),
#         # 'w[:,0] len match index2': len(search_sequence_numpy(w[:,0]-1,alignment.index2)),
#         # 'w[:,0] len match index1s': len(search_sequence_numpy(w[:,0]-1,alignment.index1s)),
#         # 'w[:,0] len match index2s': len(search_sequence_numpy(w[:,0]-1,alignment.index2s)),
#         # 'w[:,1] len match index1': len(search_sequence_numpy(w[:,1]-1,alignment.index1)),
#         # 'w[:,1] len match index2': len(search_sequence_numpy(w[:,1]-1,alignment.index2)),
#         # 'w[:,1] len match index1s': len(search_sequence_numpy(w[:,1]-1,alignment.index1s)),
#         # 'w[:,1] len match index2s': len(search_sequence_numpy(w[:,1]-1,alignment.index2s)),
#         }
#
#
#
#     alignment_idxs = {
#         'idx1':alignment.index1,
#         'idx1s':alignment.index1s,
#         'idx2':alignment.index2,
#         'idx2s':alignment.index2s
#         }
#
#     for al_key,al_index in alignment_idxs.items():
#         for w_key,w_vals in w_.items():
#             if len(w_vals) > len(al_index):
#                 n_match = len(search_sequence_numpy(w_vals, al_index))
#             else:
#                 n_match = len(search_sequence_numpy(al_index, w_vals))
#
#
#             row[f'n_matches with {w_key} & {al_key}'] = n_match
#     rows_list.append(row)
#
#     # except:
#     #     print(f'failed with {sp}')
#
# df = pd.DataFrame(rows_list)
#
# # inspecting df suggests none of the step patterns correctly get w (but symmetric1 gets a lot of the other stuff)
# # buuuut plottting thigns like this might bemore informative:
#
# fig, axs = plt.subplots(nrows=9,ncols=3,figsize=(8,16))
#
# axs = axs.flatten()
#
# for i,sp in enumerate(step_patterns):
#     alignment = dtw(test_template,test_trace,step_pattern=sp,keep_internals=True)
#     axs[i].plot(w[:,0]-1,w[:,1]-1,label='w',alpha=0.5)
#     axs[i].plot(alignment.index1,alignment.index2,label='new',alpha=0.5)
#     axs[i].set_title(sp)
#     axs[i].legend()
# plt.tight_layout()
# #plt.savefig('step_patterns.pdf', format='pdf')
