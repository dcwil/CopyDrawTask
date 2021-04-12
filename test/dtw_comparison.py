from pathlib import Path
from dtw_py import dtw_features
from utils import remove_nans_2d
from itertools import permutations
from processing import computeScoreSingleTrial

import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _get_dtw_comparison_dict(alternative_template=False):

    step_patterns = ['MATLAB', 'symmetric1', 'symmetric2', 'asymmetric', 'symmetricP0', 'asymmetricP0',
                     'symmetricP05', 'asymmetricP05', 'symmetricP1',  'asymmetricP1',
                     'symmetricP2', 'asymmetricP2', 'typeIa', 'typeIb', 'typeIc', 'typeId',
                     'typeIas', 'typeIbs', 'typeIcs', 'typeIds', 'typeIIa', 'typeIIb', 'typeIIc', 'typeIId',
                     'typeIIIc', 'typeIVc', 'mori2006']

    results_dir = Path('/home', 'dan', 'Documents', 'AGTANGERMANN', 'CopyDrawEnv', 'CopyDrawTask', 'results', 'S7')
    all_blocks = results_dir.glob("copyDraw*")

    if alternative_template:
        # new smaller template was used for the results I have
        word_to_id = {''.join(w): i
                      for i, w in enumerate(permutations(['1', '2', '3']))}

        tmps = sio.loadmat('../templates/Size_20.mat', simplify_cells=True)

    dtw_dict = {}
    for block in all_blocks:
        block_name = block.name.split('_')[-1]
        print(block_name)
        dtw_dict[block_name] = {}
        for trial in block.glob('*.mat'):

            trial_idx = trial.name.split('_')[1].strip('copyDraw')
            dtw_dict[block_name][trial_idx] = {}

            trial_data = sio.loadmat(trial, simplify_cells=True)

            trace = trial_data['traceLet'].astype(float)  # not sure if the astype is necessary
            template = trial_data['templateLet'].astype(float)

            if alternative_template:
                template = tmps['new_shapes'][word_to_id[trial_data['theWord']]].astype(float)

            # TODO: should this nan removal be noted down somewhere else too? It
            #  does have an impact on calulations
            trace = remove_nans_2d(trace)
            template = remove_nans_2d(template)

            print(trial)
            for step_pattern in step_patterns:
                try:
                    # res = dtw_features(trace, template, step_pattern=step_pattern)
                    print(step_pattern)
                    print(trace.shape)
                    print(template.shape)
                    res = computeScoreSingleTrial(trace,
                                                  template,
                                                  trial_data['trialTime'],
                                                  step_pattern=step_pattern)
                    dtw_dict[block_name][trial_idx][step_pattern] = res
                    dtw_dict[block_name][trial_idx]['trace'] = trace
                    dtw_dict[block_name][trial_idx]['template'] = template
                except ValueError as e:
                    print(e)
                    print(step_pattern)
                    dtw_dict[block_name][trial_idx][step_pattern] = None

    return dtw_dict


def plot_comparison(trial_data, step_pattern='symmetric2', axs=None, trial_no=None):
    if trial_data[step_pattern] is None:
        raise ValueError('dtw failed with this step pattern')
    vals = {}

    rows_list = []
    # work out relative sizes
    for val in ['dt', 'dt_l', 'pathlen']:
        # vals[f'{val}_MATLAB'] = trial_data['MATLAB'][val] / trial_data['MATLAB'][val]
        # vals[f'{val}_{step_pattern}'] = trial_data[step_pattern][val] / trial_data['MATLAB'][val]
        row = {
            'Relative Size (%)': 100 * (trial_data[step_pattern][val] / trial_data['MATLAB'][val]),
            'Attribute': val,
        }
        rows_list.append(row)

    df = pd.DataFrame(rows_list)
    sns.set()
    if axs is None:
        fig, axs = plt.subplots(figsize=(10, 8), ncols=2)

    sns.lineplot(x=trial_data['MATLAB']['w'].T[0], y=trial_data['MATLAB']['w'].T[1], ax=axs[0], label='MATLAB')
    sns.lineplot(x=trial_data[step_pattern]['w'].T[0], y=trial_data[step_pattern]['w'].T[1], ax=axs[0],
                 label=step_pattern)
    sns.barplot(data=df, x='Attribute', y='Relative Size (%)', ax=axs[1])
    # axs[1].set_ylim([50, 150])
    if trial_no is not None:
        plt.suptitle(f'Trial {trial_no}')
    plt.tight_layout()
    return axs


def plot_all_step_patterns(trial_data):
    successful_sps = [sp for sp, v in trial_data.items() if v is not None and sp not in ['MATLAB', 'trace', 'template']]

    print(f'dtw-python failed to dtw with {len(trial_data) - len(successful_sps)}/{len(trial_data)} step patterns')

    fig, axs = plt.subplots(figsize=(10, 10))
    sns.lineplot(x=trial_data['MATLAB']['w'].T[0], y=trial_data['MATLAB']['w'].T[1], ax=axs, label='MATLAB')

    for sp in successful_sps:
        sns.lineplot(x=trial_data[sp]['w'].T[0], y=trial_data[sp]['w'].T[1], ax=axs, label=sp, linewidth=2.5, alpha=0.3)


def subplot_all_step_patterns(trial_data):
    successful_sps = [sp for sp, v in trial_data.items() if v is not None and sp not in ['MATLAB', 'trace', 'template']]

    print(f'dtw-python failed to dtw with {len(trial_data) - len(successful_sps)}/{len(trial_data)} step patterns')

    fig, axs = plt.subplots(figsize=(10, len(successful_sps) * 5), nrows=len(successful_sps))

    for i, sp in enumerate(successful_sps):
        sns.lineplot(x=trial_data['MATLAB']['w'].T[0], y=trial_data['MATLAB']['w'].T[1], ax=axs[i], label='MATLAB',
                     linewidth=2.5, alpha=0.75)
        sns.lineplot(x=trial_data[sp]['w'].T[0], y=trial_data[sp]['w'].T[1], ax=axs[i], label=sp, linewidth=2.5,
                     alpha=0.75)


def plot_closest_N(trial_data, N=10, step_pattern='MATLAB'):
    trace = trial_data['trace']
    template = trial_data['template']
    closest_n = trial_data[step_pattern]['dist_t'].argsort()[:N]

    sns.set()
    fig, axs = plt.subplots(figsize=(14, 14))
    plt.scatter(template.T[0][closest_n], template.T[1][closest_n], label=f'closest_{N}', color='k', s=100, zorder=2)
    plt.plot(trace.T[0], trace.T[1], label='trace', linewidth=5, zorder=1)
    plt.plot(template.T[0], template.T[1], label='template', linewidth=5, zorder=1)
    plt.legend()


def plot_closest_N_comparison(trial_data, N=10, step_patterns=['MATLAB', 'asymmetric', 'symmetric1', 'symmetric2']):
    trace = trial_data['trace']
    template = trial_data['template']

    sns.set()
    fig, axs = plt.subplots(figsize=(14, 14))
    plt.plot(trace.T[0], trace.T[1], label='trace', linewidth=3, zorder=1, color='k')
    plt.plot(template.T[0], template.T[1], label='template', linewidth=3, zorder=1, color='c')

    for sp in step_patterns:
        closest_n = trial_data[sp]['dist_t'].argsort()[:N]
        plt.scatter(template.T[0][closest_n], template.T[1][closest_n], label=f'closest_{N}_{sp}', s=100, zorder=2)

    plt.legend()

if __name__ == '__main__':
    data = _get_dtw_comparison_dict(alternative_template=True)
    df = pd.DataFrame(data)
    df.to_pickle('dtw_comparison_data.pkl')
    # step_patterns = ['symmetric1', 'symmetric2', 'asymmetric', 'symmetricP0', 'asymmetricP0',
    #                  'symmetricP05', 'asymmetricP05', 'symmetricP1', 'asymmetricP1',
    #                  'symmetricP2', 'asymmetricP2', 'typeIa', 'typeIb', 'typeIc', 'typeId',
    #                  'typeIas', 'typeIbs', 'typeIcs', 'typeIds', 'typeIIa', 'typeIIb', 'typeIIc', 'typeIId',
    #                  'typeIIIc', 'typeIVc', 'mori2006']
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # df = pd.read_pickle('dtw_comparison_data.pkl')
    # # plot_comparison(df['block04']['1'])
    #
    # fig, axs = plt.subplots(ncols=2, nrows=8, figsize=(8,20))
    # # axs=axs.flatten()
    # for i, sp in enumerate(step_patterns):
    #     try:
    #         plot_comparison(df['block06']['2'], step_pattern=sp, axs=axs[i])
    #     except ValueError:
    #         continue
    #     except IndexError:
    #         continue
    # plt.show()