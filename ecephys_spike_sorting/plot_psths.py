import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from collections import defaultdict
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

from utils import ioutils
from utils import figutils
from utils import styleutils
from utils import phy_utils
from utils import psth_utils
from utils import sort_utils
import configs

sns.reset_defaults()
styleutils.update_mpl_params(fontsize=14, linewidth=1)

# args
mouse = configs.NPIX1
mouse_path = os.path.join(configs.BASE_PATH, mouse.path)
dates = list(mouse.exp_params.keys())
trial_duration = mouse.duration

bool_heatmap = 1
bool_psths = 0

# constants
celltypes_names = [
    'ss',
    '_ss', # cliques
    'cs',
    'mli',
    '>40']
fs = 30000
ccg_win_size = 80 # ms
ccg_bin_size = 1 # ms
ccg_smooth_win = 3 # ms
default_refractory_period = 1 # ms

psthb_cs = 1000
psthb = 250
filter_window = int(0.5 / (psthb * 1e-3))
zscore_sort_thres_pos = 1
zscore_sort_thres_neg = -1
sort_time_window = [5, 10]

intervals = {'ss': 1, 'rest': 10}
psthw = np.array([0, trial_duration * 1000])
xticks = np.array([5, 7, 9, 10])
xticklabels = ['O', '+2', '', 'W']
times_in_bins = np.array(xticks) / (psthb * 1e-3)
xlabel='Time (s)'
ylabel='FR (spk/s)'
stim_colors = ['darkgreen', 'green', 'red', 'magenta', 'turquoise']

base_folders = []
catgt_folders = []
phy_folders = []
for date in dates:
    try:
        base_folders.append(os.path.join(mouse_path, date))
        catgt_folder = glob.glob(os.path.join(mouse_path, date, 'catgt*'))[0]
        catgt_folders.append(catgt_folder)

        catgt_imec_folder = glob.glob(os.path.join(catgt_folder, '*imec0'))[0]
        data_folder = os.path.join(catgt_imec_folder, 'imec0_ks2')
        phy_folders.append(data_folder)
    except:
        pass

for ix in range(len(catgt_folders)):
    catgt_folder = catgt_folders[ix]
    base_folder = base_folders[ix]
    phy_folder = phy_folders[ix]
    behavior_folder = os.path.join(base_folder, 'behavior')
    processed_folder = os.path.join(base_folder, 'processed')
    fig_folder = os.path.join(base_folder, 'FIGURES')

    package = ioutils.pload(os.path.join(processed_folder, 'package'))
    channel_pos = np.load(os.path.join(phy_folder, 'channel_positions.npy'))
    channel_map = np.load(os.path.join(phy_folder, 'channel_map.npy')).flatten()
    assert np.array_equal(channel_map, np.arange(len(channel_map)))
    metrics = pd.read_csv(os.path.join(phy_folder, 'metrics.csv'))
    cluster_ids = metrics['cluster_id'].to_numpy()
    cluster_depth = channel_pos[metrics['peak_channel'].to_numpy(), 1]

    ''' load data'''
    # get timing of trials per stim condition
    onset_files = glob.glob(os.path.join(catgt_folder, r'*xa_1*'))
    offset_files = glob.glob(os.path.join(catgt_folder, r'*xia_1*'))
    assert len(onset_files) == 1 and len(offset_files) == 1

    onsets = np.loadtxt(onset_files[0])
    offsets = np.loadtxt(offset_files[0])

    if 'NPIX1' in phy_folder and '11.14' in phy_folder:
        onsets = np.delete(onsets, 25) #npix1, 11/14
        offsets = np.delete(offsets, 25) #npix1, 11/14
    if 'NPIX1' in phy_folder and '11.15' in phy_folder:
        onsets = onsets[np.r_[True, np.diff(onsets) > 10]] #npix1, 11/15
    assert len(onsets) == len(offsets)
    n_trials_nidaq = len(onsets)

    stimuli = list(mouse.stimuli_params.keys())
    npy_files = sorted(glob.glob(os.path.join(behavior_folder, f'*.npy')))
    n_trials_npy = len(npy_files)
    assert n_trials_npy == n_trials_nidaq, print(n_trials_npy, n_trials_nidaq)

    stim_epochs_dict = {}
    for stim in stimuli:
        trial_ixs = np.where([stim in x for x in npy_files])[0]
        if len(trial_ixs):
            stim_onsets = onsets[trial_ixs]
            stim_offsets = offsets[trial_ixs]
            stim_epochs = np.array([stim_onsets, stim_offsets])
            stim_epochs_dict[stim] = stim_epochs.T[:, 0]
    stim_names = [x for x in stim_epochs_dict.keys()]
    stim_trial_times = [x for x in stim_epochs_dict.values()]

    for k, v in stim_epochs_dict.items():
        print(f'{k}: {v.shape[0]}')

    package = ioutils.pload(os.path.join(processed_folder, 'package'))
    fs = package['fs']
    matches = package['matches']
    spike_trains_dict = {k: v / fs for k, v in package['spike_trains_dict_after_rp'].items()}

    if bool_heatmap:
        sf = os.path.join(fig_folder, 'psth_heatmaps')
        for celltype in ['ss', 'rest']:
            dat_per_stim_per_cell = []
            unit_ids = np.array(list(package['celltypes_parcellated_dict'][celltype].keys()))
            trains = [spike_trains_dict[unit_id] for unit_id in unit_ids]
            for k, event in stim_epochs_dict.items():
                dat_per_cell = []
                for train in trains:
                    x, ys, y_p, y_p_var = phy_utils.get_processed_ifr(
                        train, event, psthb, psthw,
                        bsl_subtract=False,
                        zscore=True,
                        zscoretype='across')
                    y_p = savgol_filter(y_p, window_length=filter_window, polyorder=0)
                    dat_per_cell.append(y_p)
                dat_per_stim_per_cell.append(np.array(dat_per_cell))
            dat_per_stim_per_cell = np.array(dat_per_stim_per_cell)
            print(dat_per_stim_per_cell.shape)

            ''' Sort by depth + plot'''
            depth_dict = {k: v for k, v in zip(cluster_ids, cluster_depth)}
            depths = np.array([depth_dict[k] for k in unit_ids])
            sort_ixs = np.argsort(depths)[::-1]  # from dorsal to ventral
            depths = depths[sort_ixs]
            unit_ids_sorted = unit_ids[sort_ixs]

            interval = intervals[celltype]
            yticks = np.arange(len(depths))[::interval]
            yticklabels = [f'{d}, {u}' for d, u in zip(depths[::interval].astype(int), unit_ids_sorted[::interval])]
            figutils.imagePanes(
                figsize=(8, 6),
                mats=dat_per_stim_per_cell[:, sort_ixs, :],
                titles=stim_names,
                vmin=-2.5,
                vmax=2.5,
                vcenter=0,
                axargs={'xticks': times_in_bins, 'xticklabels': xticklabels,
                        'yticks': yticks, 'yticklabels': yticklabels
                        }
            )
            plt.suptitle('Responses by depth')
            name = f'{celltype}_sort_by_depth'
            figutils.save_fig(sf, name, show=False)

            ''' Sort by responses + plot'''
            sort_times_in_bins = np.array(sort_time_window) / (psthb * 1e-3)
            sort_times_in_bins = sort_times_in_bins.astype(int)
            sort_ixs = sort_utils.sortByOnset(
                dat_per_stim_per_cell[:2],
                on=sort_times_in_bins[0],
                off=sort_times_in_bins[1],
                thres=zscore_sort_thres_pos,
                n_thres=zscore_sort_thres_neg
            )
            figutils.imagePanes(
                figsize=(8, 6),
                mats=dat_per_stim_per_cell[:, sort_ixs, :],
                titles=stim_names,
                vmin=-2.5,
                vmax=2.5,
                vcenter=0,
                axargs={'xticks': times_in_bins, 'xticklabels': xticklabels}
            )
            plt.suptitle('Responses by onset')
            name = f'{celltype}_sort_by_onset'
            figutils.save_fig(sf, name, show=False)

    if bool_psths:
        # specified celltypes
        celltypes = ['ss', 'cs', 'mli']
        psthbs = [psthb, psthb_cs, psthb]
        for b, celltype in zip(psthbs, celltypes):
            save_folder = os.path.join(fig_folder, 'psth', celltype)
            os.makedirs(save_folder, exist_ok=True)

            unit_ids = list(package['celltypes_parcellated_dict'][celltype].keys())
            unit_names = list(package['celltypes_parcellated_dict'][celltype].values())
            for unit_id, unit_name in zip(unit_ids, unit_names):
                train = [spike_trains_dict[unit_id]]
                psth_dict = {}
                for e, event in enumerate(stim_trial_times):
                    x, ys, y_p, y_p_var = phy_utils.get_processed_ifr(train, event, b, psthw)
                    psth_dict[0, e] = [x, y_p, y_p_var]

                psth_utils.make_psth(psth_dict,
                                     psthw,
                                     [unit_id],
                                     stim_names,
                                     xticks=xticks,
                                     xticklabels=xticklabels,
                                     ylabel=ylabel,
                                     xlabel=xlabel,
                                     colors=stim_colors)
                figutils.save_fig(save_folder, figname=f'{unit_name}__{unit_id}', close=True, show=False)

        # SS cliques
        psth_folder = os.path.join(fig_folder, 'psth', 'ss_cliques')
        os.makedirs(psth_folder, exist_ok=True)

        clique_dict = defaultdict(list)
        for unit_id, clique in package['celltypes_parcellated_dict']['_ss'].items():
            clique_dict[clique].append(unit_id)

        for clique_name, unit_ids in clique_dict.items():
            trains = [spike_trains_dict[x] for x in unit_ids]
            psth_dict = {}
            for t, train in enumerate(trains):
                for e, event in enumerate(stim_trial_times):
                    x, ys, y_p, y_p_var = phy_utils.get_processed_ifr(train, event, psthb, psthw)
                    psth_dict[t, e] = [x, y_p, y_p_var]

            psth_utils.make_psth(psth_dict,
                                 psthw,
                                 unit_ids,
                                 stim_names,
                                 xticks=xticks,
                                 xticklabels=xticklabels,
                                 ylabel=ylabel,
                                 xlabel=xlabel,
                                 colors=stim_colors)

            name = f'{clique_name}_{len(unit_ids)}'
            figutils.save_fig(psth_folder, figname=name, close=True, show=False)

        # matches
        psth_folder = os.path.join(fig_folder, 'psth', 'matches')
        os.makedirs(psth_folder, exist_ok=True)

        for match_ix in range(matches.shape[0]):
            unit_names = matches[match_ix]
            trains = [spike_trains_dict[x] for x in unit_names]

            psth_dict = {}
            for t, train in enumerate(trains):
                for e, event in enumerate(stim_trial_times):
                    b = psthb_cs if t == 0 else psthb
                    x, ys, y_p, y_p_var = phy_utils.get_processed_ifr(train, event, b, psthw)
                    psth_dict[t, e] = [x, y_p, y_p_var]

            psth_utils.make_psth(psth_dict,
                                 psthw,
                                 unit_names,
                                 stim_names,
                                 xticks=xticks,
                                 xticklabels=xticklabels,
                                 ylabel=ylabel,
                                 xlabel=xlabel,
                                 colors=stim_colors)

            name = f'match_{match_ix}__cs_{unit_names[0]}_ss_{unit_names[1]}'
            figutils.save_fig(psth_folder, figname=name, close=True, show=False)
