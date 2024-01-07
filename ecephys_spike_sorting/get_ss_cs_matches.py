import os
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm
import npyx.corr
import pandas as pd
from collections import defaultdict
from scipy.signal import savgol_filter
import logging

from utils import figutils
from utils import styleutils
from utils import ioutils
import configs

sns.reset_defaults()
styleutils.update_mpl_params(fontsize=7, linewidth=1)

mouse = configs.NPIX1
mouse_path = os.path.join(configs.BASE_PATH, mouse.path)
dates = list(mouse.exp_params.keys())

bool_plot_summary = True

firing_rate_cutoff = 0.01
amplitude_cutoff = 0.05
presence_ratio_cutoff = 0.7
adjacency_threshold = 200 #um, # peak channel of CS waveform must be within this threshold distance away from peak channel of SS waveform
ss_min_firing_rate_soft = 30
cs_max_firing_rate_soft = 4
cs_min_firing_rate_soft = 0.4
hard_inh_thresh = 0.3

ccg_win_size = 80 # ms
ccg_bin_size = 1 # ms
ccg_smooth_win = 3 # ms
ccg_t_start = 5 # ms, start of time interval to assess CS induced inhibition
ccg_t_end = 20 # ms, end of time interval to assess CS induced inhibition

# get directories
base_folders = []
phy_folders = []
for date in dates:
    try:
        catgt_folder = glob.glob(os.path.join(mouse_path, date, 'catgt*'))[0]
        catgt_imec_folder = glob.glob(os.path.join(catgt_folder, '*imec0'))[0]
        data_folder = os.path.join(catgt_imec_folder, 'imec0_ks2')
        base_folders.append(os.path.join(mouse_path, date))
        phy_folders.append(data_folder)
    except:
        pass

# override
base_folders = [r'D:\NPIX\NPIX2\npix2_2024.01.06']
phy_folders = [r'D:\NPIX\NPIX2\npix2_2024.01.06\catgt_s2bot_g0\s2bot_g0_imec0\imec0_ks2']

for base_folder, phy_folder in zip(base_folders, phy_folders):
    date_folder = os.path.dirname(os.path.dirname(os.path.dirname(phy_folder)))
    processed_folder = os.path.join(date_folder, 'PROCESSED')
    os.makedirs(processed_folder, exist_ok=True)

    log_f = os.path.join(processed_folder, 'log.txt')
    if os.path.exists(log_f):
        try:
            os.remove(log_f)
        except:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_f),
            logging.StreamHandler()
        ]
    )
    logging.info(phy_folder)

    metrics = pd.read_csv(os.path.join(phy_folder, 'metrics.csv'))
    metrics.head()
    if bool_plot_summary:
        # (KW, logx, logy)
        to_plot = [
            ('firing_rate', True, False),
            ('isi_viol', False, False),
            ('presence_ratio', False, False),
            ('amplitude_cutoff', False, False),
        ]
        f, axs = figutils.pretty_fig(figsize=(3 * len(to_plot), 2), rows=1, cols=len(to_plot))
        for i, tup in enumerate(to_plot):
            plt.sca(axs[i])
            metric = tup[0]
            logx = tup[1]
            logy = tup[2]

            data = metrics[metric]
            if logx:
                data = np.log10(data + 0.00001)
            plt.hist(data, log=logy)
            sns.despine()
            if logx:
                xlabel = 'log ' + metric
            else:
                xlabel = metric
            plt.xlabel(xlabel)
        plt.tight_layout()
        figutils.save_fig(processed_folder, 'summary_histograms', transparent=False, show=False)

    criteria = np.array([
        metrics['firing_rate'] > firing_rate_cutoff,
        # metrics['presence_ratio'] > 0.7,
        metrics['amplitude_cutoff'] < amplitude_cutoff,
    ])

    good_mask = np.all(criteria, axis=0)
    metrics['QC'] = good_mask
    metrics.to_csv(os.path.join(phy_folder, 'metrics.csv'), index=False)
    logging.info(f'N good cells: {good_mask.sum()}, N total cells: {len(good_mask)}')

    meta = npyx.read_metadata(phy_folder)
    recording_s = meta['recording_length_seconds']
    samp_rate = meta['highpass']['sampling_rate']
    chan_map = np.array(np.load(os.path.join(phy_folder, 'channel_map.npy'))).flatten()
    chan_pos = np.load(os.path.join(phy_folder, 'channel_positions.npy'))
    pos_x = chan_pos[:, 0]
    pos_y = chan_pos[:, 1]

    depths = []
    widths = []
    for chan in metrics['peak_channel']:
        ix = np.argwhere(chan == chan_map)[0][0]
        depths.append(pos_y[ix])
        widths.append(pos_x[ix])
    depths = np.array(depths)
    widths = np.array(widths)
    cluster_ids = metrics['cluster_id'].to_numpy()

    # calculate probe adjacency matrix
    adjacencies = {}
    for id, depth, width in zip(cluster_ids, depths, widths):
        within_reach_depth = np.abs(depths - depth) < adjacency_threshold
        within_reach_width = np.abs(widths - width) < adjacency_threshold
        within_reach = np.logical_and(within_reach_depth, within_reach_width)
        adjacencies[id] = cluster_ids[within_reach]

    # putative SS clusters
    criteria = np.array([
        metrics['firing_rate'].to_numpy() > ss_min_firing_rate_soft,
        metrics['QC']
    ])
    good_mask = np.all(criteria, axis=0)
    ss_cluster_ids = cluster_ids[good_mask]

    # putative CS clusters
    criteria = np.array([
        metrics['firing_rate'].to_numpy() < cs_max_firing_rate_soft,
        metrics['firing_rate'].to_numpy() > cs_min_firing_rate_soft,
        metrics['QC']
    ])
    good_mask = np.all(criteria, axis=0)
    cs_cluster_ids = cluster_ids[good_mask]
    logging.info(f'N SS clusters: {len(ss_cluster_ids)}, N CS clusters: {len(cs_cluster_ids)}')

    sum = 0
    cc_dict = {}
    for ss in ss_cluster_ids:
        valid_cs = np.intersect1d(adjacencies[ss], cs_cluster_ids)
        cc_dict[ss] = list(valid_cs)
        sum += len(valid_cs)
    logging.info(f'N pairs to compute: {sum}')

    pairs = defaultdict(list)
    for ss, css in tqdm.tqdm(cc_dict.items()):
        for cs in css:
            try:
                c = npyx.corr.ccg(phy_folder,
                                  [cs, ss],
                                  bin_size=ccg_bin_size,
                                  win_size=ccg_win_size,
                                  normalize='Hertz',
                                  sav=False,
                                  again=True,
                                  verbose=False)
                cc = c[0, 1]
                cc = savgol_filter(cc, window_length=ccg_smooth_win, polyorder=0)
                middle = int(1 + ccg_win_size / 2)
                metric = np.min(cc[middle + ccg_t_start:middle + ccg_t_end]) / np.mean(cc[middle - ccg_t_end:middle - ccg_t_start])
                if metric < hard_inh_thresh:
                    pairs[int(ss)].append([int(cs), float(round(metric, 2))])
            except:
                print(f'Cannot deal with {cs}')

    with open(os.path.join(processed_folder, 'matches.txt'), 'w') as f:
        for ss, css in pairs.items():
            css_f = [x for x in css if x[1] < hard_inh_thresh]
            if len(css_f):
                f.write(f'{ss}: ')
                for cs in css_f:
                    f.write(f'({cs[0]}, {cs[1]:0.2f}) ')
                f.write('\n')
                print(f'{ss}: {css_f}')

    ioutils.jsave(dict(pairs), os.path.join(processed_folder, 'matches'))

    # labeling
    match_dict = ioutils.jload(os.path.join(processed_folder, 'matches'))
    match_dict = {int(k): v for k, v in match_dict.items()}
    metrics = pd.read_csv(os.path.join(phy_folder, 'metrics.csv'))
    cluster_ids = metrics['cluster_id'].to_numpy()

    match_order_dict = {x: '' for x in cluster_ids}
    cb_dict = {x: '' for x in cluster_ids}
    ss_min_fr = 40

    criteria = np.array([
        metrics['firing_rate'].to_numpy() > ss_min_fr,
        metrics['QC']
    ])
    ss_cluster_ids = cluster_ids[np.all(criteria, axis=0)]
    for id in ss_cluster_ids:
        cb_dict[id] = '>40'

    ix = 1
    for ss, css in match_dict.items():
        if len(css):
            cb_dict[ss] = f'ss'
            match_order_dict[ss] = f'{ix}_ss'
            for cs in css:
                cb_dict[cs[0]] = 'cs'
                match_order_dict[cs[0]] = f'{ix}_cs'
            ix += 1

    celltype_fn = os.path.join(phy_folder, 'cluster_cb.tsv')
    if not os.path.exists(celltype_fn):
        df = pd.DataFrame({'cluster_id': list(cb_dict.keys()), 'cb': list(cb_dict.values())})
        df.to_csv(celltype_fn, index=False, sep='\t')
    else:
        print(f'Exists: {celltype_fn}')

    match_fn = os.path.join(phy_folder, 'cluster_match.tsv')
    if not os.path.exists(match_fn):
        df = pd.DataFrame({'cluster_id': list(match_order_dict.keys()),
                           'match': list(match_order_dict.values())})
        df.to_csv(match_fn, index=False, sep='\t')
    else:
        print(f'Exists: {match_fn}')

