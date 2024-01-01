
import os
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import phylib.io.model
import seaborn as sns

import configs
from utils import styleutils, ioutils

sns.reset_defaults()
styleutils.update_mpl_params(fontsize=14, linewidth=1)

mouse = configs.NPIX1
mouse_path = os.path.join(configs.BASE_PATH, mouse.path)
dates = list(mouse.exp_params.keys())
trial_duration = mouse.duration

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
        base_folders.append(os.path.join(mouse_path, date))
        phy_folders.append(data_folder)
    except:
        pass

print(catgt_folders)

catgt_folder = catgt_folders[0]
base_folder = base_folders[0]
phy_folder = phy_folders[0]
processed_folder = os.path.join(base_folder, 'processed')
fig_folder = os.path.join(base_folder, 'FIGURES')

print(phy_folder)

package = ioutils.pload(os.path.join(processed_folder, 'package'))
fs = package['fs']
matches = package['matches']
spike_trains_dict = {k: v for k, v in package['spike_trains_dict_after_rp'].items()}

cluster_id = matches[1, 0]

model = phylib.io.model.load_model(os.path.join(phy_folder, 'params.py'))
spikes_before = model.get_cluster_spikes(cluster_id)
spikes_after = spike_trains_dict[cluster_id]
