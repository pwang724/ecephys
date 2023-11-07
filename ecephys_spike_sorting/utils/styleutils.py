import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List, Text, Union
import pandas as pd


def update_mpl_params(fontsize=9,
                      linewidth=2,
                      wspace=0.01,
                      hspace=0.01):
    # Figure Style settings
    mpl.rcParams.update({
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'figure.subplot.wspace': wspace,
        'figure.subplot.hspace': hspace,
    })

    mpl.rcParams.update({
        'font.size': fontsize,
        'axes.linewidth': linewidth,
        'font.sans-serif': 'Arial',
        'font.family': 'sans-serif',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })


niceNames = {
    'ODOR_ON': 'On',
    'ODOR_OFF': 'Off',
    'WATER_ON': 'H2O',
    'SOCIAL_ON': 'Start of Social Bout',
    'CAROUSEL_ROTATION_START': 'Rot. On',
    'CAROUSEL_ROTATION_END': 'Rot. Off',
    'SOCIAL_UNIQUE_BOOL': 'Interacted'
}

def getNiceNames(keys: Union[str, List[str]]):
    def helper(key):
        for word in niceNames.keys():
            if word in key:
                return key.replace(word, niceNames[word])
        return key

    if keys == None:
        return keys
    elif isinstance(keys, str):
        return helper(keys)
    else:
        return [helper(x) for x in keys]

def untruncatePandas():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)