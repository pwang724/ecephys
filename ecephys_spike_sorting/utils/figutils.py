import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
import seaborn as sns
import matplotlib.colors
import utils.styleutils

# Figure Style settings
utils.styleutils.update_mpl_params()
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_PATH = os.path.join(ROOT_DIR, '_FIGURES')


def save_fig(save_path, figname='', dpi=300, pdf=False, show=True, close=True, transparent=False, ext='png',
             verbose=True):
    os.makedirs(save_path, exist_ok=True)
    if transparent:
        plt.savefig(os.path.join(save_path, figname + '.' + ext),
                    dpi=dpi,
                    transparent=transparent)
    else:
        plt.savefig(os.path.join(save_path, figname + '.' + ext),
                    dpi=dpi,
                    transparent=transparent,
                    facecolor='white')
    if pdf:
        plt.savefig(os.path.join(save_path, figname + '.pdf'), transparent=True)

    if verbose:
        print('Figure saved at: {}'.format(os.path.join(save_path, figname)))

    if show:
        plt.show()

    if close:
        plt.close()


def pretty_fig(
        figsize=(3, 2),
        rect=(0.15, 0.15, 0.6, 0.6),
        rows=1,
        cols=1,
        sharex=False,
        sharey=False,
        ax_args={}):
    if (rows > 1) ^ (cols > 1):
        f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharex=sharex, sharey=sharey, **ax_args)
        for a in ax:
            plt.sca(a)
            sns.despine()
    elif rows > 1 and cols > 1:
        f, ax = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, sharex=sharex, sharey=sharey, **ax_args)
        for a in ax:
            for b in a:
                plt.sca(b)
                sns.despine()
    else:
        f = plt.figure(figsize=figsize)
        ax = f.add_axes(rect)
    sns.despine()
    return f, ax

def despine(ax, normal=True, all=False):
    if normal:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    if all:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)



def addColorbar(fig, im=None, rect_cb=(0.77, 0.15, 0.02, 0.6)):
    ax = fig.add_axes(rect_cb)
    if im is None:
        cb = plt.colorbar(cax=ax)
    else:
        cb = plt.colorbar(im, cax=ax)
    return cb


def plotPanel(dffs,
              ax=None,
              width=20,
              height_per_cell=0.5,
              gap_between_plots=0.5,
              ymin=0.95,
              ymax=2,
              dff_baseline=0,
              xlines=[],
              axargs={},
              plotargs={},
              title='',
              ticks=True,
              show=True,
              save=False,
              saveFolder=None,
              saveName=None):
    '''
    :param dffs: matrix of dimensions [cells, frames]
    '''
    gap = (ymax - ymin) * gap_between_plots
    nCells = dffs.shape[0]
    y_increments = (ymax - ymin + gap) * np.arange(nCells) * -1
    plot_data = dffs + y_increments.reshape(-1, 1)

    if ax is None:
        plt.figure(figsize=(width, nCells * height_per_cell))
        ax = plt.gca()

    ax.plot(plot_data.T, color='k', **plotargs)
    if 'yticks' in axargs.keys():
        yticks = axargs['yticks']
        axargs['yticks'] = (y_increments[yticks]) + dff_baseline
        axargs['yticklabels'] = yticks
    ax.set_ylim([y_increments[-1] + ymin - height_per_cell,
                 ymax + height_per_cell])
    ax.set_xlim([0, dffs.shape[1]])
    ax.set_title(title)

    ax.set(**axargs)
    if ticks:
        ax.tick_params(axis="y", length=4, width=1, direction="in")
        ax.tick_params(axis="x", length=4, width=1, direction="in")
    else:
        ax.tick_params(axis="y", length=0, width=0, direction="in")
        ax.tick_params(axis="x", length=0, width=0, direction="in")

    for xline in xlines:
        ax.plot([xline, xline], [*ax.get_ylim()],
                linewidth=1,
                linestyle='dotted',
                alpha=1,
                color='gray')

    if save:
        if saveFolder is None:
            saveFolder = os.path.join(FIGURE_PATH, 'panel')
        save_fig(saveFolder, saveName, show=True, close=True)

    if show:
        plt.show()
    return ax, y_increments + dff_baseline


def plot_regions(ax, y_increments, regions, region_y_gap=0.5, color='k'):
    for y, rs in zip(y_increments, regions):
        yn = y - region_y_gap
        if rs:
            for r in rs:
                ax.plot([r.start, r.end], [yn, yn], color,
                        linewidth=2,
                        alpha=1)
        # if rs:
        #     regions = base.Regions(rs, length=rs[-1].end)
        #     vec = regions.to_vector()
        #     vec[vec < 0] = np.nan
        #     vec[vec > 0] = yn
        #     ax.plot(yn, 'k', linewidth=2, alpha=0.5)


def imagePanel(dffs,
               ax=None,
               width=20,
               height_per_cell=0.5,
               vmin=-0.2,
               vcenter=0.,
               vmax=0.7,
               xlines=[],
               axargs={},
               show=True,
               save=False,
               saveFolder=None,
               saveName=None):
    '''
    :param dffs: matrix of dimensions [cells, frames]
    '''
    nCells = dffs.shape[0]

    if ax is None:
        plt.figure(figsize=(width, nCells * height_per_cell))
        ax = plt.gca()

    plt.set_cmap('bwr')
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    plt.imshow(dffs, interpolation='none', norm=norm, origin='upper')
    plt.axis('auto')

    plt.clim(vmin, vmax)
    plt.xlim([0, dffs.shape[1]])
    #
    ax.set(**axargs)
    ax.tick_params(axis="y", length=4, width=2, direction="out")
    ax.tick_params(axis="x", length=4, width=2, direction="out")

    for xline in xlines:
        plt.plot([xline, xline], [*plt.ylim()],
                 linewidth=0.75,
                 linestyle='dashed',
                 alpha=0.5,
                 color='black')
    if save:
        if saveFolder is None:
            saveFolder = os.path.join(FIGURE_PATH, 'panel')
        save_fig(saveFolder, saveName, show=True)

    if show:
        plt.show()


def plotPanes(axs,
              mats,
              titles,
              axargs,
              ymin,
              ymax,
              figsize=[10, 15],
              show=True,
              save=False,
              saveFolder=None,
              saveName=None):
    if axs is None:
        f, axs = pretty_fig(figsize,
                            rect=(0.1, 0.1, 0.7, 0.7),
                            rows=1,
                            cols=len(titles))
    axargs = dict(axargs)
    if 'xticks' in axargs.keys():
        xlines = axargs['xticks']
    else:
        xlines = []

    yincs = []
    for i in range(len(titles)):
        plt.sca(axs[i])
        if i == 0:
            axargs.update({'yticks': np.arange(0, mats[0].shape[0], 10)})
        else:
            axargs.update({'yticks': []})

        _, yinc = plotPanel(mats[i],
                            width=3,
                            gap_between_plots=0.5,
                            height_per_cell=.25,
                            ymin=ymin,
                            ymax=ymax,
                            xlines=xlines,
                            plotargs={'linewidth': 1},
                            axargs=axargs,
                            ax=axs[i],
                            show=False,
                            save=False)
        sns.despine(top=True, right=True, bottom=True, left=True)
        plt.title(titles[i])
        yincs.append(yinc)
    if save:
        if saveFolder is None:
            saveFolder = os.path.join(FIGURE_PATH, 'panel')
        save_fig(saveFolder, saveName, show=False)
    return axs, yincs


def imagePanes(mats,
               titles,
               axargs,
               figsize = [5, 3],
               vmin=-0.2,
               vcenter=0.5,
               vmax=0.7,
               colorbar=True,
               cbar_title='',
               show=True,
               save=False,
               saveFolder=None,
               saveName=None):
    f, axs = pretty_fig(figsize,
                        rect=(0.1, 0.1, 0.7, 0.7),
                        rows=1,
                        cols=len(titles))
    axargs = dict(axargs)
    if 'xticks' in axargs.keys():
        xlines = axargs['xticks']
    else:
        xlines = []

    for i in range(len(titles)):
        if len(titles) == 1:
            ax = axs
        else:
            ax = axs[i]
        plt.sca(ax)
        if i == 0:
            pass
            # axargs.update({'yticks': np.arange(0, mats[0].shape[0], 10)})
        else:
            axargs.update({'yticks': [], 'yticklabels': []})

        imagePanel(mats[i],
                   width=3,
                   height_per_cell=.25,
                   vmin=vmin,
                   vcenter=vcenter,
                   vmax=vmax,
                   xlines=xlines,
                   axargs=axargs,
                   ax=ax,
                   show=False,
                   save=False)
        sns.despine(top=True, right=True, bottom=True, left=True)
        plt.title(titles[i])

    if colorbar:
        ax = plt.gca()
        cax = f.add_axes([ax.get_position().x1 + 0.02,
                          ax.get_position().y0,
                          0.02,
                          ax.get_position().height])
        cb = plt.colorbar(cax=cax)
        cb.outline.set_linewidth(0)
        cb.set_label(cbar_title, rotation=270)
        plt.sca(axs[0])
    #
    # if save:
    #     save_fig(saveFolder, saveName, show=False, close=False)
    #     save_fig(saveFolder, saveName, show=False, pdf=True)
    #
    # if show:
    #     plt.show()

def label_plot(mask, y1, y2, ax=None, label=None):
    if ax is None:
        plt.figure(figsize=(15, 1))
        ax = plt.gca()
        plt.yticks([]);
    plt.sca(ax)
    plt.fill_between(np.arange(len(mask)), y1=y1, y2=y2, where=mask>0, alpha=0.5)
    sns.despine(left=True)
    locs, labels = plt.yticks()
    plt.yticks([]);
    if label is not None:
        locs = list(locs)
        labels = list(labels)
        locs.append((y2+y1)/2)
        labels.append(label)
        plt.yticks(locs, labels)
    return ax
