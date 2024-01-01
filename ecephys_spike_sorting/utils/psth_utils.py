import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from ast import literal_eval as ale

from utils import figutils

# Make matplotlib saved figures text text editable
mpl.rcParams["svg.fonttype"] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

default_mplp_params = dict(
    # title default parameters
    title_w='regular',
    title_s=22,

    # axes labels default parameters
    axlab_w='regular',
    axlab_s=22,

    # tick labels default parameters
    ticklab_w='regular',
    ticklab_s=22,
    ticks_direction='out',

    # ticks default parameters
    xtickrot=0,
    ytickrot=0,
    xtickha='center',
    xtickva='top',
    ytickha='right',
    ytickva='center',

    # spines and layout default parameters
    lw=1,
    hide_top_right=True,
    hide_axis=False,
    tight_layout=False,

    # legend default parameters
    show_legend=False,
    hide_legend=False,
    legend_loc=(1,1),

    # figure saving default parameters
    saveFig=False,
    saveDir = "~/Downloads",
    figname="figure",
    _format="pdf",

    # colorbar default parameters
    colorbar=False,
    cbar_w=0.03,
    cbar_h=0.4,
    clabel=None,
    clabel_w='regular',
    clabel_s=22,
    cticks_s=22,

    # horizontal and vertical lines default parameters
    hlines = None, # provide any iterable of values to plot horizontal lines along the y axis
    vlines = None, # provide any iterable of values to plot vertical lines along the x axis
    lines_kwargs = {'lw':1.5, 'ls':'--', 'color':'k', 'zorder':-1000}, # add any other matplotlib.lines.Line2D arguments
)

def make_psth(psth_dict, psthw, unit_names, stim_names, xticks, xticklabels, xlabel, ylabel, colors, **psth_kwargs):
    f, axs = figutils.pretty_fig(figsize=(12, 2.5 * len(unit_names)), rows=len(unit_names), cols=len(stim_names),
                                 sharex=True, sharey='row',
                                 )
    for t, _ in enumerate(unit_names):
        for e, _ in enumerate(stim_names):
            if len(unit_names) == 1:
                ax = axs[e]
            else:
                ax = axs[t, e]
            plt.sca(ax)
            x, y_p, y_p_var = psth_dict[t, e]

            title = stim_names[e] if t == 0 else ''
            yl = f'Cell: {unit_names[t]}\n{ylabel}' if e ==0 else ''
            xl = xlabel if t==len(unit_names)-1 else ''

            psth_plt(x,
                     y_p,
                     y_p_var,
                     psthw / 1000,
                     events_toplot=xticks, events_color='k', xticks=xticks, xticklabels=xticklabels,
                     title=title,
                     color=colors[e],
                     xlabel=xl,
                     ylabel=yl,
                     ax=ax,
                     **psth_kwargs
                     )


def psth_plt(
    x, y_p, y_p_var, psthw, events_toplot=[0], events_color='r',
    title='', color='darkgreen',
    zscore=False, bsl_subtract=False, ylim=None,
    convolve=False, xticks=None, xticklabels=None,
    xlabel='Time (ms)', ylabel='IFR (spk/s)', legend_label=None, legend=False,
    ax=None, figsize=None, tight_layout=True, hspace=None, wspace=None,
    prettify=True, **mplp_kwargs):
    """
    Plots peri-event PSTHs
    Arguments:
        - x: time vector
        - y_p: mean PSTH
        - y_p_var: std of PSTH
        - psthw: peri-stimulus time window
        - events_toplot: list of event indices to plot
        - events_color: color of event lines
        - title: plot title
        - color: color of PSTH
        - zscore: bool, PSTH was zscored or not
        - bsl_subtract: bool, whether to subtract baseline from PSTH or not
        - ylim: y-axis limits
        - convolve: bool, whether PSTH was convolved or not
        - xticks: x-axis ticks
        - xticklabels: x-axis tick labels
        - xlabel: x-axis label
        - ylabel: y-axis label
        - legend_label: label for legend
        - legend: bool, whether to plot legend or not
        - ax: axis to plot on
        - figsize: figure size
        - tight_layout: bool, whether to apply tight_layout() or not
        - hspace: horizontal space between subplots
        - wspace: vertical space between subplots
        - prettify: bool, whether to apply mplp() prettification or not
        - **mplp_kwargs: any additional formatting parameters, passed to mplp()
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig=ax.get_figure()

    areasteps=None if convolve else 'post'

    if zscore or bsl_subtract:
        ax.fill_between(x, y_p-y_p_var, y_p+y_p_var,
                        color=color, alpha=0.8, step=areasteps, label=legend_label)
    else:
        ax.fill_between(x, y_p-y_p_var, y_p+y_p_var, color=color, alpha=0.5, step=areasteps)
        ax.fill_between(x, y_p*0, y_p,
                        color=color, alpha=1, step=areasteps, label=legend_label)
    if legend: ax.legend()
    if convolve:
        if zscore or bsl_subtract: ax.plot(x, y_p-y_p_var, color='black', lw=0.5)
        ax.plot(x, y_p+y_p_var, color='black', lw=0.5)
        ax.plot(x, y_p, color='black', lw=1.5)
    else:
        if zscore or bsl_subtract: ax.step(x, y_p-y_p_var, color='black', lw=0.5, where='post')
        ax.step(x, y_p+y_p_var, color='black', lw=0.5, where='post')
        ax.step(x, y_p, color='black', lw=1.5,where='post')

    yl=ax.get_ylim() if ylim is None else ylim
    if not (zscore or bsl_subtract): yl=[0,yl[1]]
    for etp in events_toplot:
        ax.plot([etp,etp], yl, ls='--', lw=1, c=events_color)
        ax.set_ylim(yl)

    xl=psthw
    if bsl_subtract or zscore:
        ax.plot(xl,[0,0],lw=1,ls='--',c='black',zorder=-1)
        if zscore:
            if yl[0]<-2: ax.plot(xl,[-2,-2],lw=1,ls='--',c='black',zorder=-1)
            if yl[1]>2: ax.plot(xl,[2,2],lw=1,ls='--',c='black',zorder=-1)
    ax.set_xlim(xl)

    if ylabel is None:
        ylabel='IFR\n(zscore)' if zscore else r'$\Delta$ FR (spk/s)' if bsl_subtract else 'IFR (spk/s)'
    if xlabel is None: xlabel=''

    fig,ax=mplp(fig=fig, ax=ax, figsize=figsize,
                xlim=psthw, ylim=yl, xlabel=xlabel, ylabel=ylabel,
                xticks=xticks, xtickslabels=xticklabels,
                axlab_w='bold', axlab_s=12,
                ticklab_w='regular',ticklab_s=12, lw=1,
                title=title, title_w='bold', title_s=12,
                hide_top_right=True, tight_layout=tight_layout, hspace=hspace, wspace=wspace,
                prettify=prettify, **mplp_kwargs)
    return fig


def mplp(fig=None, ax=None, figsize=None, axsize=None,
         xlim=None, ylim=None, xlabel=None, ylabel=None,
         xticks=None, yticks=None, xtickslabels=None, ytickslabels=None,
         reset_xticks=None, reset_yticks=None,
         xtickrot=None, ytickrot=None,
         xtickha=None, xtickva=None, ytickha=None, ytickva=None,
         axlab_w=None, axlab_s=None,
         ticklab_w=None, ticklab_s=None, ticks_direction=None,
         title=None, title_w=None, title_s=None,
         lw=None, hide_top_right=None, hide_axis=None,
         tight_layout=None, hspace=None, wspace=None,
         show_legend=None, hide_legend=None, legend_loc=None,
         saveFig=None, saveDir = None, figname=None, _format="pdf",
         colorbar=None, vmin=None, vmax=None, cmap=None, cticks=None,
         cbar_w=None, cbar_h=None, clabel=None, clabel_w=None, clabel_s=None, cticks_s=None,
         hlines = None, vlines = None, lines_kwargs = None,
         prettify=True):
    """
    make plots pretty
    matplotlib plotter

    Awesome utility to format matplotlib plots.
    Simply add mplp() at the end of any plotting script, feeding it with your fav parameters!

    IMPORTANT If you set prettify = False, it will only reset the parameters that you provide actively, and leave the rest as is.

    In a breeze,
        - change the weight/size/alignment/rotation of the axis labels, ticks labels, title
        - edit the x, y and colorbar axis ticks and ticks labels
        - hide the splines (edges of your figure)
        - hide all the axis, label etc in one go with hide_axis
        - save figures in any format
        - add or remove a legend
        - add a custom colorbar
        - apply tight_layout to fit your subplots properly (in a way which prevents saved plots from being clipped)

    How it works: it will grab the currently active figure and axis (plt.gcf() and plt.gca()).
    Alternatively, you can pass a matplotlib figure and specific axes as arguments.

    Default Arguments:
        {0}
    """

    global default_mplp_params

    if fig is None:
        if ax is None:
            fig = plt.gcf()
        else:
            fig = ax.get_figure()
    if ax is None: ax=plt.gca()

    # if prettify is set to True (default),
    # mplp() will change the plot parameters in the background,
    # even if not actively passed
    if prettify:

        # limits default parameters
        if xlim is None: xlim = ax.get_xlim()
        if ylim is None: ylim = ax.get_ylim()

        # title default parameters
        if title is None: title = ax.get_title()
        if title_w is None: title_w = default_mplp_params['title_w']
        if title_s is None: title_s = default_mplp_params['title_s']

        # axes labels default parameters
        if ylabel is None: ylabel = ax.get_ylabel()
        if xlabel is None: xlabel = ax.get_xlabel()
        if axlab_w is None: axlab_w = default_mplp_params['axlab_w']
        if axlab_s is None: axlab_s = default_mplp_params['axlab_s']

        # tick labels default parameters
        if ticklab_w is None: ticklab_w = default_mplp_params['ticklab_w']
        if ticklab_s is None: ticklab_s = default_mplp_params['ticklab_s']
        if ticks_direction is None: ticks_direction = default_mplp_params['ticks_direction']

        # ticks default parameters
        if xtickrot is None: xtickrot = default_mplp_params['xtickrot']
        if ytickrot is None: ytickrot = default_mplp_params['ytickrot']
        if xtickha is None: xtickha = default_mplp_params['xtickha']
        if xtickva is None: xtickva = default_mplp_params['xtickva']
        if ytickha is None: ytickha = default_mplp_params['ytickha']
        if ytickva is None: ytickva = default_mplp_params['ytickva']

        # spines and layout default parameters
        if lw is None: lw = default_mplp_params['lw']
        if hide_top_right is None: hide_top_right = default_mplp_params['hide_top_right']
        if hide_axis is None: hide_axis = default_mplp_params['hide_axis']
        if tight_layout is None: tight_layout = default_mplp_params['tight_layout']

        # legend default parameters
        if show_legend is None: show_legend = default_mplp_params['show_legend']
        if hide_legend is None: hide_legend = default_mplp_params['hide_legend']
        if legend_loc is None: legend_loc = default_mplp_params['legend_loc']

        # figure saving default parameters
        if saveFig is None: saveFig = default_mplp_params['saveFig']
        if saveDir is None: saveDir = default_mplp_params['saveDir']
        if figname is None: figname = default_mplp_params['figname']
        if _format is None: _format = default_mplp_params['_format']

        # colorbar default parameters
        if colorbar is None: colorbar = default_mplp_params['colorbar']
        if cbar_w is None: cbar_w = default_mplp_params['cbar_w']
        if cbar_h is None: cbar_h = default_mplp_params['cbar_h']
        if clabel is None: clabel = default_mplp_params['clabel']
        if clabel_w is None: clabel_w = default_mplp_params['clabel_w']
        if clabel_s is None: clabel_s = default_mplp_params['clabel_s']
        if cticks_s is None: cticks_s = default_mplp_params['cticks_s']


    hfont = {'fontname':'Arial'}
    if figsize is not None:
        assert axsize is  None, \
            "You cannot set both the axes and figure size - the axes size is based on the figure size."
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
    if axsize is not None:
        assert figsize is  None, \
            "You cannot set both the axes and figure size - the axes size is based on the figure size."
        set_ax_size(ax, *axsize)

    # Opportunity to easily hide everything
    if hide_axis is not None:
        if hide_axis:
            ax.axis('off')
        else: ax.axis('on')

    # Axis labels
    if ylabel is not None: ax.set_ylabel(ylabel, weight=axlab_w, size=axlab_s, **hfont)
    if xlabel is not None: ax.set_xlabel(xlabel, weight=axlab_w, size=axlab_s, **hfont)

    # Setup x/y limits BEFORE altering the ticks
    # since the limits will alter the ticks
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # Tick values
    if prettify and xticks is None:
        if reset_xticks:
            ax.xaxis.set_major_locator(AutoLocator())
        xticks = ax.get_xticks()
    if xticks is not None: ax.set_xticks(xticks)

    if prettify and yticks is None:
        if reset_yticks:
            ax.yaxis.set_major_locator(AutoLocator())
        yticks = ax.get_yticks()
    if yticks is not None: ax.set_yticks(yticks)

    # Tick labels
    fig.canvas.draw() # To force setting of ticklabels
    if xtickslabels is None and prettify:
        if any(ax.get_xticklabels()):
            if isnumeric(ax.get_xticklabels()[0].get_text()):
                xtickslabels, x_nflt = get_labels_from_ticks(xticks)
            else:
                xtickslabels = ax.get_xticklabels()
    if ytickslabels is None and prettify:
        if any(ax.get_yticklabels()):
            if isnumeric(ax.get_yticklabels()[0].get_text()):
                ytickslabels, y_nflt = get_labels_from_ticks(yticks)
            else:
                ytickslabels = ax.get_yticklabels()

    if xtickslabels is not None:
        if xticks is not None:
            assert len(xtickslabels)==len(xticks), \
                'WARNING you provided too many/few xtickslabels! Make sure that the default/provided xticks match them.'
        if xtickha is None: xtickha = ax.xaxis.get_ticklabels()[0].get_ha()
        if xtickva is None: xtickva = ax.xaxis.get_ticklabels()[0].get_va()
        ax.set_xticklabels(xtickslabels, fontsize=ticklab_s, fontweight=ticklab_w,
                           color=(0,0,0), **hfont, rotation=xtickrot, ha=xtickha, va=xtickva)
    if ytickslabels is not None:
        if yticks is not None:
            assert len(ytickslabels)==len(yticks), \
                'WARNING you provided too many/few ytickslabels! Make sure that the default/provided yticks match them.'
        if ytickha is None: ytickha = ax.yaxis.get_ticklabels()[0].get_ha()
        if ytickva is None: ytickva = ax.yaxis.get_ticklabels()[0].get_va()
        ax.set_yticklabels(ytickslabels, fontsize=ticklab_s, fontweight=ticklab_w,
                           color=(0,0,0), **hfont, rotation=ytickrot, ha=ytickha, va=ytickva)

    # Reset x/y limits a second time
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # Title
    if title is not None: ax.set_title(title, size=title_s, weight=title_w)

    # Ticks and spines aspect
    if prettify:
        ax.tick_params(axis='both', bottom=1, left=1, top=0, right=0, width=lw, length=4, direction=ticks_direction)
    elif lw is not None or ticks_direction is not None:
        ax.tick_params(axis='both', width=lw, direction=ticks_direction)

    if hide_top_right is not None:
        spine_keys = list(ax.spines.keys())
        hide_spine_keys = ['polar'] if 'polar' in spine_keys else ['top', 'right']
        lw_spine_keys = ['polar'] if 'polar' in spine_keys else ['left', 'bottom', 'top', 'right']
        if hide_top_right and 'top' in hide_spine_keys: [ax.spines[sp].set_visible(False) for sp in hide_spine_keys]
        else: [ax.spines[sp].set_visible(True) for sp in hide_spine_keys]
        for sp in lw_spine_keys:
            ax.spines[sp].set_lw(lw)

    # Optionally plot horizontal and vertical dashed lines
    if lines_kwargs is None: lines_kwargs = {}
    l_kwargs = default_mplp_params['lines_kwargs']
    l_kwargs.update(lines_kwargs) # prevalence of passed arguments

    if hlines is not None:
        assert hasattr(hlines, '__iter__'), 'hlines must be an iterable!'
        for hline in hlines:
            ax.axhline(y=hline, **l_kwargs)
    if vlines is not None:
        assert hasattr(vlines, '__iter__'), 'vlines must be an iterable!'
        for vline in vlines:
            ax.axvline(x=vline, **l_kwargs)

    # Aligning and spacing axes and labels
    if tight_layout: fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if hspace is not None: fig.subplots_adjust(hspace=hspace)
    if wspace is not None: fig.subplots_adjust(wspace=wspace)
    if prettify:
        axis_to_align = [AX for AX in fig.axes if 'AxesSubplot' in AX.__repr__()]
        fig.align_ylabels(axis_to_align)
        fig.align_xlabels(axis_to_align)

    assert not (show_legend and hide_legend), \
        "You instructed to both show and hide the legend...?"
    if legend_loc is not None:
        assert len(legend_loc)==2 or len(legend_loc)==4, \
            "legend_loc must comply to the bbox_to_anchor format ( (x,y) or (x,y,width,height))."
    if show_legend: plt.legend(bbox_to_anchor=legend_loc, prop={'family':'Arial'})
    elif hide_legend: plt.legend([],[], frameon=False)

    if prettify:
        fig.patch.set_facecolor('white')
    return fig, ax

def get_labels_from_ticks(ticks):
    ticks=npa(ticks)
    nflt=0
    for t in ticks:
        t=round(t,4)
        for roundi in range(4):
            if t == round(t, roundi):
                if nflt<(roundi):nflt=roundi
                break
    ticks_labels=ticks.astype(np.int64) if nflt==0 else np.round(ticks.astype(float), nflt)
    jump_n=1 if nflt==0 else 2
    ticks_labels=[str(l)+'0'*(nflt+jump_n-len(str(l).replace('-',''))) for l in ticks_labels]
    return ticks_labels, nflt

def npa(arr=[], **kwargs):
    '''Returns np.array of some kind.
    Optional params:
        - zeros: tuple. If provided, returns np.zeros(zeros)
        - ones: tuple. If provided, returns np.ones(ones)
        - empty: tuple. If provided, returns np.empty(empty)
        - dtype: numpy datatype. If provided, returns np.array(arr, dtype=dtype) .'''

    dtype=kwargs['dtype'] if 'dtype' in kwargs.keys() else None
    if 'zeros' in kwargs.keys():
        arr = np.zeros(kwargs['zeros'], dtype=dtype)
    elif 'ones' in kwargs.keys():
        arr = np.ones(kwargs['ones'], dtype=dtype)
    elif 'empty' in kwargs.keys():
        arr = np.empty(kwargs['empty'], dtype=dtype)
    else:
        arr=np.array(arr, dtype=dtype)
    return arr

def isnumeric(x):
    x=str(x).replace('−','-')
    try:
        ale(x)
        return True
    except:
        return False

def set_ax_size(ax,w,h):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)
