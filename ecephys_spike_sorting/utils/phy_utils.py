import pandas as pd
import numpy as np
import scipy.stats as stats

def load_phy_tsv_as_dict(fn):
    '''
    Load a phy tsv file into a dict. Assumes that
    - index column is 'cluster_id'
    - there is only a single data column
    - data separated by tabs
    '''
    celltypes = pd.read_csv(fn, delimiter='\t', na_filter=False)
    celltypes = celltypes.to_dict('record')
    celltypes_dict = {}
    k = list(celltypes[0].keys())[1]
    for d in celltypes:
        cluster_id = d['cluster_id']
        datum = d[k]
        if datum is not '':
            celltypes_dict[cluster_id] = datum
    return celltypes_dict

def duplicates_mask(t, enforced_rp=0, fs=30000):
    '''
    Get rid of duplicates within a time window. Policy is to keep the first spike of potential duplicates.
    - t: in samples,sampled at fs Hz
    - enforced_rp: in ms
    '''
    duplicate_m = np.append([False], np.diff(t)<=enforced_rp*fs/1000)
    return duplicate_m


def get_processed_ifr(times,
                      events,
                      b,
                      window,
                      remove_empty_trials=True,
                      zscore=False,
                      zscoretype='within',
                      convolve=False,
                      gsd=1,
                      method='gaussian',
                      bsl_subtract=False,
                      bsl_window=[-4000, 0],
                      process_y=False):
    '''
    Returns the "processed" (averaged and/or smoothed and/or z-scored) instantaneous firing rate of a neuron.

    Arguments:
        - times:  list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b:      float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
        - convolve:      boolean, set to True to convolve the aligned binned train with a half-gaussian window to smooth the ifr
        - gsd:           float, gaussian window standard deviation in ms
        - method:        convolution window shape: gaussian, gaussian_causal, gamma | Default: gaussian
        - bsl_subtract: whether to baseline substract the trace. Baseline is taken as the average of the baseline window bsl_window
        - bsl_window:    [t1,t2], window on which the baseline is computed, in ms -> used for zscore and for baseline subtraction (i.e. zscoring without dividing by standard deviation)
        - process_y:     whether to also process the raw trials x bins matrix y (returned raw by default)

    Returns:
        - x:       (n_bins,) array tiling bins, in milliseconds
        - y:       (n_trials, n_bins,) array, the unprocessed ifr (by default - can be processed if process_y is set to True)
        - y_p:     (n_bins,) array, the processed instantaneous firing rate (averaged and/or smoothed and/or z-scored)
        - y_p_var: (n_bins,) array, the standard deviation across trials of the processed instantaneous firing rate
    '''

    # Window and bins translation
    x, ys = align_times(times, events, b, window, remove_empty_trials)
    ys = ys / (b * 1e-3)
    assert x.shape[0] == ys.shape[1]
    assert not np.any(np.isnan(ys.ravel())), 'WARNING nans found in aligned ifr!!'

    _, y_bsl = align_times(times, events, b, bsl_window, remove_empty_trials)
    y_bsl = y_bsl / (b * 1e-3)
    assert not np.any(np.isnan(y_bsl.ravel())), 'WARNING nans found in aligned ifr!!'

    ys, y_p, y_p_var = process_2d_trials_array(
        ys, y_bsl, zscore, zscoretype, convolve, gsd, method, bsl_subtract, process_y)

    return x, ys, y_p, y_p_var


def align_times(times,
                events,
                b,
                window,
                remove_empty_trials=True):
    '''
    Arguments:
        - times: list/array in seconds, timestamps to align around events. Concatenate several units for population rate!
        - events: list/array in seconds, events to align timestamps to
        - b: float, binarized train bin in millisecond
        - window: [w1, w2], where w1 and w2 are in milliseconds.
        - remove_empty_trials: boolean, remove from the output trials where there were no timestamps around event. | Default: True
    Returns:
        - aligned_t: dictionnaries where each key is an event in absolute time and value the times aligned to this event within window.
        - aligned_tb: a len(events) x window/b matrix where the spikes have been aligned, in counts.
    '''
    assert np.any(events), 'You provided an empty array of events!'
    t = np.sort(times)
    tbins = np.arange(window[0], window[1] + b, b)
    aligned_tb = np.zeros((len(events), len(tbins) - 1)).astype(float)
    for i, e in enumerate(events):
        ts = t - e  # ts: t shifted
        mask = (ts >= window[0] / 1000) & (ts <= window[1] / 1000)  # only take spikes within specified window
        tsc = ts[mask]  # tsc: ts clipped
        if np.any(tsc) and remove_empty_trials:
            tscb = np.histogram(tsc * 1000, bins=tbins)[0]  # tscb: tsc binned
            aligned_tb[i, :] = tscb
        else:
            aligned_tb[i, :] = np.nan
    aligned_tb = aligned_tb[~np.isnan(aligned_tb).any(axis=1)]

    # edge condition: no spike on triggered on the specified window on any event
    if not np.any(aligned_tb):
        aligned_tb = np.zeros((len(events), len(tbins) - 1))
    return tbins[:-1] / 1000, aligned_tb


def process_2d_trials_array(ys, y_bsl, zscore=False, zscoretype='within',
                            convolve=False, gsd=1, method='gaussian',
                            bsl_subtract=False,
                            process_y=False):
    # zscore or not
    assert zscoretype in ['within', 'across', 'pw']
    if zscore or bsl_subtract:  # use baseline of ifr far from stimulus
        y_mn = np.mean(np.mean(y_bsl, axis=0))
        if zscore:
            assert not bsl_subtract, 'WARNING, cannot zscore AND baseline subtract - pick either!'
            if zscoretype == 'within':
                y_mn = np.mean(np.mean(y_bsl, axis=0))
                y_sd = np.std(np.mean(y_bsl, axis=0))
                if y_sd == 0 or np.isnan(y_sd): y_sd = 1
                y_p = (np.mean(ys, axis=0) - y_mn) / y_sd
                y_p_var = stats.sem((ys - y_mn) / y_sd, axis=0)  # variability across trials in zscore values??
                if process_y: ys = (ys - y_mn) / y_sd
            elif zscoretype == 'across':
                y_mn = np.mean(y_bsl.flatten())
                y_sd = np.std(y_bsl.flatten())
                if y_sd == 0 or np.isnan(y_sd): y_sd = 1
                y_p = (np.mean(ys, axis=0) - y_mn) / y_sd
                y_p_var = stats.sem((ys - y_mn) / y_sd, axis=0)  # variability across trials in zscore values??
                if process_y: ys = (ys - y_mn) / y_sd
            elif zscoretype == 'pw':
                '''
                - First baseline subtract mean for each individual trial
                - Calculate STD across all baseline trials
                - Use this mean and STD to calculate Z-scores
                '''
                y_mn_per_trial = np.mean(y_bsl, axis=1, keepdims=True)
                y_sd = np.std((y_bsl - y_mn_per_trial).flatten())
                y_p = np.mean(ys - y_mn_per_trial, axis=0) / y_sd
                y_p_var = stats.sem((ys - y_mn_per_trial)/y_sd, axis=0)

        elif bsl_subtract:
            y_p = np.mean(ys, axis=0) - y_mn
            y_p_var = stats.sem(ys, axis=0)
            if process_y: ys = ys - y_mn

    else:
        y_p = np.mean(ys, axis=0)
        y_p_var = stats.sem(ys, axis=0)  # sd across trials

    assert not np.any(np.isnan(y_p)), 'WARNING nans found in trials array!'
    # Convolve or not
    if convolve:
        y_p = smooth(y_p, method=method, sd=gsd)
        y_p_var = smooth(y_p_var, method=method, sd=gsd)
        if process_y: ys = smooth(ys, method=method, sd=gsd)

    if np.any(np.isnan(y_p_var)):
        y_p_var = np.ones(y_p.shape)
        print(
            'WARNING not enough spikes around events to compute std, y_p_var was filled with nan. Patched by filling with ones.')
    return ys, y_p, y_p_var


def smooth(arr, method='gaussian_causal', sd=5, axis=1, gamma_a=5):
    '''
    Smoothes a 1D array or a 2D array along specified axis.
    Arguments:
        - arr: ndarray/list, array to smooth
        - method: string, see methods implemented below | Default 'gaussian'
        - sd: int, gaussian window sd (in unit of array samples - e.g. use 10 for a 1ms std if bin size is 0.1ms) | Default 5
        - axis: int axis along which smoothing is performed.
        - a_gamma: sqrt of Gamma function rate (essentially std) | Default 5

    methods implemented:
        - gaussian
        - gaussian_causal
        - gamma (is causal)
    '''
    assert arr.ndim <= 2, \
        "WARNING this function runs on 3D arrays but seems to shift data leftwards - not functional yet."
    if arr.ndim == 1: axis = 0

    ## Checks and formatting
    assert method in ['gaussian', 'gaussian_causal', 'gamma']
    assert type(sd) in [int, np.int32, np.int64]

    ## pad array at beginning and end to prevent edge artefacts
    C = arr.shape[axis] // 2
    pad_width = [[C, C] if i == axis else [0, 0] for i in range(arr.ndim)]
    padarr = np.pad(arr, pad_width, 'symmetric')

    ## Compute the kernel
    if method in ['gaussian', 'gaussian_causal']:
        X = np.arange(-4 * sd, 4 * sd + 1)
        kernel = stats.norm.pdf(X, 0, sd)
        if method == 'gaussian_causal':
            kernel[:len(kernel) // 2] = 0
    elif method == 'gamma':
        # a = shape, b = scale = 1/rate. std=sqrt(a)/b = sqrt(a) for b=1
        X = np.arange(gamma_a ** 2 // 2, max((gamma_a ** 2) * 3 // 2 + 1, 10))
        kernel = stats.gamma.pdf(X, gamma_a ** 2)

    # center the maximum to prevent data shift in time
    # This is achieved by padding the left/right of the kernel with zeros.
    mx = np.argmax(kernel)
    if mx < len(kernel) / 2:
        kernel = np.append(np.zeros(len(kernel) - 2 * mx), kernel)
    elif mx > len(kernel) / 2:
        kernel = np.append(kernel, np.zeros(mx - (len(kernel) - mx)))
    assert len(kernel) < padarr.shape[axis], \
        'The kernel is longer than the array to convolved, you must decrease sd.'

    # normalize kernel to prevent vertical scaling
    kernel = kernel / sum(kernel)

    ## Convolve array with kernel
    sarr = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=axis, arr=padarr)

    ## Remove padding
    sarr = sarr[slice_along_axis(C + 1, -C + 1, axis=axis)]
    assert np.all(sarr.shape == arr.shape)

    return sarr


def slice_along_axis(a, b, s=1, axis=0):
    """
    Returns properly formatted slice to slice array/list along specified axis.
    - a: start
    - b: end
    - s: step
    """
    slc = slice(a, b, s)
    return (slice(None),) * axis + (slc,)





# make my own histogram
# bins = bounds * 2
# bins_per_ms = bins / ccg_win_size
# y, x = np.histogram(ccg_time, bins=bins, range=[-bounds, bounds])
# x = 1000 * x / fs # convert to ms
# y = savgol_filter(y, int(0.5*fs/1000), polyorder=0) # filter
# y = y * bins_per_ms * 1000 / cs_train.size # convert to hz
# plt.figure()
# plt.plot(x[:-1], y)
# plt.xlabel('Time (ms)')
# plt.ylabel('Firing rate (Hz)')

