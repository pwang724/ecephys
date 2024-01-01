import numpy as np
from itertools import chain

def sortByOnset(data, on, off, thres, n_thres, return_categories=False):
    '''
    data = conditions X cells X frames

    '''
    top_ixs = []
    for d in data:
        boo = d[:, on:off] > thres
        onsets = np.array([np.where(x)[0][0] if np.any(x) else 1000 for x in boo])
        ixs = np.argsort(onsets)
        ixs = ixs[:np.sum(onsets < 1000)]
        [top_ixs.append(x) for x in ixs if x not in top_ixs]

    bottom_ixs = []
    for d in data:
        boo = d[:, on:off] < n_thres
        onsets = np.array([np.where(x)[0][0] if np.any(x) else 1000 for x in boo])
        ixs = np.argsort(onsets)
        ixs = ixs[:np.sum(onsets < 1000)]
        [bottom_ixs.append(x) for x in ixs if x not in bottom_ixs]

    top_ixs = [x for x in top_ixs if x not in bottom_ixs]
    middle_ixs = [x for x in range(data.shape[1]) if x not in (bottom_ixs + top_ixs)]
    if return_categories:
        return top_ixs, middle_ixs, bottom_ixs
    else:
        ixs = np.array(top_ixs + middle_ixs + bottom_ixs)
        return ixs

def sortBySelectivity(mats: np.ndarray, thres: float, positive: bool):
    list_of_ixs = []
    if positive:
        for mat in mats:
            ixs = np.argsort(mat)[::-1]
            cutoff = np.argmin(mat[ixs] > thres)
            list_of_ixs.append(ixs[:cutoff])
    else:
        for mat in mats:
            ixs = np.argsort(mat)
            cutoff = np.argmin(mat[ixs] < thres)
            list_of_ixs.append(ixs[:cutoff])
    top_ixs = list(dict.fromkeys(chain(*list_of_ixs)))
    return top_ixs


def sortByExclusion(mat: np.ndarray, thres: float):
    '''
    Only include indices if signal from all conditions is greater than set threshold

    :param mat:
    :param thres:
    :return:
    '''
    list_of_ixs = []
    for m in mat:
        ixs = np.argsort(m)
        cutoff = np.argmax(m[ixs] > thres)
        list_of_ixs.append(ixs[:cutoff])

    result = list_of_ixs[0]
    for l in list_of_ixs[1:]:
        result = [x for x in result if x in l]
    return result
