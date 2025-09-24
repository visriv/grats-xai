import numpy as np
def _stack_mean_abs(mats):
    """mats: list of np.ndarray with same shape → mean of absolute values."""
    return np.mean(np.stack([np.abs(M) for M in mats], axis=0), axis=0)  # same shape as each M

def _aggregate_true_lags(A_lags_all, idxs):
    """
    A_lags_all: list (len=N) of [A^(1),...,A^(L)], each A^(ell) is (D,D)
    idxs: indices to aggregate over (validation subset you evaluated)
    returns: [A_true_global^(1),...,A_true_global^(L)], each (D,D)
    """
    # assume all samples have same L
    L = len(A_lags_all[0])
    agg = []
    for ell in range(L):
        mats = [A_lags_all[i][ell] for i in idxs]
        agg.append(_stack_mean_abs(mats))
    return agg
