
import numpy as np

def infidelity_comprehensiveness(A, S, k_frac=0.05):
    D,T = A.shape
    k = max(1, int(k_frac*D*T))
    idxA = np.argsort(A.ravel())[::-1][:k]
    idxS = np.argsort(S.ravel())[::-1][:k]
    jacc = len(set(idxA).intersection(set(idxS))) / k
    return {"topk_overlap": jacc}
