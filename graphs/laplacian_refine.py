
import numpy as np

def laplacian_refine(A, W, lam=0.3, iters=200):
    """
    Graph-Laplacian smoothing of attribution A (D,T) using adjacency W (D,D).
    Uses simple Jacobi iterations for clarity.
    """
    D,T = A.shape
    Wsym = 0.5*(W + W.T)
    Dg = np.diag(Wsym.sum(axis=1))
    L = Dg - Wsym
    S = A.copy()
    alpha = lam / (1.0 + lam)
    for _ in range(iters):
        S = (1-alpha)*A + alpha*(S - L @ S)
    return S
