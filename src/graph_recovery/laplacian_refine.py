import numpy as np

def laplacian_refine_closed_form(A, W, lam=0.3):
    """
    Graph-refined attribution using closed-form:
    S = (I + λL)^(-1) A
    A: base attribution (D,T)
    W: adjacency (D,D)
    """
    D = W.shape[0]
    deg = np.diag(W.sum(axis=1))
    L = deg - W
    I = np.eye(D)
    M = np.linalg.inv(I + lam*L)
    return M @ A
