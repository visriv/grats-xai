import numpy as np

def laplacian_refine_closed_form(A, W, lam=0.3):
    """
    Graph-refined attribution using closed-form smoothing:
        S = (I + λL)^(-1) A

    Args:
        A : np.ndarray of shape (N, D, T)
            Base attribution maps for N samples.
        W : np.ndarray of shape (D, D)
            Graph adjacency matrix.
        lam : float
            Regularization strength.

    Returns:
        S : np.ndarray of shape (N, D, T)
            Refined attributions after Laplacian smoothing.
    """
    # Validate input dimensions
    if A.ndim == 2:
        A = A[None, ...]  # Promote to (1, D, T)
    N, D, T = A.shape

    # Compute Laplacian and closed-form smoother
    deg = np.diag(W.sum(axis=1))
    L = deg - W
    I = np.eye(D)
    M = np.linalg.inv(I + lam * L)  # (D, D)

    # Apply smoothing for each sample (efficient batch matmul)
    # M @ A[n] for each n → use einsum for vectorization
    S = np.einsum('ij,ndj->ndi', M, A)

    return S
