
import numpy as np

def generate_var_dataset(n_series=512, T=100, D=8, n_classes=2, seed=0):
    """
    Nonlinear VAR with controlled lagged dependencies.
    Returns:
        X: (n_series, D, T)
        y: (n_series,) in {0..n_classes-1}
        true_W: dict lag -> (D,D) weighted adjacency used to generate data
    """
    rng = np.random.default_rng(seed)
    L = 2  # lags
    true_W = {ell: np.zeros((D, D), dtype=float) for ell in range(1, L+1)}
    # build a sparse directed graph with lags
    for ell in range(1, L+1):
        for j in range(D):
            parents = rng.choice(D, size=rng.integers(1, max(2, D//3)), replace=False)
            for i in parents:
                if i == j and ell == 1: 
                    continue
                true_W[ell][i, j] = rng.uniform(0.3, 0.9) * (1 if rng.random() < 0.7 else -1)

    X = np.zeros((n_series, D, T), dtype=np.float32)
    y = np.zeros((n_series,), dtype=np.int64)

    for n in range(n_series):
        noise = rng.normal(0, 0.5, size=(D, T+L))
        for t in range(L, T+L):
            x_t = noise[:, t].copy()
            for ell in range(1, L+1):
                x_t += true_W[ell].T @ np.tanh(noise[:, t-ell])
            # add nonlinear cross-feature multiplicative term to inject synergy
            cross = 0.1 * (noise[:, t-1] * noise[::-1, t-2 % (T)])
            x_t += cross
            noise[:, t] = x_t
        X[n] = noise[:, L:L+T]
        # label by a nonlinear rule on a small subgraph (unknown to learner)
        feat = X[n, :3, :].mean() + 0.7*X[n, 3, -10:].mean() - 0.5*np.abs(X[n, 2, 5:15]).mean()
        y[n] = int(feat > 0)

    return X, y, true_W
