
import numpy as np
import torch

def _conditional_impute(x, mask_keep, rng):
    """Very simple conditional imputer: mean-fill per feature over kept time points.
    Replace masked-out entries by the feature mean within the kept set; if none kept, zero.
    x: (D,T) numpy, mask_keep: (D,T) in {0,1}
    """
    D,T = x.shape
    x_imp = x.copy()
    for d in range(D):
        kept = mask_keep[d].astype(bool)
        mu = x[d, kept].mean() if kept.any() else 0.0
        x_imp[d, ~kept] = mu
    return x_imp

def shapley_interaction_score(model, x, target, P, Q, S=16, seed=0):
    """
    Estimate 2-player Shapley interaction for nodes P=(d1,t1), Q=(d2,t2).
    x: (1,D,T) torch tensor
    Returns scalar psi.
    """
    device = next(model.parameters()).device
    model.eval()
    x_np = x[0].detach().cpu().numpy()
    D,T = x_np.shape
    rng = np.random.default_rng(seed)
    def eval_mask(M):
        x_imp = _conditional_impute(x_np, M, rng)
        with torch.no_grad():
            xt = torch.from_numpy(x_imp).to(device).float().unsqueeze(0)
            out = model(xt)[0, target].item()
        return out

    psi = 0.0
    for _ in range(S):
        M = (rng.random((D,T)) < 0.1).astype(np.float32)  # random context
        Mp = M.copy(); Mq = M.copy(); Mpq = M.copy()
        d1,t1 = P; d2,t2 = Q
        Mp[d1,t1]=1; Mq[d2,t2]=1; Mpq[d1,t1]=1; Mpq[d2,t2]=1
        val = eval_mask(Mpq)-eval_mask(Mp)-eval_mask(Mq)+eval_mask(M)
        psi += val
    return psi / S

def build_graph_from_topk(model, x, target, A, topk=20, max_edges=64, S=16, seed=0, lags=(0,1,2)):
    """
    x: (1,D,T) tensor; A: (D,T) numpy base attribution
    Returns adjacency W (D,D) aggregated over allowed lags (directional, unsigned magnitude).
    """
    D,T = A.shape
    flat_idx = np.argsort(A.ravel())[::-1][:topk]
    candidates = [np.unravel_index(i, (D,T)) for i in flat_idx]
    W = np.zeros((D,D), dtype=float)
    rng = np.random.default_rng(seed)
    scores = []
    for i in range(len(candidates)):
        for j in range(len(candidates)):
            if i==j: continue
            (d1,t1) = candidates[i]; (d2,t2) = candidates[j]
            if (t2 - t1) not in lags: 
                continue
            psi = shapley_interaction_score(model, x, target, (d1,t1), (d2,t2), S=S, seed=rng.integers(1<<31))
            scores.append((abs(psi), d1, d2))
    scores.sort(reverse=True)
    for k,(s,d1,d2) in enumerate(scores[:max_edges]):
        W[d1,d2] += s
    if W.max()>0: W /= W.max()
    return W
