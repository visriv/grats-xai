import numpy as np, torch

def asymmetric_interaction_response(model, x, A, target, lags, rho=1.0, S=8, device="cpu"):
    """
    Compute asymmetric perturbation response between nodes (p->q).
    A: base attribution (D,T)
    Returns: W_hat (D,D,L) interaction strengths per lag
    """
    model.eval()
    D, T = A.shape
    L = max(lags)+1
    W_hat = np.zeros((D,D,L))

    for d1 in range(D):
        for d2 in range(D):
            if d1 == d2: continue
            for ell in lags:
                vals = []
                for _ in range(S):
                    mask = torch.ones_like(x)
                    mask[:, :, d1] = 0  # perturb feature d1
                    x_pert = x * mask
                    with torch.no_grad():
                        out = model(x_pert.to(device))[0, target].item()
                    vals.append(out)
                delta = A[d2].mean() - np.mean(vals)
                W_hat[d1,d2,ell] = delta * rho
    return W_hat
