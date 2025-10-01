import numpy as np
import torch

def asymmetric_interaction_response(model, x, A, target, max_lag, rho=1.0, S=8, device="cpu"):
    """
    Estimate lagged directed interactions W_hat[p,q,ell].
    
    Args:
        model  : nn.Module, trained classifier
        x      : torch.Tensor (B, D, T)
        A      : np.ndarray (B, D, T), base attribution
        target : np.ndarray (B,), class indices per sample
        max_lag   : int, max lag value to test - hence lag \in [0, 1, 2, ..., max_lag]
        rho    : scaling factor
        S      : number of perturbation samples
    Returns:
        W_hat : np.ndarray (D, D, max(lags)+1), aggregated over B

    
    """
    model.eval()
    B, D, T = x.shape
    L = max_lag + 1
    W_hat = np.zeros((D, D, L))
    x = x.to(device)


    from tqdm import tqdm

    # Assuming L, D, and S are defined
    # Nested loops with tqdm
    for p in tqdm(range(D), desc="Processing p", leave=True):  # Outer loop for p
        for q in tqdm(range(D), desc="Processing q", leave=False):  # Nested loop for q
            if p == q:
                continue
            for ell in range(L):
                vals = []
                for s in tqdm(range(S), desc="Processing s", leave=False):  # Innermost loop for s

                    # mask for all batches, same shape as x
                    mask = torch.ones_like(x)
                    mask[:, p, :T-ell] = 0.0  # perturb p at valid time indices
                    
                    x_pert = x * mask
                    with torch.no_grad():
                        out = model(x_pert.to(device))  # (B, C)
                        out_sel = out[torch.arange(B), target].cpu().numpy()

                    # attribution difference aggregated across time & batch
                    A_q = A[:, q, ell:T]  # shape (B, T-ell)
                    delta = A_q - out_sel[:, None]  # broadcast subtraction
                    vals.append(delta)

                vals = np.stack(vals, axis=0)  # (S, B, T-ell)
                W_hat[p, q, ell] = rho * np.mean(vals)

    return W_hat
