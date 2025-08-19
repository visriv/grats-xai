
# Simple TimeRISE-style randomized masking for time series (placeholder).
import torch, numpy as np

def random_mask_explainer(model, x, target=None, n_masks=128, p_keep=0.2, seed=0):
    """
    Black-box saliency via randomized keep masks on (D,T).
    Returns attribution A in shape (D,T) (normalized).
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    model.eval()
    x = x.to(device)
    B, D, T = x.shape
    base = model(x).detach()
    if target is None:
        target = base.argmax(dim=1).item()
    A = np.zeros((D,T), dtype=np.float32)
    with torch.no_grad():
        for _ in range(n_masks):
            M = (rng.random((D,T)) < p_keep).astype(np.float32)
            x_mask = x.clone()
            x_mask *= torch.from_numpy(M).to(device).unsqueeze(0)
            out = model(x_mask)
            score = out[0, target].item()
            A += M * score
    A /= (n_masks + 1e-8)
    A -= A.min()
    A /= (A.max() + 1e-8)
    return A
