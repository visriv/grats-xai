
# Simple TimeRISE-style randomized masking for time series (placeholder).
import torch, numpy as np
def random_mask_explainer(model, x, target=None, n_masks=128, p_keep=0.2, seed=0):
    """
    Black-box saliency via randomized keep masks on (D,T).
    Args:
        x: (B, D, T) tensor
        target: (B,) tensor of class indices, or None (will use model argmax)
    Returns:
        A: (B, D, T) numpy array of normalized attributions
    """
    rng = np.random.default_rng(seed)
    device = next(model.parameters()).device
    model.eval()
    x = x.to(device)
    B, D, T = x.shape

    with torch.no_grad():
        base = model(x).detach()  # (B, C)
        if target is None:
            target = base.argmax(dim=1)  # (B,)
        if not torch.is_tensor(target):
            target = torch.tensor(target, device=device)
        else:
            target = target.to(device)

        A = np.zeros((B, D, T), dtype=np.float32)

        for _ in range(n_masks):
            M = (rng.random((D, T)) < p_keep).astype(np.float32)  # (D, T)
            M_torch = torch.from_numpy(M).to(device).unsqueeze(0).expand(B, -1, -1)  # (B, D, T)

            x_mask = x * M_torch
            out = model(x_mask)  # (B, C)
            scores = out.gather(1, target.view(-1, 1)).squeeze(1).cpu().numpy()  # (B,)

            # add contribution for each sample
            for i in range(B):
                A[i] += M * scores[i]

        # normalize each sample separately
        for i in range(B):
            Ai = A[i]
            Ai -= Ai.min()
            if Ai.max() > 0:
                Ai /= Ai.max()
            A[i] = Ai

    return A  # (B, D, T)
