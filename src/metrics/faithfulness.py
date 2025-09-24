import numpy as np, torch

def mask_topk(x, A, k, mode="remove"):
    flat = A.flatten()
    idx = np.argsort(flat)[::-1][:k]
    mask = np.ones_like(flat)
    if mode == "remove": mask[idx] = 0
    elif mode == "retain": mask[:] = 0; mask[idx] = 1
    return x * torch.tensor(mask.reshape(A.shape), dtype=x.dtype)

def comp_suff_curves(model, x, target, A, ks, device="cpu"):
    model.eval()
    comp, suff = [], []
    with torch.no_grad():
        base = model(x.to(device))[0, target].item()
    for k in ks:
        x_masked = mask_topk(x, A, k, mode="remove").to(device)
        x_ret = mask_topk(x, A, k, mode="retain").to(device)
        with torch.no_grad():
            fx_masked = model(x_masked)[0, target].item()
            fx_ret = model(x_ret)[0, target].item()
        comp.append(base - fx_masked)
        suff.append(fx_ret - base)
    return {"comp": comp, "suff": suff}
