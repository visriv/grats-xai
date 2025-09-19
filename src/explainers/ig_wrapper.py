
import torch
import numpy as np
from captum.attr import IntegratedGradients

def integrated_gradients(model, x, target=None, baseline='zero', steps=32):
    """
    x: (1, D, T) tensor
    returns attribution A: (D,T) numpy
    """
    device = next(model.parameters()).device
    model.eval()
    x = x.to(device)
    if baseline == 'zero':
        baseline_t = torch.zeros_like(x)
    else:
        baseline_t = torch.randn_like(x)*0.0
    ig = IntegratedGradients(model)
    target_idx = int(target) if target is not None else None
    # attr = ig.attribute(x, baselines=baseline_t, target=target_idx, n_steps=steps)


    with torch.backends.cudnn.flags(enabled=False):
        attr = ig.attribute(x, baselines=baseline_t, target=target_idx, n_steps=steps)
        
    return attr[0].detach().cpu().numpy()
