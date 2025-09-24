
import torch
import numpy as np
from captum.attr import IntegratedGradients

def integrated_gradients(model, x, target=None, baseline='zero', steps=32):
    device = next(model.parameters()).device
    model.eval()
    x = x.to(device)

    if baseline == 'zero':
        baseline_t = torch.zeros_like(x)
    else:
        baseline_t = torch.randn_like(x) * 0.0

    ig = IntegratedGradients(model)

    # If target is a tensor (batch), loop over samples
    if isinstance(target, torch.Tensor) and target.ndim == 1:
        attrs = []
        for i in range(x.size(0)):
            with torch.backends.cudnn.flags(enabled=False):
                a = ig.attribute(
                    x[i:i+1], baselines=baseline_t[i:i+1],
                    target=int(target[i].item()), n_steps=steps
                )
            attrs.append(a.detach().cpu().numpy()[0])
        return np.stack(attrs, axis=0)

    # Else keep old behavior
    target_idx = int(target) if target is not None else None
    with torch.backends.cudnn.flags(enabled=False):
        attr = ig.attribute(x, baselines=baseline_t, target=target_idx, n_steps=steps)
    return attr.detach().cpu().numpy()
