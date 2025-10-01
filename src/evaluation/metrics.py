import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def infidelity_comprehensiveness(A, S, k_frac=0.05):
    D,T = A.shape
    k = max(1, int(k_frac*D*T))
    idxA = np.argsort(A.ravel())[::-1][:k]
    idxS = np.argsort(S.ravel())[::-1][:k]
    jacc = len(set(idxA).intersection(set(idxS))) / k
    return {"topk_overlap": jacc}



# Function to compute the AUROC drop after occlusion
def evaluate_auroc_drop(model, X_eval, y_eval, explainer_fn, expl_kwargs, auroc_ks=[5, 10, 20, 30, 50]):
    """
    Evaluate AUROC drop after occluding top-K salient points based on the explainer.
    
    Args:
        model: The trained model.
        X_eval: (N_eval, D, T) - Evaluation data.
        y_eval: (N_eval,) - True labels.
        explainer_fn: The explainer function (e.g., IG, TimeRISE).
        expl_kwargs: Additional kwargs for the explainer (e.g., n_masks, p_keep).
        auroc_ks: List of top-K salient points to occlude.
    
    Returns:
        auroc_drops: List of AUROC drops for each value in auroc_ks.
    """
    # Step 1: Get saliency map (attribution map)
    attr_batch = explainer_fn(model, X_eval, target=y_eval, **expl_kwargs)  # (N_eval, D, T)
    
    # Step 2: Initialize lists to store results
    auroc_drops = []
    
    # Step 3: Iterate over different values of K (top-K salient points to occlude)
    for K in auroc_ks:
        # Step 3.1: Create occluded version of the input by zeroing out the top-K salient points
        X_occluded = X_eval.clone()  # (N_eval, D, T)
        for i in range(X_eval.shape[0]):  # iterate over batch
            saliency = attr_batch[i]  # (D, T) for each sample
            
            # Flatten the saliency map and get indices of top K salient points
            flat_saliency = saliency.flatten()
            top_k_indices = np.argsort(flat_saliency)[-K:]  # Indices of the top K salient points
            
            # Convert to 2D indices
            top_k_coords = np.unravel_index(top_k_indices, saliency.shape)
            
            # Occlude (set to 0) the top K salient points
            X_occluded[i, top_k_coords[0], top_k_coords[1]] = 0
        
        # Step 3.2: Evaluate AUROC for the occluded data
        with torch.no_grad():
            logits = model(X_occluded)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Class probabilities for class 1

        # Step 3.3: Calculate AUROC score for original and occluded
        original_probs = torch.softmax(model(X_eval), dim=1)[:, 1].cpu().numpy()
        
        # Calculate AUROC for both original and occluded
        auroc_original = roc_auc_score(y_eval, original_probs)
        auroc_occluded = roc_auc_score(y_eval, probs)
        
        # Calculate AUROC drop (normalized)
        auroc_drop = auroc_original - auroc_occluded
        auroc_drops.append(auroc_drop)

    return auroc_drops