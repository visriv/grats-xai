import os
import argparse
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle, sys
from sklearn.model_selection import train_test_split
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.utils.io_utils import set_seed, ensure_dir, save_pickle, save_json
from typing import Dict, List, Tuple
import sys



# ---------- Intra-slice DAG ----------
def generate_intra_slice(d, k=2, model="ER", seed=0):
    rng = np.random.default_rng(seed)
    if model == "ER":
        p = k / (d - 1)
        G = nx.gnp_random_graph(d, p, directed=True, seed=seed)
    elif model == "BA":
        G = nx.barabasi_albert_graph(d, max(1, k//2), seed=seed)
        G = nx.DiGraph(G)
    else:
        raise ValueError("model must be 'ER' or 'BA'")

    perm = rng.permutation(d)
    P = np.eye(d)[perm]
    A = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
    A = P @ A @ P.T
    A = np.tril(A, k=-1)

    W = np.zeros_like(A)
    mask = A > 0
    n_edges = mask.sum()
    
    # Note: this is not sequence/class label : just for different kinds of weights
    if n_edges > 0:
      
        weight_label = rng.integers(0, 2)  
        if weight_label == 0:
            weights = rng.uniform(-2.0, -0.5, size=n_edges)
        else:
            weights = rng.uniform(0.5, 2.0, size=n_edges)
        W[mask] = weights
    else:
        weight_label = 0
    return W # W is d x d


# ---------- Inter-slice DAGs ----------
def generate_inter_slice(d, k=2, p=1, model="ER", eta=1, seed=0):
    rng = np.random.default_rng(seed)
    A_list = []
    alpha = 1.0 / (d ** (p - 1))

    for lag in range(1, p + 1):
        if model == "ER":
            prob = k / d
            A_bin = rng.binomial(1, prob, size=(d, d))
        else:
            raise NotImplementedError("Only ER inter-slice model implemented")

        A = np.zeros_like(A_bin, dtype=float)
        mask = A_bin > 0
        n_edges = mask.sum()
        if n_edges > 0:
            scale = alpha / (eta ** (lag - 1))
            w_neg = rng.uniform(-0.5 * scale, -0.3 * scale, size=n_edges // 2)
            w_pos = rng.uniform(0.3 * scale, 0.5 * scale, size=n_edges - len(w_neg))
            weights = np.concatenate([w_neg, w_pos])
            rng.shuffle(weights)
            A[mask] = weights
        A_list.append(A)
    return A_list # p x d x d


# ---------- SEM Simulation (with optional change-point) ----------
def simulate_sem(n, d, W, A_list, noise="normal", seed=0, change_point=None, alt_noise=None):
    """
    Simulate SEM with optional change-point in noise distribution.
    
    Args:
      n : sequence length
      d : number of variables
      W : intra-slice adjacency (d x d)
      A_list : list of inter-slice adjacencies (p x d x d)
      noise : str, initial noise type ("normal" or "exp")
      seed : int
      change_point : int or None, index at which to switch noise regime
      alt_noise : str or None, alternative noise type after change_point
    """
    rng = np.random.default_rng(seed)
    p = len(A_list)
    X = np.zeros((n, d))

    for t in range(p, n):
        xt = np.zeros(d)
        xt += X[t] @ W
        for lag, A in enumerate(A_list, start=1):
            xt += X[t - lag] @ A

        # choose noise regime
        if change_point is not None and t >= change_point and alt_noise is not None:
            noise_type = alt_noise
        else:
            noise_type = noise

        if noise_type == "normal":
            eps = rng.normal(0, 1, size=d)
        elif noise_type == "exp":
            eps = rng.exponential(1, size=d)
        else:
            raise ValueError("noise must be 'normal' or 'exp'")

        X[t] = xt + eps
    return X


# ---------- Full dataset (N sequences) ----------
def generate_dataset(num_samples=200, T=500, d=5, p=1,
                     k_intra=2, k_inter=1,
                     model_intra="ER", model_inter="ER",
                     noise="normal", eta=1.0, seed=0,
                     use_change_point=True):
    """
    Generate a dataset of multiple independent time series samples.
    Each sample is simulated from its own random DBN.
    
    If use_change_point=True, half of the sequences will undergo a noise regime
    switch at T/2 (e.g., normal -> exponential). Labels reflect presence (1) or absence (0) of change.
    
    Returns:
      X: (num_samples, T, d)
      y: (num_samples,)
      W: intra-slice adjacency
      A_all: list of inter-slice adjacencies  (p x d x d)
    """

    # For simplicity, generate a single intra/inter DAG structure shared across samples
    W = generate_intra_slice(d, k=k_intra, model=model_intra, seed=seed)
    A_list = generate_inter_slice(d, k=k_inter, p=p,
                                  model=model_inter, eta=eta, seed=seed + 1)

    X_all, y_all = [], []
    for n in range(num_samples):
        rng = np.random.default_rng(seed + 2 + n)
        
        if use_change_point and n % 2 == 1:  # assign ~half samples to have change-points
            cp = T // 2
            X = simulate_sem(T, d, W, A_list, noise=noise,
                             seed=seed + 2 + n,
                             change_point=cp,
                             alt_noise="exp" if noise == "normal" else "normal")
            label = 1
        else:
            X = simulate_sem(T, d, W, A_list, noise=noise,
                             seed=seed + 2 + n)
            label = 0

        X_all.append(X)
        y_all.append(label)

    X_all = np.stack(X_all, axis=0)  # (N,T,D)
    y = np.array(y_all, dtype=int)

    return X_all, y, W, A_list





def dataset_name(exp):
    return f"dbn_n{exp['n']}_d{exp['d']}_p{exp['p']}_{exp.get('model_intra','ER')}{exp['k_intra']}_{exp.get('model_inter','ER')}{exp['k_inter']}"

def dataset_dir(root, exp):
    return os.path.join(root, dataset_name(exp))

