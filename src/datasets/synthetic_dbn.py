import os
import argparse
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split


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
    if n_edges > 0:
        # choose either negative or positive class consistently
        label = rng.integers(0, 2)  # 0=negative class, 1=positive class
        if label == 0:
            weights = rng.uniform(-2.0, -0.5, size=n_edges)
        else:
            weights = rng.uniform(0.5, 2.0, size=n_edges)
        W[mask] = weights
    else:
        label = 0
    return W, label


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
    return A_list


# ---------- SEM Simulation ----------
def simulate_sem(n, d, W, A_list, noise="normal", seed=0):
    rng = np.random.default_rng(seed)
    p = len(A_list)
    X = np.zeros((n, d))

    for t in range(p, n):
        xt = np.zeros(d)
        xt += X[t] @ W
        for lag, A in enumerate(A_list, start=1):
            xt += X[t - lag] @ A
        if noise == "normal":
            eps = rng.normal(0, 1, size=d)
        elif noise == "exp":
            eps = rng.exponential(1, size=d)
        else:
            raise ValueError("noise must be 'normal' or 'exp'")
        X[t] = xt + eps
    return X


# ---------- Full dataset ----------
def generate_dataset(n=500, d=5, p=1, k_intra=2, k_inter=1,
                     model_intra="ER", model_inter="ER",
                     noise="normal", eta=1.0, seed=0):

    W, label = generate_intra_slice(d, k=k_intra, model=model_intra, seed=seed)
    A_list = generate_inter_slice(d, k=k_inter, p=p,
                                  model=model_inter, eta=eta, seed=seed + 1)
    X = simulate_sem(n, d, W, A_list, noise=noise, seed=seed + 2)
    return X, W, A_list, label


# ---------- Main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    noise = cfg.get("noise", "normal")
    eta = cfg.get("eta", 1.0)
    seed = cfg.get("seed", 0)
    model_intra = cfg.get("model_intra", "ER")
    model_inter = cfg.get("model_inter", "ER")

    outdir = "runs"
    os.makedirs(outdir, exist_ok=True)

    for exp_id, exp in enumerate(cfg["experiments"]):
        n = exp["n"]; d = exp["d"]; p = exp["p"]
        k_intra = exp["k_intra"]; k_inter = exp["k_inter"]

        # Generate dataset
        X, W, A_list, label = generate_dataset(
            n=n, d=d, p=p, k_intra=k_intra, k_inter=k_inter,
            model_intra=model_intra, model_inter=model_inter,
            noise=noise, eta=eta, seed=seed + exp_id * 10
        )

        # Create labels for each sample
        y = np.full(shape=(X.shape[0],), fill_value=label, dtype=int)

        # Split into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        # Save as PKL
        dataset_name = f"dbn_n{n}_d{d}_p{p}_{model_intra}{k_intra}_{model_inter}{k_inter}"
        exp_dir = os.path.join(outdir, dataset_name)
        os.makedirs(exp_dir, exist_ok=True)

        with open(os.path.join(exp_dir, "train.pkl"), "wb") as f:
            pickle.dump({"X": X_train, "y": y_train, "W": W, "A_list": A_list}, f)

        with open(os.path.join(exp_dir, "val.pkl"), "wb") as f:
            pickle.dump({"X": X_val, "y": y_val, "W": W, "A_list": A_list}, f)

        # Print class distribution
        print(f"[✓] {dataset_name}")
        print("  Train class distribution:", np.bincount(y_train))
        print("  Val   class distribution:", np.bincount(y_val))
