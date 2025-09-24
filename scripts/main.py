import argparse, os, yaml, numpy as np, torch, torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx

from datasets.synthetic_var import generate_var_dataset
from models.lstm import LSTMClassifier
from explainers.ig_wrapper import integrated_gradients
from explainers.time_rise import random_mask_explainer
from graphs.interaction_shapley import build_graph_from_topk
from graphs.laplacian_refine import laplacian_refine
from evaluation.metrics import infidelity_comprehensiveness
from sklearn.model_selection import train_test_split
from tqdm import trange


# ------------------------- utils -------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def save_txt(path, text):
    with open(path, "w") as f:
        f.write(text)

def maybe_flatten_ts_attr(A):
    """
    Try to coerce attributions to shape [T, D] for nicer plotting.
    Expected input x is [N=1, T, D], so A often matches.
    Returns (A2D, title_suffix)
    """
    A = to_numpy(A)
    title = ""
    if A.ndim == 3:
        # [N, T, D] -> [T, D]
        if A.shape[0] == 1:
            A = A[0]
            title = "[T × D]"
        else:
            # average over batch
            A = A.mean(axis=0)
            title = "avg over N → [T × D]"
    elif A.ndim == 2:
        title = "[T × D]"
    else:
        title = f"shape {A.shape}"
    return A, title

def plot_heatmap(A, out_path, title="Attributions", xlabel="Time (t)", ylabel="Feature (d)"):
    A2D, suffix = maybe_flatten_ts_attr(A)
    plt.figure(figsize=(8, 4))
    im = plt.imshow(A2D.T, aspect="auto", origin="lower")
    plt.title(f"{title} {suffix}")
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_side_by_side(A, S, out_path, titles=("Base A", "Refined S")):
    A2D, _ = maybe_flatten_ts_attr(A)
    S2D, _ = maybe_flatten_ts_attr(S)
    vmax = np.percentile(np.abs(np.concatenate([A2D.flatten(), S2D.flatten()])), 99)
    vmax = max(vmax, 1e-8)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(A2D.T, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax)
    plt.title(titles[0]); plt.xlabel("Time (t)"); plt.ylabel("Feature (d)")
    plt.colorbar(im1, fraction=0.046, pad=0.04)

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(S2D.T, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax)
    plt.title(titles[1]); plt.xlabel("Time (t)"); plt.ylabel("Feature (d)")
    plt.colorbar(im2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_topk_bars(A, S, k, out_path):
    A2D, _ = maybe_flatten_ts_attr(A)
    S2D, _ = maybe_flatten_ts_attr(S)
    a = np.abs(A2D).flatten()
    s = np.abs(S2D).flatten()

    idx_a = np.argsort(a)[::-1][:k]
    idx_s = np.argsort(s)[::-1][:k]
    # Build a union to show overlap visually
    idx_union = list(dict.fromkeys(list(idx_a) + list(idx_s)))[:k]

    vals_a = a[idx_union]
    vals_s = s[idx_union]

    x = np.arange(len(idx_union))
    w = 0.4
    plt.figure(figsize=(10, 4))
    plt.bar(x - w/2, vals_a, width=w, label="Base A")
    plt.bar(x + w/2, vals_s, width=w, label="Refined S")
    plt.xticks(x, [str(i) for i in idx_union], rotation=45)
    plt.ylabel("|attribution|")
    plt.title(f"Top-{k} comparison on union indices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def try_draw_graph(W, out_path, max_nodes=150, threshold=None):
    """
    Try to visualize adjacency/edge list matrix W.
    - If W is square (N×N): treat as weighted adjacency.
    - Else, try to coerce via top edges.
    """
    W_np = to_numpy(W)
    plt.figure(figsize=(6, 6))
    try:
        if W_np.ndim == 2 and W_np.shape[0] == W_np.shape[1]:
            N = W_np.shape[0]
            if threshold is None:
                # choose a percentile to avoid hairballs
                threshold = np.percentile(np.abs(W_np), 95)
            # Build graph
            G = nx.Graph()
            for i in range(N):
                G.add_node(i)
            # add edges above threshold
            rows, cols = np.where(np.abs(W_np) >= threshold)
            edges = list(zip(rows.tolist(), cols.tolist()))
            # undirected simple graph – drop self loops & dupes
            edges = [(i, j) for (i, j) in edges if i < j]
            if len(edges) > 2000:
                edges = edges[:2000]
            G.add_edges_from(edges)
            # maybe sample nodes for readability
            if G.number_of_nodes() > max_nodes:
                nodes_kept = set(sorted(list(G.nodes()))[:max_nodes])
                G = G.subgraph(nodes_kept).copy()
            pos = nx.spring_layout(G, seed=0, k=None)
            nx.draw(G, pos, node_size=20, width=0.5)
            plt.title(f"Graph W (|E|={G.number_of_edges()}, |V|={G.number_of_nodes()})")
        else:
            # Fallback: show W as a heatmap
            im = plt.imshow(W_np, aspect="auto", origin="lower")
            plt.title(f"W (shape={W_np.shape})")
            plt.colorbar(im, fraction=0.046, pad=0.04)
        # plt.tight_layout()
        plt.savefig(out_path, dpi=200)
    finally:
        plt.close()


# ------------------------- training -------------------------
def train(model, Xtr, ytr, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    X = torch.from_numpy(Xtr).float().to(device)
    Y = torch.from_numpy(ytr).long().to(device)
    for _ in trange(epochs, leave=False, desc="train"):
        opt.zero_grad()
        out = model(X)
        l = loss(out, Y)
        l.backward(); opt.step()
    return model





def plot_samples(X, y, n, out_path):
    N, T, D = X.shape
    idx = np.random.choice(N, n, replace=False)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3*n), sharex=True)
    if n == 1: axes = [axes]
    for i, ax in zip(idx, axes):
        for d in range(D):
            ax.plot(X[i,:,d], label=f"feat {d}")
        ax.set_title(f"Series {i}, label={y[i]}")
        ax.legend()
    # plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    plt.show()




# ------------------------- main -------------------------
def main(args):
    # ---------- config override ----------
    if args.config is not None and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        # override argparse values if present in config
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    plots_dir = ensure_dir(os.path.join(os.getcwd(), "plots"))

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    X, y, true_W = generate_var_dataset(
        n_series=args.n_series, T=args.T, D=args.D, seed=args.seed
    )
    print("X.shape:", X.shape)
    plot_samples(X, y, n=3, out_path = os.path.join(plots_dir, "samples.png"))

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    model = LSTMClassifier(D=args.D, hidden=64, n_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train(model, Xtr, ytr, epochs=args.epochs, lr=1e-3, device=device)

    # pick one test sample
    x = torch.from_numpy(Xte[:1]).float()
    with torch.no_grad():
        pred = model(x.to(device)).argmax(dim=1).item()

    # Base attributions
    if args.method == "ig":
        A = integrated_gradients(model, x, target=pred, steps=32)
    else:
        A = random_mask_explainer(model, x, target=pred, n_masks=128, p_keep=0.2, seed=args.seed)

    # Build graph on top-k nodes using Shapley-style interactions
    W = build_graph_from_topk(
        model, x, pred, A, topk=args.topk, max_edges=args.max_edges,
        S=args.S, seed=args.seed, lags=(0,1,2)
    )

    # Refine attributions with Laplacian smoothing
    S_attr = laplacian_refine(A, W, lam=args.lam, iters=200)

    # Simple metric
    m = infidelity_comprehensiveness(A, S_attr)
    topk_overlap = float(m.get("topk_overlap", np.nan))
    print("Top-k overlap between base and refined:", topk_overlap)

    # Save numpy dumps
    np.save("A_base.npy", to_numpy(A))
    np.save("W_graph.npy", to_numpy(W))
    np.save("S_refined.npy", to_numpy(S_attr))
    print("Saved A_base.npy, W_graph.npy, S_refined.npy in CWD.")

    # ------------------ visualizations ------------------
    plot_heatmap(A, os.path.join(plots_dir, "A_base_heatmap.png"), title="Base Attributions (A)")
    plot_heatmap(S_attr, os.path.join(plots_dir, "S_refined_heatmap.png"), title="Refined Attributions (S)")
    plot_side_by_side(A, S_attr, os.path.join(plots_dir, "A_vs_S.png"))
    plot_topk_bars(A, S_attr, args.topk, os.path.join(plots_dir, "topk_bars.png"))
    try_draw_graph(W, os.path.join(plots_dir, "graph_W.png"))

    # Save metrics summary
    summary = []
    summary.append(f"predicted_class: {pred}")
    summary.append(f"topk_overlap(A,S): {topk_overlap:.4f}")
    if true_W is not None:
        # optional – if true graph given by synthetic generator
        try:
            tw = to_numpy(true_W)
            summary.append(f"true_W_shape: {tw.shape}")
        except Exception:
            pass
    save_txt(os.path.join(plots_dir, "metrics.txt"), "\n".join(summary))
    print(f"Plots & metrics saved to: {plots_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_series", type=int, default=256)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--D", type=int, default=6)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--method", type=str, default="ig", choices=["ig","timerise"])
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max_edges", type=int, default=64)
    p.add_argument("--S", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.3)
    args = p.parse_args()
    main(args)
