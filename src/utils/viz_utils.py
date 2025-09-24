import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns


def plot_attribution(A, out_path, title="Attribution"):
    plt.figure(figsize=(6, 4))
    plt.imshow(A, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_graph(W, out_path, title="Graph Weights"):
    plt.figure(figsize=(5, 5))
    plt.imshow(W, cmap="bwr", vmin=-np.max(np.abs(W)), vmax=np.max(np.abs(W)))
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Feature j")
    plt.ylabel("Feature i")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()



def plot_graph_recovery_summary(metrics_list, out_path, expl_name):
    df = pd.DataFrame([m["graph_recovery"] for m in metrics_list])
    mean = df.mean(); std = df.std()

    fig, ax = plt.subplots(figsize=(6,4))
    df.boxplot(ax=ax)
    ax.set_title(f"Graph Recovery ({expl_name})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()




def plot_graph_comparison(W_hat, W_true_list, save_path, title="Graph Recovery"):
    """
    Side-by-side heatmaps of True vs Estimated adjacency matrices.

    Args:
        W_hat: np.ndarray, shape (D, D, L)
            Predicted lag-weight matrices.
        W_true_list: list[np.ndarray], each (D, D)
            Ground-truth lag matrices. (first one being W_true, others being A^(l))
        save_path: str
            Where to save the PNG.
        title: str
            Plot title.
    """
    L_pred = W_hat.shape[-1]
    L_true = len(W_true_list)
    L = min(L_pred, L_true)

    fig, axes = plt.subplots(2, L, figsize=(3*L, 6))
    vmax1 = np.max([np.abs(A).max() for A in W_true_list]) + 1e-6
    vmax2 = np.abs(W_hat).max() + 1e-6

    # Ensure axes is always 2D [2, L]
    if L == 1:
        axes = np.array([[axes[0]], [axes[1]]])   # special case
    else:
        axes = np.array(axes).reshape(2, L)

    for lag in range(L):
        # True
        sns.heatmap(W_true_list[lag], ax=axes[0, lag], cmap="RdBu_r", center=0,
                    cbar=(lag==L-1), vmin=-vmax1, vmax=vmax1, annot=False)
        axes[0, lag].set_title(f"True (lag {lag+1})")

        # Estimated
        sns.heatmap(W_hat[:, :, lag], ax=axes[1, lag], cmap="RdBu_r", center=0,
                    cbar=(lag==L-1), vmin=-vmax2, vmax=vmax2, annot=False)
        axes[1, lag].set_title(f"Estimated (lag {lag+1})")

    axes[0, 0].set_ylabel("True")
    axes[1, 0].set_ylabel("Estimated")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
