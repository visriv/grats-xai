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

    # Ensure axes is always 2D [2, L]
    if L == 1:
        axes = np.array([[axes[0]], [axes[1]]])   # special case
    else:
        axes = np.array(axes).reshape(2, L)

    for lag in range(L):
        # True
        vmax1 = np.abs(W_true_list[lag]).max() + 1e-6  # per subplot vmax
        sns.heatmap(W_true_list[lag], ax=axes[0, lag], cmap="RdBu_r", center=0,
                    cbar=True, vmin=-vmax1, vmax=vmax1, annot=False,
                    cbar_kws={"shrink": 0.8})  # individual colorbar
        if lag == 0:
            axes[0, lag].set_title(r"$W_{true," + str(lag) + "}$")
        else:
            axes[0, lag].set_title(r"$A_{true," + str(lag) + "}$")

        # Estimated
        vmax2 = np.abs(W_hat[:, :, lag]).max() + 1e-6  # per subplot vmax
        sns.heatmap(W_hat[:, :, lag], ax=axes[1, lag], cmap="RdBu_r", center=0,
                    cbar=True, vmin=-vmax2, vmax=vmax2, annot=False,
                    cbar_kws={"shrink": 0.8})  # individual colorbar
        axes[1, lag].set_title(r"$\hat{W}_{" + str(lag) + "}$")

    axes[0, 0].set_ylabel("True")
    axes[1, 0].set_ylabel("Estimated")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# Function to plot AUROC drop vs top-K salient points
def plot_auroc_drop(auroc_ks, auroc_drops, expl_name, plot_dir):
    """
    Plot AUROC drop vs Top-K salient points and save the plot.
    
    Args:
        auroc_ks: List of top-K values.
        auroc_drops: Corresponding AUROC drop values for each K.
        expl_name: Explainer name (used for the plot title).
        plot_dir: Directory to save the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(auroc_ks, auroc_drops, marker='o', color='b', linestyle='-', linewidth=2)
    plt.xlabel("Top-K salient points")
    plt.ylabel("AUROC Drop")
    plt.title(f"AUROC Drop after Occlusion - {expl_name}")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_file = os.path.join(plot_dir, f"auroc_drop_{expl_name}.png")
    plt.savefig(plot_file, dpi=200)
    plt.close()
