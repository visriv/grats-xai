import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec

def aggregate_runs(runs_root="runs"):
    rows = []
    for root, _, files in os.walk(runs_root):
        if "metrics.json" in files:
            with open(os.path.join(root, "metrics.json"), "r") as f:
                metrics = json.load(f)
            # Parse dataset, model, and explainer
            parts = Path(root).parts
            ds, model, expl = parts[-4], parts[-3], parts[-2]
            d = ds.split('_')[3][1:]
            k1 = ds.split('_')[-2][2:]
            k2 = ds.split('_')[-1][2:]
            T = ds.split('_')[2][1:]

            # Extract the graph recovery metrics from the JSON
            if "global_graph_recovery" in metrics:
                global_metrics = metrics["global_graph_recovery"]
                row = {
                    "dataset": ds,
                    "dim (d)": d,
                    "k1": int(k1),
                    "k2": int(k2),
                    "T": int(T),
                    "model": model,
                    "explainer": expl,  # keep explainer if it exists
                    "roc_auc": global_metrics.get("roc_auc", None),
                    "pr_auc": global_metrics.get("pr_auc", None),
                    "f1": global_metrics.get("f1", None),
                    "shd": global_metrics.get("shd", None),
                    "K": global_metrics.get("K", None),
                    "n": global_metrics.get("n", None),
                }
                rows.append(row)
            else:
                print(f"Skipping directory {root} (missing 'global_graph_recovery' metrics)")
    
    return pd.DataFrame(rows)

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_metric_grid(df, metric, fig, k1_values, k2_values, T_values, metric_title):
    """
    Plot a grid of subplots for each combination of k1, k2, and T.
    Group rows so that all subplots with the same T are stacked together.
    Each subplot shows one line per explainer.
    """
    # ICML-style aesthetics
    sns.set(style="whitegrid", font_scale=1.1, rc={
        "axes.edgecolor": "0.3",
        "axes.linewidth": 0.6,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "grid.color": "0.85",
        "grid.linestyle": "--",
        "legend.frameon": False
    })
    palette = sns.color_palette("husl", n_colors=df["explainer"].nunique())

    # Total rows = len(T_values) * len(k1_values)
    total_rows = len(T_values) * len(k1_values)
    gs = GridSpec(total_rows, len(k2_values), figure=fig)

    for i, k1 in enumerate(sorted(map(int, k1_values))):
        for j, k2 in enumerate(sorted(map(int, k2_values))):
            for t_idx, T in enumerate(sorted(map(int, T_values))):
                # Compute subplot index (grouped by T)
                ax_idx = t_idx * len(k1_values) + i
                ax = fig.add_subplot(gs[ax_idx, j])

                # Filter dataframe
                filtered_df = df[
                    (df["k1"].astype(int) == k1) &
                    (df["k2"].astype(int) == k2) &
                    (df["T"].astype(int) == T)
                ].copy()

                if filtered_df.empty:
                    ax.set_visible(False)
                    continue

                # Ensure dim(d) numeric and sorted
                filtered_df["dim (d)"] = filtered_df["dim (d)"].astype(int)
                filtered_df = filtered_df.sort_values("dim (d)")

                # Plot one curve per explainer
                sns.lineplot(
                    data=filtered_df,
                    x="dim (d)",
                    y=metric,
                    hue="explainer",
                    style="explainer",
                    markers=True,
                    dashes=False,
                    palette=palette,
                    linewidth=1.8,
                    ax=ax
                )

                # Aesthetic adjustments
                ax.set_title(f"ER{k2}, T={T}", fontsize=9)
                ax.set_xlabel("Number of Nodes (d)", fontsize=9)
                ax.set_ylabel(f"{metric_title} (ER{k1})", fontsize=9)
                ax.grid(True, linestyle="--", alpha=0.6)
                ax.legend(fontsize=7, title="Explainer", loc="lower right", frameon=False)

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

if __name__ == "__main__":
    df = aggregate_runs("runs")
    print(df.head(100))


    # Metrics to be plotted
    metrics = ["f1", "roc_auc", "pr_auc", "shd"]
    metric_titles = {
        "f1": "F1 Score",
        "shd": "SHD",
        "roc_auc": "AUROC",
        "pr_auc": "AUPRC"
    }

    k1_values = df["k1"].unique()  # Unique values of k1
    k2_values = df["k2"].unique()  # Unique values of k2
    T_values = df["T"].unique()  # Unique values of T

    print(k1_values)
    print(k2_values)
    
    for metric in metrics:
        # Create a figure for this metric
        fig = plt.figure(figsize=(15, 18))

        # Plot the grid for this metric
        plot_metric_grid(df, metric, fig, k1_values, k2_values, T_values, metric_titles[metric])

        # Adjust layout and save the plot
        plt.tight_layout()
        plot_filename = f"runs/{metric}_comparison_grid_with_T.png"
        plt.savefig(plot_filename, dpi=400)
        plt.close()
