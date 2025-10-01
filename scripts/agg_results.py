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

def plot_metric_grid(df, metric, fig, k1_values, k2_values, T_values, metric_title):
    """
    Plot the grid of subplots for each combination of k1, k2, and T.
    Group rows so that all subplots with the same T are stacked together.
    """
    # Total rows = len(T_values) * len(k1_values)
    total_rows = len(T_values) * len(k1_values)
    gs = GridSpec(total_rows, len(k2_values), figure=fig)

    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            for t_idx, T in enumerate(T_values):
                # Group by T first, then k1
                ax_idx = t_idx * len(k1_values) + i
                ax = fig.add_subplot(gs[ax_idx, j])

                # Filter data for current k1, k2, T
                filtered_df = df[
                    (df['k1'] == int(k1)) & 
                    (df['k2'] == int(k2)) & 
                    (df['T'] == int(T))
                ]

                # Ensure dim(d) is numeric and sorted
                filtered_df = filtered_df.copy()
                filtered_df["dim (d)"] = filtered_df["dim (d)"].astype(int)
                filtered_df = filtered_df.sort_values("dim (d)")

                # Plot lineplot
                if not filtered_df.empty:
                    sns.lineplot(
                        data=filtered_df, 
                        x="dim (d)", 
                        y=metric, 
                        ax=ax, 
                        hue="model", 
                        style="model", 
                        markers=True
                    )
                ax.set_title(f'ER{k2}, T={T}', fontsize=9)
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax.set_xlabel('Number of Nodes (d)')
                ax.set_ylabel(f"{metric_title} (ER{k1})")
                ax.legend(title="Model", loc='best', fontsize=7)


if __name__ == "__main__":
    df = aggregate_runs("runs")
    print(df.head(100))

    # Ensure 'explainer' column exists
    if "explainer" not in df.columns:
        print("Column 'explainer' not found, skipping hue.")
        hue_column = None  # Skip hue if explainer column is missing
    else:
        hue_column = "explainer"  # Use explainer for hue if available

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
