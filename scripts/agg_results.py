import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

            # Extract the graph recovery metrics from the JSON
            if "global_graph_recovery" in metrics:
                global_metrics = metrics["global_graph_recovery"]
                row = {
                    "dataset": ds,
                    "dim (d)": d,
                    "k1": k1,
                    "k2": k2,
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

def plot_metric_grid(df, metric, ax, k1_values, k2_values, metric_title):
    """
    Plot the grid of subplots for each combination of k1 and k2.
    """
    for i, k1 in enumerate(k1_values):
        for j, k2 in enumerate(k2_values):
            # Filter data for current k1, k2 combination
            filtered_df = df[(df['k1'] == str(k1)) & (df['k2'] == str(k2))]

            # Plot the data for this specific k1, k2
            sns.lineplot(data=filtered_df, x="dim (d)", y=metric, ax=ax[i, j], hue="model", style="model", markers=True)
            ax[i, j].set_title(f'k1={k1}, k2={k2}')
            ax[i, j].grid(True, which='both', linestyle='--', linewidth=0.5)
            ax[i, j].set_xlabel('Number of Nodes (d)')
            ax[i, j].set_ylabel(metric_title)
            ax[i, j].legend(title="Model", loc='best')

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

    for metric in metrics:
        # Create a grid of subplots for each metric
        fig, axes = plt.subplots(len(k1_values), len(k2_values), figsize=(15, 12), sharex=True, sharey=True)
        axes = axes.ravel()  # Flatten the 2D array to iterate through each subplot easily

        # Plot the grid for this metric
        plot_metric_grid(df, metric, axes.reshape(len(k1_values), len(k2_values)), k1_values, k2_values, metric_titles[metric])

        # Adjust layout and save the plot
        plt.tight_layout()
        plot_filename = f"runs/{metric}_comparison_grid.png"
        plt.savefig(plot_filename, dpi=200)
        plt.close()
