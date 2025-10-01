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

if __name__ == "__main__":
    df = aggregate_runs("runs")
    print(df.head(100))

    # Ensure 'explainer' column exists
    if "explainer" not in df.columns:
        print("Column 'explainer' not found, skipping hue.")
        hue_column = None  # Skip hue if explainer column is missing
    else:
        hue_column = "explainer"  # Use explainer for hue if available

    # Create a 2x2 plot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=False)
    axes = axes.ravel()  # Flatten the 2x2 grid to easily iterate

    # Define metrics
    metrics = ["f1", "shd", "roc_auc", "pr_auc"]
    metric_titles = {
        "f1": "F1 Score",
        "shd": "SHD",
        "roc_auc": "AUROC",
        "pr_auc": "AUPRC"
    }

    # Iterate over each metric and plot
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(data=df, x="dim (d)", y=metric, hue="model", style="model", markers=True, ax=ax)

        # Set the title, labels, and grids for each subplot
        ax.set_title(f'{metric_titles[metric]} vs Number of Nodes')
        ax.set_xlabel('Number of Nodes (d)')
        ax.set_ylabel(metric_titles[metric])
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Set y-axis scale depending on the metric
        if metric == "f1":
            ax.set_ylim(0, 1)  # F1 score ranges from 0 to 1
        elif metric == "shd":
            ax.set_ylim(0, 500)  # SHD could be higher (adjust as necessary)
        elif metric == "roc_auc":
            ax.set_ylim(0, 1)  # AUROC ranges from 0 to 1
        elif metric == "pr_auc":
            ax.set_ylim(0, 1)  # AUPRC ranges from 0 to 1

        # Add legend for model types
        ax.legend(title="Model", loc='best')

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig("runs/graph_recovery_comparison_metrics.png", dpi=200)
    plt.show()
