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
            ds, model, expl = parts[-3], parts[-2], parts[-1]
            
            # Extract the graph recovery metrics from the JSON
            if "global_graph_recovery" in metrics:
                global_metrics = metrics["global_graph_recovery"]
                row = {
                    "dataset": ds,
                    "model": model,
                    "explainer": expl,
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
    print(df.head())

    # Plot comparison of F1 scores across datasets
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="dataset", y="f1", hue="explainer", style="model", marker="o")
    plt.xticks(rotation=45)
    plt.title("Graph Recovery F1 across datasets")
    plt.tight_layout()
    plt.savefig("runs/graph_recovery_comparison.png")

    # Optionally, plot ROC AUC and PR AUC comparisons as well
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="dataset", y="roc_auc", hue="explainer", style="model", marker="o")
    plt.xticks(rotation=45)
    plt.title("Graph Recovery ROC AUC across datasets")
    plt.tight_layout()
    plt.savefig("runs/graph_recovery_roc_auc_comparison.png")

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="dataset", y="pr_auc", hue="explainer", style="model", marker="o")
    plt.xticks(rotation=45)
    plt.title("Graph Recovery PR AUC across datasets")
    plt.tight_layout()
    plt.savefig("runs/graph_recovery_pr_auc_comparison.png")
