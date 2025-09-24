import os, json, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path

def aggregate_runs(runs_root="runs"):
    rows = []
    for root, _, files in os.walk(runs_root):
        if "metrics.json" in files:
            with open(os.path.join(root, "metrics.json"), "r") as f:
                metrics = json.load(f)
            # parse dataset, model, explainer
            parts = Path(root).parts
            ds, model, expl = parts[-3], parts[-2], parts[-1]
            df = pd.DataFrame([m["graph_recovery"] for m in metrics])
            row = df.mean().to_dict()
            row.update({"dataset": ds, "model": model, "explainer": expl})
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = aggregate_runs("runs")
    print(df.head())

    plt.figure(figsize=(8,6))
    sns.lineplot(data=df, x="dataset", y="f1", hue="explainer", style="model", marker="o")
    plt.xticks(rotation=45)
    plt.title("Graph Recovery F1 across datasets")
    plt.tight_layout()
    plt.savefig("runs/graph_recovery_comparison.png")
