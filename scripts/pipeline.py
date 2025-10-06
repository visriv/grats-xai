import argparse, yaml, os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.io_utils import set_seed, ensure_dir, save_pickle, save_json, NpEncoder
from src.datasets.synthetic_dbn import generate_dataset
from src.utils.training import build_model, train_classifier
from src.explainers.ig_wrapper import integrated_gradients
from src.explainers.time_rise import random_mask_explainer
from src.graph_recovery.asym_interaction import asymmetric_interaction_response
from src.graph_recovery.laplacian_refine import laplacian_refine_closed_form
from src.metrics.graph_metrics import graph_recovery_metrics
from src.metrics.faithfulness import comp_suff_curves
from src.utils.viz_utils import plot_attribution, plot_graph, plot_graph_recovery_summary, plot_graph_comparison, plot_auroc_drop
from src.evaluation.metrics import evaluate_auroc_drop
import numpy as np, torch, pickle, json
from sklearn.model_selection import train_test_split
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICE"] = '3'

import logging

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


EXPLAINER_REGISTRY = {
    "ig": integrated_gradients,
    "timerise": random_mask_explainer,
    # add more here
}

def run_pipeline(cfg_path):
    with open(cfg_path, "r") as f:
        CFG = yaml.safe_load(f)

    ROOT = Path(".").resolve()
    DATA_ROOT = ensure_dir(os.path.join(ROOT, "data"))
    RUNS_ROOT = ensure_dir(os.path.join(ROOT, "runs"))

    set_seed(CFG.get("seed", 0))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_exps   = CFG["data"]["experiments"]
    model_sweep = CFG["models"]
    expl_sweep  = CFG["explainers"]

    noise = CFG.get("noise", "normal")
    eta   = float(CFG.get("eta", 1.0))
    max_lag  = CFG.get("interactions", {}).get("max_lag", 2)
    rho   = float(CFG.get("interactions", {}).get("rho", 1.0))
    S     = int(CFG.get("interactions", {}).get("S", 1))
    lam   = float(CFG.get("refinement", {}).get("lambda", 0.3))
    ks_curve = CFG.get("metrics", {}).get("aopc_ks", [5,10,20,50])

    for exp in data_exps:
        # dataset name based on exp config
        ds_name = f"dbn_ns{exp.get('num_samples',200)}_T{exp['n']}_d{exp['d']}_p{exp['p']}_ER{exp['k_intra']}_ER{exp['k_inter']}"

        # data lives in ROOT/data/
        ds_data_dir = ensure_dir(os.path.join(DATA_ROOT, ds_name))
        tr_pkl = os.path.join(ds_data_dir, "train.pkl")
        va_pkl = os.path.join(ds_data_dir, "val.pkl")

        # runs (artifacts) live in ROOT/runs/
        ds_run_dir = ensure_dir(os.path.join(RUNS_ROOT, ds_name))

        # load or generate dataset
        if os.path.isfile(tr_pkl) and os.path.isfile(va_pkl):
            with open(tr_pkl, "rb") as f: train = pickle.load(f)
            with open(va_pkl, "rb") as f: val   = pickle.load(f)
            X_all = np.concatenate([train["X"], val["X"]], axis=0)
            y_all = np.concatenate([train["y"], val["y"]], axis=0)
            W_true = train["W_true"]; A_lags_all = train["A_lags_all"]
        else:
            X_all, y_all, W_true, A_lags_all = generate_dataset(
                num_samples=exp.get("num_samples", 200),
                T=exp["n"], d=exp["d"], p=exp["p"],
                k_intra=exp["k_intra"], k_inter=exp["k_inter"],
                model_intra=CFG.get("model_intra", "ER"),
                model_inter=CFG.get("model_inter", "ER"),
                noise=noise, eta=eta, seed=CFG.get("seed", 0)
            )
            Xtr, Xva, ytr, yva = train_test_split(
                X_all, y_all, test_size=0.2,
                random_state=CFG.get("seed", 0), stratify=y_all
            )
            save_pickle(tr_pkl, {"X": Xtr, "y": ytr, "W_true": W_true, "A_lags_all": A_lags_all})
            save_pickle(va_pkl, {"X": Xva, "y": yva, "W_true": W_true, "A_lags_all": A_lags_all})
            train, val = {"X": Xtr, "y": ytr}, {"X": Xva, "y": yva}
            # end dataset
            log.info(f"[dataset] ready: {ds_name} | train {len(train['X'])} | val {len(val['X'])}")
        # end dataset

        Xtr, ytr = train["X"], train["y"]
        Xva, yva = val["X"], val["y"]
        Xtr = np.transpose(Xtr, (0, 2, 1))
        Xva = np.transpose(Xva, (0, 2, 1))

        D, T = Xtr.shape[1], Xtr.shape[2]
        C = int(np.max(np.concatenate([ytr,yva])) + 1)



        # Print class distribution for training and validation data
        print("Class distribution in training data:")
        train_classes, train_counts = np.unique(ytr, return_counts=True)
        for cls, count in zip(train_classes, train_counts):
            print(f"Class {cls}: {count} samples")

        print("\nClass distribution in validation data:")
        val_classes, val_counts = np.unique(yva, return_counts=True)
        for cls, count in zip(val_classes, val_counts):
            print(f"Class {cls}: {count} samples")


        for mcfg in model_sweep:
            model_name = mcfg["name"]
            model_dir  = ensure_dir(os.path.join(ds_run_dir, model_name))
            model_ckpt = os.path.join(model_dir, "model.pt")

            if os.path.isfile(model_ckpt):
                log.info(f"[model] found pretrained {model_name} for {ds_name}, loading...")
                model = build_model(model_name, D, C, mcfg)
                model.load_state_dict(torch.load(model_ckpt, map_location=device))
                model.to(device)
            else:
                log.info(f"[model] training {model_name} on dataset {ds_name}")

                model = build_model(model_name, D, C, mcfg)
                model = train_classifier(
                    model, Xtr, ytr, Xva, yva,
                    epochs=mcfg.get("epochs",8),
                    lr=mcfg.get("lr",1e-3),
                    batch_size=mcfg.get("batch_size", 32),
                    device=device
                )
                torch.save(model.state_dict(), model_ckpt)
                log.info(f"[model] saved checkpoint to {model_ckpt}")

            for ecfg in expl_sweep:
                expl_name = ecfg["name"].lower()
                if expl_name not in EXPLAINER_REGISTRY:
                    raise ValueError(f"Unknown explainer: {expl_name}")
                explainer_fn = EXPLAINER_REGISTRY[expl_name]
                expl_kwargs = {k: v for k, v in ecfg.items() if k != "name"}

                run_dir   = ensure_dir(os.path.join(model_dir, expl_name))
                attr_dir  = ensure_dir(os.path.join(run_dir, "attr"))
                graph_dir = ensure_dir(os.path.join(run_dir, "graph"))
                metr_dir  = ensure_dir(os.path.join(run_dir, "metrics"))
                plot_dir  = ensure_dir(os.path.join(run_dir, "plots"))

                metr_file = os.path.join(metr_dir, "metrics.json")
                attr_npz = os.path.join(attr_dir, "attr_base_summary.npz")

                # 1.1) ------------------ Explanation ------------------

                if os.path.isfile(attr_npz):
                    log.info(f"[skip] {ds_name} | {model_name} | {expl_name} explanation already done.")
                    log.info(f"[skip] {ds_name} | {model_name} | {expl_name} moving to evaluation.")
                else:



                    # ------------------ batch eval config ------------------
                    max_eval = int(CFG.get("max_eval_samples", len(Xva)))
                    idxs     = np.arange(min(len(Xva), max_eval))
                    eval_bs  = int(CFG.get("eval_batch_size", 128))
                    viz_k    = int(CFG.get("viz_k", 6))
                    log.info(f"[explainer] {expl_name}: evaluating {len(idxs)} samples in batches of {eval_bs}")

                    # (N_eval, D, T) on device
                    X_eval = torch.from_numpy(Xva[idxs]).float().to(device)

                    # 1) forward once to get predicted targets per sample
                    with torch.no_grad():
                        logits  = model(X_eval)                         # (N_eval, C)
                        targets = logits.argmax(dim=1)                  # (N_eval,)
                        base_p  = torch.softmax(logits, dim=1)

                    # 2) Explanation for the whole set (in chunks)
                    steps = int(ecfg.get("steps", 32))
                    attr_chunks = []
                    for s in tqdm(range(0, len(idxs), eval_bs), desc=f" {expl_name}", leave=False):
                        xb = X_eval[s:s+eval_bs]                        # (b, D, T)
                        tb = targets[s:s+eval_bs]                       # (b,)
                        attr_b = explainer_fn(model, xb, target=tb, **expl_kwargs) # (b, D, T)
                        if isinstance(attr_b, torch.Tensor):
                            attr_b = attr_b.detach().cpu().numpy()
                        attr_chunks.append(attr_b)
                    attr_batch = np.concatenate(attr_chunks, axis=0)          # (N_eval, D, T)

                    # Save compact attribution summary (mean/std) + plots
                    attr_mean = attr_batch.mean(axis=0)                       # (D, T)
                    attr_std  = attr_batch.std(axis=0)                        # (D, T)
                    np.savez(os.path.join(attr_dir, "attr_base_summary.npz"), mean=attr_mean, std=attr_std)


                # 2.1) ------------------ AUROC Drop Evaluation ------------------
                # Read the auroc_ks from the config
                auroc_ks = ecfg.get("auroc_ks", [5, 10, 20, 30, 50])

                # Define the file path to save AUROC drop results
                auroc_drop_file = os.path.join(attr_dir, "auroc_drop.npy")  # or .pkl

                # Check if the file already exists, and skip evaluation if it does
                if os.path.isfile(auroc_drop_file):
                    print(f"[skip] AUROC drop already evaluated for {expl_name}. Loading existing results.")
                    auroc_drops = np.load(auroc_drop_file)  # or pickle.load for .pkl
                else:
                    # Evaluate AUROC drop after occluding top-K salient points
                    auroc_drops = evaluate_auroc_drop(model, X_eval, targets.cpu().numpy(), attr_batch, auroc_ks, plot_dir)
                    # Save AUROC drops to file (as .npy or .pkl)
                    np.save(auroc_drop_file, auroc_drops)  # Or use pickle.dump for .pkl
                    print(f"[saved] AUROC drop results for {expl_name} saved to {auroc_drop_file}")

                # Plot the AUROC drop vs. top-K
                plot_auroc_drop(auroc_ks, auroc_drops, expl_name, plot_dir)


                
                # 3) GLOBAL lagged interactions from the whole eval set (vectorized)
                # asymmetric_interaction_response expects:
                #   x: torch.Tensor (B,D,T)   attr_batch: np.ndarray (B,D,T)   target: torch.LongTensor (B,)
                w_hat_file = os.path.join(graph_dir, "W_hat_global.npy")
                if os.path.isfile(w_hat_file):
                    log.info(f"[skip] {ds_name} | {model_name} | graph estimation already done ")
                else:                
                    W_hat_global = asymmetric_interaction_response(
                        model,
                        X_eval,                           # (N_eval, D, T) tensor on device
                        attr_batch,                          # (N_eval, D, T) numpy
                        targets,                          # (N_eval,) tensor on device
                        max_lag=max_lag, rho=rho, S=S, device=device
                    )                                     # -> (D, D, L) (ensure that max_lag should be p, hence L = p+1)
                    # Plot global feature graph (sum over lags)
                    plot_graph(
                        W_hat_global.sum(axis=-1),
                        os.path.join(plot_dir, "W_hat_global_feat.png"),
                        title=f"Global Feature Graph ({expl_name})"
                    )   
                    np.save(os.path.join(graph_dir, "W_hat_global.npy"), W_hat_global)


                # 4) Aggregate ground truth across the same subset and compute global metrics
                # TODO: redundant code here
                L_true = len(A_lags_all[0])
                L_use  = min(L_true, W_hat_global.shape[-1])

                gr_global = graph_recovery_metrics(W_hat_global[:, :, :L_use], [W_true] + A_lags_all)

                plot_graph_comparison(
                    W_hat_global[:, :, :L_use], [W_true] + A_lags_all,
                    os.path.join(plot_dir, "graph_comp_global.png"),
                    title=f"Graph recovery (GLOBAL)"
                )

                # 5) Laplacian refinement of MEAN attribution using GLOBAL graph 
                attr_refined_path = os.path.join(graph_dir, "attr_refined.npy")
                if os.path.isfile(attr_refined_path):
                    log.info(f"[skip] {ds_name} | {model_name} | {expl_name} attr refinement already already done ")
                W_feat_global = W_hat_global.sum(axis=-1)               # (D,D)
                attr_ref = laplacian_refine_closed_form(attr_batch, W_feat_global, lam)  # (D,T)
                np.save(os.path.join(attr_dir, "attr_refined.npy"), attr_ref)


                # 6) Optional: a few example plots (base + refined via global graph)
                if viz_k > 0:
                    ex = idxs[:min(viz_k, len(idxs))]
                    for j, i in enumerate(ex):
                        Aj = attr_batch[j]  # j aligns with first 'viz_k' indices
                        plot_attribution(Aj, os.path.join(plot_dir, f"attr_base_ex{i}.png"),
                                        title=f"attr base (example {i})")
                        attr_ref_j = laplacian_refine_closed_form(Aj, W_feat_global, lam)
                        plot_attribution(attr_ref_j, os.path.join(plot_dir, f"attr_refined_ex{i}.png"),
                                        title=f"attr refined (example {i})")

                # 7) Save compact metrics (global only, no per-sample clutter)
                ensure_dir(metr_dir)
                with open(os.path.join(metr_dir, "metrics.json"), "w") as f:
                    json.dump({
                        "global_graph_recovery": gr_global,
                        "eval_samples": int(len(idxs)),
                        "eval_batch_size": int(eval_bs),
                        "lags_used": list(range(max_lag+1)),
                        "attr_summary": {
                            "mean_path": "attr/attr_base_summary.npz",
                            "refined_path": "attr/attr_refined.npy"
                        }
                    }, f, indent=2, cls=NpEncoder)

                log.info(f"[metrics] saved: {os.path.join(metr_dir, 'metrics.json')}")

                

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    run_pipeline(args.config)
