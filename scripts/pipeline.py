import argparse, yaml, os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.io_utils import set_seed, ensure_dir, save_pickle, save_json, NpEncoder
from src.datasets.synthetic_dbn import load_or_generate_dataset, generate_dataset
from src.utils.training import build_model, train_classifier
from src.explainers.ig_wrapper import integrated_gradients
from src.explainers.time_rise import random_mask_explainer
from src.graph_recovery.asym_interaction import asymmetric_interaction_response
from src.graph_recovery.laplacian_refine import laplacian_refine_closed_form
from src.metrics.graph_metrics import graph_recovery_metrics
from src.metrics.faithfulness import comp_suff_curves
from src.utils.viz_utils import plot_attribution, plot_graph, plot_graph_recovery_summary, plot_graph_comparison
from src.utils.compute_utils import _aggregate_true_lags, _stack_mean_abs

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
    lags  = CFG.get("interactions", {}).get("lags", [0,1,2])
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
            W_list = train["W_list"]; A_lags_all = train["A_lags_all"]
        else:
            X_all, y_all, W_list, A_lags_all = generate_dataset(
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
            save_pickle(tr_pkl, {"X": Xtr, "y": ytr, "W_list": W_list, "A_lags_all": A_lags_all})
            save_pickle(va_pkl, {"X": Xva, "y": yva, "W_list": W_list, "A_lags_all": A_lags_all})
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
                expl_name = ecfg["name"]
                run_dir   = ensure_dir(os.path.join(model_dir, expl_name))
                attr_dir  = ensure_dir(os.path.join(run_dir, "attr"))
                graph_dir = ensure_dir(os.path.join(run_dir, "graph"))
                metr_dir  = ensure_dir(os.path.join(run_dir, "metrics"))

                metr_file = os.path.join(metr_dir, "metrics.json")
                if os.path.isfile(metr_file):
                    log.info(f"[skip] {ds_name} | {model_name} | {expl_name} already done.")
                    continue

                ## Computation
                idxs = np.arange(min(len(Xva), CFG.get("max_eval_samples", len(Xva))))
                log.info(f"[explainer] running {expl_name} on {len(idxs)} validation samples")

                W_per_sample = []   # <-- collect per-sample What for global aggregation
                metrics_list = []

                for i in tqdm(idxs, desc=f"iterating over val samples: {ds_name}|{model_name}|{expl_name}"):

                    x = torch.from_numpy(Xva[i:i+1]).float()
                    yhat = model(x.to(device)).argmax(dim=1).item()

                    # Base attribution
                    attr = integrated_gradients(model, x, target=yhat, steps=ecfg.get("steps",32))
                    np.save(os.path.join(attr_dir, f"A_base_{i}.npy"), attr)

                    # log.info(f"[explainer] completed {expl_name} on this validation sample")
                    # log.info(f"[Estimating graph] estimating W now")

                    # Graph interactions
                    What_lag = asymmetric_interaction_response(model, x, attr, yhat, lags, rho, S, device)
                    np.save(os.path.join(graph_dir, f"W_lag_{i}.npy"), What_lag)
                    W_per_sample.append(What_lag)   # <-- collect for global





                    # Laplacian refinement
                    W_feat = What_lag.sum(axis=-1)
                    att_ref = laplacian_refine_closed_form(attr, W_feat, lam)
                    np.save(os.path.join(attr_dir, f"attr_refined_{i}.npy"), att_ref)

                    # Save plots
                    plot_dir = ensure_dir(os.path.join(run_dir, "plots"))
                    plot_attribution(attr, os.path.join(plot_dir, f"attr_base_{i}.png"), title=f"attr base (sample {i})")
                    plot_attribution(att_ref, os.path.join(plot_dir, f"attr_refined_{i}.png"), title=f"attr refined (sample {i})")

                    # plot_graph(What_lag.sum(axis=-1), os.path.join(plot_dir, f"W_feat_{i}.png"), title=f"Feature graph (sample {i})")

                    # Metrics
                    A_true_lags_all = A_lags_all[i]
                    gr = graph_recovery_metrics(What_lag, A_true_lags_all)
                    ks = [k for k in ks_curve if k <= attr.size] or [min(10, attr.size)]
                    cs_base = comp_suff_curves(model, x, yhat, attr, ks, device)
                    cs_ref  = comp_suff_curves(model, x, yhat, att_ref, ks, device)

                    # Graph recovery: Per-sample true vs estimated plot
                    # plot_graph_comparison(
                    #     What_lag, A_true_lags_all,
                    #     os.path.join(plot_dir, f"graph_comp_{i}.png"),
                    #     title=f"Graph recovery (sample {i})"
                    # )

                    metrics_list.append({
                        "idx": i,
                        "graph_recovery": gr,
                        "comp_suff_base": cs_base,
                        "comp_suff_ref": cs_ref
                    })


                # -------- GLOBAL aggregation over samples --------
                # W_global (D,D,L) by mean-abs over the evaluated validation subset
                W_global = _stack_mean_abs(W_per_sample)
                np.save(os.path.join(graph_dir, "W_global.npy"), W_global)
                
                # Plot global feature graph (aggregate over lags)
                plot_graph(
                    W_global.sum(axis=-1),
                    os.path.join(plot_dir, "W_global_feat.png"),
                    title=f"Global Feature Graph ({expl_name})"
                )
                
                # Aggregate ground truth across same subset
                A_true_global_lags = _aggregate_true_lags(A_lags_all, idxs)
                np.save(os.path.join(graph_dir, "A_true_global.npy"), np.stack(A_true_global_lags, axis=-1))

                # Global graph recovery metrics
                gr_global = graph_recovery_metrics(W_global, A_true_global_lags)
                # Global figure: side-by-side True vs Estimated per lag
                plot_graph_comparison(
                    W_global, A_true_global_lags,
                    os.path.join(plot_dir, "graph_comp_global.png"),
                    title=f"Graph recovery (GLOBAL)"
                )

                # per sample plot (not useful)
                plot_dir = ensure_dir(os.path.join(run_dir, "plots"))
                plot_graph_recovery_summary(metrics_list, os.path.join(plot_dir, "graph_recovery_summary.png"), expl_name)

                
                # Save both per-sample and global metrics
                with open(os.path.join(metr_dir, "metrics.json"), "w") as f:
                    json.dump({
                        "per_sample": metrics_list,
                        "global_graph_recovery": gr_global
                    }, f, indent=2, cls=NpEncoder)

                log.info(f"[metrics] saved: {os.path.join(metr_dir, 'metrics.json')}")

                

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    run_pipeline(args.config)
