
import argparse, numpy as np, torch, torch.nn as nn
from datasets.synthetic_var import generate_var_dataset
from models.simple_lstm import LSTMClassifier
from explainers.ig_wrapper import integrated_gradients
from explainers.time_rise import random_mask_explainer
from graphs.interaction_shapley import build_graph_from_topk
from graphs.laplacian_refine import laplacian_refine
from evaluation.metrics import infidelity_comprehensiveness
from sklearn.model_selection import train_test_split
from tqdm import trange

def train(model, Xtr, ytr, epochs=10, lr=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    X = torch.from_numpy(Xtr).float().to(device)
    Y = torch.from_numpy(ytr).long().to(device)
    for _ in trange(epochs, leave=False, desc="train"):
        opt.zero_grad()
        out = model(X)
        l = loss(out, Y)
        l.backward(); opt.step()
    return model

def main(args):
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    X, y, true_W = generate_var_dataset(n_series=args.n_series, T=args.T, D=args.D, seed=args.seed)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)
    model = LSTMClassifier(D=args.D, hidden=64, n_classes=2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train(model, Xtr, ytr, epochs=args.epochs, lr=1e-3, device=device)

    # pick one test sample
    x = torch.from_numpy(Xte[:1]).float()
    with torch.no_grad():
        pred = model(x.to(device)).argmax(dim=1).item()

    # Base attributions
    if args.method == "ig":
        A = integrated_gradients(model, x, target=pred, steps=32)
    else:
        A = random_mask_explainer(model, x, target=pred, n_masks=128, p_keep=0.2, seed=args.seed)

    # Build graph on top-k nodes using Shapley-style interactions
    W = build_graph_from_topk(model, x, pred, A, topk=args.topk, max_edges=args.max_edges, S=args.S, seed=args.seed, lags=(0,1,2))

    # Refine attributions with Laplacian smoothing
    S_attr = laplacian_refine(A, W, lam=args.lam, iters=200)

    # Simple metric
    m = infidelity_comprehensiveness(A, S_attr)
    print("Top-k overlap between base and refined:", m["topk_overlap"])

    # Save numpy dumps
    np.save("A_base.npy", A); np.save("W_graph.npy", W); np.save("S_refined.npy", S_attr)
    print("Saved A_base.npy, W_graph.npy, S_refined.npy in CWD.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_series", type=int, default=256)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--D", type=int, default=6)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--method", type=str, default="ig", choices=["ig","timerise"])
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max_edges", type=int, default=64)
    p.add_argument("--S", type=int, default=16)
    p.add_argument("--lam", type=float, default=0.3)
    args = p.parse_args()
    main(args)
