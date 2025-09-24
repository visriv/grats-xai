import torch, torch.nn as nn, torch.optim as optim
from src.models.lstm import LSTMClassifier
from src.models.tcn import TCNClassifier
from src.models.transformer import TransformerClassifier
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score



def build_model(name: str, D: int, C: int, cfg: dict):
    if name.lower() == "lstm":
        return LSTMClassifier(D, hidden=cfg.get("hidden", 64), n_classes=C)
    elif name.lower() == "tcn":
        return TCNClassifier(D, hidden=cfg.get("hidden", 64), n_classes=C)
    elif name.lower() == "transformer":
        return TransformerClassifier(D, hidden=cfg.get("hidden", 64), n_classes=C)
    else:
        raise ValueError(f"Unknown model: {name}")



def train_classifier(
    model, X, y, Xva=None, yva=None, epochs=10, lr=1e-3, batch_size=32, device="cpu"
):
    """
    Mini-batch training loop for classifiers with optional validation metrics.
    
    Args:
        model : torch.nn.Module
        X : np.ndarray (N, D, T) - training data
        y : np.ndarray (N,) - training labels
        Xva, yva : validation data/labels (optional)
        epochs : int
        lr : float
        batch_size : int
        device : str
    """
    model.to(device)

    # Wrap training data
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        avg_loss = total_loss / len(dataset)
        print(f"[epoch {ep+1}/{epochs}] loss={avg_loss:.4f}")

        # --- Validation metrics ---
        if Xva is not None and yva is not None:
            model.eval()
            with torch.no_grad():
                Xva_t = torch.from_numpy(Xva).float().to(device)
                yva_t = torch.from_numpy(yva).long().to(device)
                logits = model(Xva_t)
                preds = logits.argmax(dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy() if logits.shape[1] > 1 else None

            prec = precision_score(yva, preds, average="macro", zero_division=0)
            rec  = recall_score(yva, preds, average="macro", zero_division=0)
            f1   = f1_score(yva, preds, average="macro", zero_division=0)
            auroc = roc_auc_score(yva, probs) if probs is not None and len(set(yva)) == 2 else float("nan")

            print(f"  [val] precision={prec:.3f} recall={rec:.3f} f1={f1:.3f} auroc={auroc:.3f}")

    return model