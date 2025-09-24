import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

def graph_recovery_metrics(W_hat, A_true_list, tau=0.1, binarize_thr=1e-8, f1_mode="k"):
    """
    Evaluate edge recovery across lags.

    Args
    ----
    W_hat : np.ndarray
        Predicted interaction strengths of shape (D, D, L_pred).
        These are continuous scores (higher = more likely an edge).
    A_true_list : list[np.ndarray]
        Ground-truth lag matrices [A_1, ..., A_L_true], each (D, D),
        containing real weights (pos/neg/zero).
    tau : float
        Threshold for F1@tau (only used if f1_mode="tau").
    binarize_thr : float
        Absolute-weight threshold to consider an edge present in ground truth.
    f1_mode : {"k", "tau"}
        - "k": F1 at top-K where K = #true positive edges (recommended).
        - "tau": F1 after thresholding predictions at tau.

    Returns
    -------
    dict with keys:
      - roc_auc
      - pr_auc
      - f1
      - shd
      - K (number of true positives)
      - n (total edges considered)
    """
    D, D2, L_pred = W_hat.shape
    assert D == D2, "W_hat must be (D, D, L)"
    L_true = len(A_true_list)
    L = min(L_pred, L_true)

    # Flatten ground-truth
    A_true_stack = np.stack(A_true_list[:L], axis=-1)            # (D, D, L)
    y_true = (np.abs(A_true_stack) > binarize_thr).astype(int)   # binary edges
    y_true = y_true.reshape(-1)                                  # (D*D*L,)

    # Flatten predictions
    y_pred = W_hat[:, :, :L].reshape(-1)                         # (D*D*L,)

    n_pos = int(y_true.sum())
    n_all = y_true.size
    if n_pos == 0 or n_pos == n_all:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1": float("nan"),
            "shd": int(n_all),
            "K": n_pos,
            "n": n_all,
        }

    # ROC-AUC / PR-AUC
    roc = float(roc_auc_score(y_true, y_pred))
    pr  = float(average_precision_score(y_true, y_pred))

    # F1
    if f1_mode == "k":
        K = n_pos
        order = np.argsort(y_pred)[::-1]
        yhat = np.zeros_like(y_true)
        yhat[order[:K]] = 1
        f1 = float(f1_score(y_true, yhat))

    elif f1_mode == "tau":
        yhat = (y_pred >= tau).astype(int)
        f1 = float(f1_score(y_true, yhat))
        K = int(yhat.sum())
    else:
        raise ValueError("f1_mode must be 'k' or 'tau'")

    # Structural Hamming Distance (SHD)
    shd = int(np.sum(np.abs(y_true - yhat)))

    return {"roc_auc": roc, "pr_auc": pr, "f1": f1, "shd": shd, "K": int(K), "n": int(n_all)}
