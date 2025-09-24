import os, random, numpy as np, torch, json, pickle

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
