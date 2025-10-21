import numpy as np
from sklearn.neighbors import NearestNeighbors

def _class_indices(y, cls): return np.where(y == cls)[0]

def knn_smote(X, y, k=5, target_counts=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    classes = np.unique(y)
    X_new, y_new = [], []
    for c in classes:
        idx = _class_indices(y, c)
        n = len(idx)
        if target_counts is None or n == 0: continue
        tgt = int(target_counts.get(c, n))
        need = max(0, tgt - n)
        if need == 0 or n < 2: continue
        Xc = X[idx]
        nbrs = NearestNeighbors(n_neighbors=min(k, n)).fit(Xc)
        for _ in range(need):
            i = rng.integers(n)
            nn_idx = nbrs.kneighbors(Xc[i:i+1], return_distance=False)[0]
            j = nn_idx[rng.integers(len(nn_idx))]
            lam = rng.random()
            synth = (1 - lam) * Xc[i] + lam * Xc[j]
            X_new.append(synth); y_new.append(c)
    if X_new:
        X_aug = np.vstack([X, np.vstack(X_new)])
        y_aug = np.concatenate([y, np.array(y_new, dtype=y.dtype)])
        return X_aug, y_aug
    return X, y

def proto_jitter(X, y, std_frac=0.03, max_per_class=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X_new, y_new = [], []
    classes = np.unique(y)
    for c in classes:
        idx = _class_indices(y, c)
        if len(idx) == 0: continue
        Xc = X[idx]
        mu = Xc.mean(axis=0)
        sigma = Xc.std(axis=0) * std_frac
        m = min(max_per_class, len(Xc))
        noise = rng.normal(0.0, 1.0, size=(m, X.shape[1])) * sigma
        X_new.append(mu + noise)
        y_new.append(np.full(m, c, dtype=y.dtype))
    if X_new:
        X_aug = np.vstack([X] + X_new)
        y_aug = np.concatenate([y] + y_new)
        return X_aug, y_aug
    return X, y
