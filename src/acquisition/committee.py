import numpy as np

def _map_votes_to_indices(trees, classes, X):
    tv = np.stack([t.predict(X) for t in trees], axis=0)  # (T,N)
    class_to_idx = {c:i for i,c in enumerate(classes)}
    T, N = tv.shape; C = len(classes)
    tv_idx = np.empty((T, N), dtype=np.int64)
    for r in range(T):
        for c in range(N):
            v = tv[r, c]
            if v in class_to_idx:
                tv_idx[r, c] = class_to_idx[v]; continue
            try:
                iv = int(float(v))
                if 0 <= iv < C: tv_idx[r, c] = iv; continue
            except Exception: pass
            vs = str(v)
            if vs in class_to_idx:
                tv_idx[r, c] = class_to_idx[vs]; continue
            raise KeyError(f'Unmappable committee vote: {v} (classes={list(classes)})')
    return tv_idx

def vote_entropy_from_trees(trees, classes, X):
    tv_idx = _map_votes_to_indices(trees, classes, X)
    N = tv_idx.shape[1]; C = len(classes)
    ve = np.zeros(N, dtype=float)
    for i in range(N):
        counts = np.bincount(tv_idx[:, i], minlength=C).astype(float)
        p = counts / (counts.sum() + 1e-9); p = p[p>0]
        ve[i] = -np.sum(p * np.log(p))
    return ve
