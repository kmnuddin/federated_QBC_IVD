import numpy as np
from sklearn.metrics import pairwise_distances
from typing import Optional

def kcenter_greedy(X, k, start_idx: Optional[int]=None, metric='euclidean', rng: Optional[np.random.Generator]=None):
    n = X.shape[0]
    if k >= n:
        return list(range(n))
    if start_idx is None:
        start_idx = int(rng.integers(n)) if rng is not None else int(np.random.randint(n))
    centers = [start_idx]
    min_dist = pairwise_distances(X, X[start_idx:start_idx+1], metric=metric).ravel()
    for _ in range(1, k):
        next_idx = int(np.argmax(min_dist))
        centers.append(next_idx)
        d = pairwise_distances(X, X[next_idx:next_idx+1], metric=metric).ravel()
        min_dist = np.minimum(min_dist, d)
    return centers
