import numpy as np

def _entropy(p, axis=-1, eps=1e-9):
    p = np.clip(p, eps, 1.0)
    return -np.sum(p * np.log(p), axis=axis)

def _js_to_mean(P, eps=1e-9):
    # P: [T, N, C]
    M = np.clip(P.mean(axis=0), eps, 1.0)
    logM = np.log(M)
    kl = (P * (np.log(np.clip(P, eps, 1.0)) - logM)).sum(axis=2)  # [T,N]
    return kl.mean(axis=0)  # [N]

def _variation_ratio(mean_probs):
    return 1.0 - mean_probs.max(axis=1)

def _top2_margin(mean_probs):
    part = np.partition(-mean_probs, 1, axis=1)
    return (-part[:,0]) - (-part[:,1])

def summary_features_from_tree_probs(P, include_T=False):
    '''
    Build summary features from per-tree probabilities.
    P: list or array of length T, each [N, C].
    Returns: X [N, D] where D = 2*C + 4 (+1 if include_T)
    '''
    if isinstance(P, list):
        P = np.stack(P, axis=0)  # [T, N, C]
    T, N, C = P.shape
    mean_p = P.mean(axis=0)
    var_p  = P.var(axis=0)
    js     = _js_to_mean(P)[:, None]
    vr     = _variation_ratio(mean_p)[:, None]
    ent    = _entropy(mean_p)[:, None]
    mrg    = _top2_margin(mean_p)[:, None]
    feats = [mean_p, var_p, js, vr, ent, mrg]
    if include_T:
        feats.append(np.full((N,1), float(T), dtype=float))
    X = np.concatenate(feats, axis=1)
    return X, mean_p
