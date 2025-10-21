from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from src.models.rf import RFModel

@dataclass
class ClientState:
    name: str
    X_pool_base: np.ndarray | None
    y_pool: np.ndarray
    pool_mask_labeled: np.ndarray
    scaler: StandardScaler | None = None
    rf: RFModel | None = None
    committee_trees: list | None = None
    classes: np.ndarray | None = None
    embedder: object | None = None
    img_paths: np.ndarray | None = None
    _pool_df: object | None = None
    meta: object | None = None

def ensure_base_embeddings(state: ClientState):
    if state.X_pool_base is not None: return
    X = [state.embedder.embed_original_cached(p) for p in state.img_paths]
    state.X_pool_base = np.vstack(X).astype(np.float32)

def initialize_client(name, df_pool, y_pool, labeled_mask, embedder, img_paths, classes):
    st = ClientState(
        name=name, X_pool_base=None, y_pool=y_pool.astype(str),
        pool_mask_labeled=labeled_mask.astype(bool),
        embedder=embedder, img_paths=img_paths, _pool_df=df_pool
    )
    st.classes = np.array(sorted(classes.tolist()), dtype=str)
    return st

def fit_rf_with_aug(state: ClientState, rf_cfg, cfg, round_id: int, progress=True):
    ensure_base_embeddings(state)
    L = np.where(state.pool_mask_labeled)[0]
    if len(L)==0: return 0,0
    n_aug = int(cfg.get('augment', {}).get('train_n_per_sample', 2))
    feats, labels = [], []
    id_col = cfg['data']['id_col']
    iterator = list(L)
    iterator = tqdm(iterator, desc=f"[{state.name}] Round {round_id} RF train (aug+embed)", unit="sample") if progress else iterator
    for idx in iterator:
        img_path = state.img_paths[idx]
        key = str(state._pool_df.iloc[idx][id_col]) if state._pool_df is not None else str(idx)
        E = state.embedder.embed_with_aug(img_path, n_aug=n_aug, key=key)
        feats.append(E); labels.append(np.full(E.shape[0], state.y_pool[idx], dtype=state.y_pool.dtype))
    X_L = np.vstack(feats); y_L = np.concatenate(labels).astype(str)
    orig_n = len(L); aug_n = int(X_L.shape[0] - orig_n)
    state.scaler = StandardScaler()
    Z_L = state.scaler.fit_transform(X_L)
    state.rf = RFModel(**rf_cfg).fit(Z_L, y_L)
    state.committee_trees = list(state.rf.trees)
    return orig_n, aug_n

def _align_proba(P, t_classes, classes):
    idx_map = {c:i for i,c in enumerate(t_classes)}
    A = np.zeros((P.shape[0], len(classes)), dtype=float)
    for j,c in enumerate(classes):
        i = idx_map.get(c, None)
        if i is not None: A[:, j] = P[:, i]
    A = (A + 1e-6); A /= A.sum(axis=1, keepdims=True)
    return A

def committee_probs(state: ClientState, Z):
    probs = []
    classes = state.rf._clf.classes_
    for t in state.committee_trees or []:
        P = t.predict_proba(Z)
        tc = getattr(t, 'classes_', classes)
        probs.append(_align_proba(P, tc, classes))
    if not probs:
        P = state.rf._clf.predict_proba(Z)
        probs = [P]
    return probs

def committee_js_score(state: ClientState, Z):
    probs = committee_probs(state, Z)
    P = np.stack(probs, axis=0)
    M = np.clip(P.mean(axis=0), 1e-9, 1.0)
    logM = np.log(M)
    kl = (P * (np.log(np.clip(P,1e-9,1.0)) - logM)).sum(axis=2)
    return kl.mean(axis=0)

def acquire_by_qbc_with_meta(state: ClientState, B, per_class_min: int, use_meta_entropy=True, blend_alpha=0.0):
    ensure_base_embeddings(state)
    U = np.where(~state.pool_mask_labeled)[0]
    if len(U)==0: return np.array([], dtype=int), {}, np.array([]), np.array([])
    ZU = state.scaler.transform(state.X_pool_base[U])
    js = committee_js_score(state, ZU)
    mean_p = np.mean(np.stack(committee_probs(state, ZU), axis=0), axis=0)
    classes = state.rf._clf.classes_.astype(str)
    y_hat = classes[mean_p.argmax(1)]
    if state.meta is not None and state.meta.W is not None:
        meta_p, _ = state.meta.predict_meta_proba_from_tree_probs(committee_probs(state, ZU))
        meta_entropy = -(np.clip(meta_p,1e-9,1.0)*np.log(np.clip(meta_p,1e-9,1.0))).sum(axis=1)
        score = js + float(blend_alpha) * meta_entropy
    else:
        meta_p = None
        score = js
    order = np.argsort(-score)
    picked, per_class_counts = [], {c:0 for c in state.classes}
    for c in state.classes:
        need = per_class_min
        for idx in order:
            if U[idx] in picked: continue
            if y_hat[idx] == c and per_class_counts[c] < need:
                picked.append(U[idx]); per_class_counts[c]+=1
                if len(picked)>=B: break
        if len(picked)>=B: break
    if len(picked)<B:
        for idx in order:
            if U[idx] in picked: continue
            picked.append(U[idx])
            if len(picked)>=B: break
    picked = np.array(picked[:B], dtype=int)
    return picked, per_class_counts, score, (meta_p if meta_p is not None else np.array([]))

def meta_pseudo_labels(state: ClientState, idxs):
    if len(idxs)==0 or state.meta is None: 
        return np.array([], dtype=str), np.array([])
    Z = state.scaler.transform(state.X_pool_base[idxs])
    probs = committee_probs(state, Z)
    meta_p, _ = state.meta.predict_meta_proba_from_tree_probs(probs)
    classes = state.rf._clf.classes_.astype(str)
    y_idx = meta_p.argmax(1); conf = meta_p.max(1)
    y_hat = classes[y_idx]
    return y_hat.astype(str), conf
