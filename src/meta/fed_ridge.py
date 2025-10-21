import numpy as np
from .summary_features import summary_features_from_tree_probs
from sklearn.model_selection import StratifiedKFold
from src.utils.math_utils import one_hot

class FedRidgeMeta:
    def __init__(self, classes, lambda_reg=1e-2, feature_include_T=False, oof_folds=5):
        self.classes = np.array(classes, dtype=str)
        self.lambda_reg = float(lambda_reg)
        self.feature_include_T = bool(feature_include_T)
        self.oof_folds = int(oof_folds)
        self.W = None

    def client_oof_stats(self, X_base, y_str, tree_list, scaler):
        L = X_base.shape[0]
        classes = self.classes
        Y = one_hot(y_str.astype(str), classes)
        n_splits = min(self.oof_folds, max(2, np.unique(y_str).size))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        D = 2*len(classes) + 4 + (1 if self.feature_include_T else 0)
        X_feat = np.zeros((L, D), dtype=float)
        for tr_idx, te_idx in skf.split(X_base, y_str):
            Z_te = scaler.transform(X_base[te_idx])
            probs = [t.predict_proba(Z_te) for t in tree_list]
            aligned = []
            for P, t in zip(probs, tree_list):
                t_classes = getattr(t, 'classes_', classes)
                A = np.zeros((P.shape[0], len(classes)), dtype=float)
                idx_map = {c:i for i,c in enumerate(t_classes)}
                for j,c in enumerate(classes):
                    i = idx_map.get(c, None)
                    if i is not None:
                        A[:, j] = P[:, i]
                A = (A + 1e-6); A /= A.sum(axis=1, keepdims=True)
                aligned.append(A)
            Xsum, _ = summary_features_from_tree_probs(aligned, include_T=self.feature_include_T)
            X_feat[te_idx] = Xsum
        Sxx = X_feat.T @ X_feat
        Sxy = X_feat.T @ Y
        return Sxx, Sxy

    def server_solve(self, Sxx_global, Sxy_global):
        D = Sxx_global.shape[0]
        A = Sxx_global + self.lambda_reg * np.eye(D, dtype=float)
        self.W = np.linalg.solve(A, Sxy_global)
        return self.W

    def predict_meta_proba_from_tree_probs(self, probs_list):
        X, mean_p = summary_features_from_tree_probs(probs_list, include_T=self.feature_include_T)
        if self.W is None:
            return mean_p, mean_p
        scores = X @ self.W
        z = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(z)
        meta_p = e / e.sum(axis=1, keepdims=True)
        return meta_p, mean_p
