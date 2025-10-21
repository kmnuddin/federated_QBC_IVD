import os, numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from src.utils.seed import set_seed
from src.selection.k_coreset import kcenter_greedy
from src.loop.client_loop import initialize_client, fit_rf_with_aug, acquire_by_qbc_with_meta, meta_pseudo_labels
from src.eval.logging import CSVLogger, JSONLLogger
from src.embeddings.mri_core_runtime import MRICoreEmbedder
from src.meta.fed_ridge import FedRidgeMeta

class EmbeddingTable:
    def __init__(self, csv_path, id_col, label_col, image_col='axial_path', emb_col=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.id_col, self.label_col, self.image_col = id_col, label_col, image_col
        self.emb_col = emb_col if (emb_col and emb_col in self.df.columns) else None
    def image_paths(self): return self.df[self.image_col].astype(str).values
    def labels(self): return self.df[self.label_col].astype(str).values

def run_sim(cfg):
    set_seed(cfg.get('seed', 42))
    os.makedirs(cfg['logging']['out_dir'], exist_ok=True)
    csv_logger = CSVLogger(os.path.join(cfg['logging']['out_dir'], 'metrics.csv'))
    jsonl_logger = JSONLLogger(os.path.join(cfg['logging']['out_dir'], 'events.jsonl'))

    A_tab = EmbeddingTable(cfg['data']['rsna_csv'], cfg['data']['id_col'], cfg['data']['label_col'], cfg['data']['image_col'], cfg['data'].get('emb_col'))
    B_tab = EmbeddingTable(cfg['data']['mendeley_csv'], cfg['data']['id_col'], cfg['data']['label_col'], cfg['data']['image_col'], cfg['data'].get('emb_col'))
    all_classes = np.array(sorted(np.unique(np.concatenate([A_tab.labels(), B_tab.labels()])).tolist()), dtype=str)

    print("[SIM] Initializing MRI-CORE...")
    embedder = MRICoreEmbedder(cfg)
    print("[SIM] MRI-CORE ready.]")

    def make_client(name, tab):
        df = tab.df if hasattr(tab, 'df') else None
        img_paths = tab.image_paths()
        y = tab.labels()
        seed_k = cfg.get('seed_k', 200)
        rng = np.random.default_rng(cfg.get('seed', 42))
        base_embs = np.vstack([embedder.embed_original_cached(p) for p in img_paths]).astype(np.float32)
        if seed_k >= len(y):
            labeled_mask = np.ones(len(y), dtype=bool)
        else:
            start = int(rng.integers(len(y)))
            seed_idx = kcenter_greedy(base_embs, seed_k, start_idx=start)
            labeled_mask = np.zeros(len(y), dtype=bool); labeled_mask[seed_idx] = True
        return initialize_client(name, df, y, labeled_mask, embedder, img_paths, all_classes)

    A = make_client("RSNA", A_tab)
    B = make_client("MENDELEY", B_tab)

    meta_cfg = cfg.get('meta', {})
    meta_enabled = bool(meta_cfg.get('enabled', True))
    meta_refresh = int(meta_cfg.get('refresh_every', 1))
    feature_include_T = bool(meta_cfg.get('include_T', False))
    lambda_reg = float(meta_cfg.get('lambda', 1e-2))
    oof_folds = int(meta_cfg.get('oof_folds', 5))

    A.meta = FedRidgeMeta(all_classes, lambda_reg=lambda_reg, feature_include_T=feature_include_T, oof_folds=oof_folds)
    B.meta = FedRidgeMeta(all_classes, lambda_reg=lambda_reg, feature_include_T=feature_include_T, oof_folds=oof_folds)

    R = cfg['rounds']; BATCH = cfg['al']['batch_B']; per_class_min = int(cfg['al'].get('per_class_min', 1))
    blend_alpha = float(cfg.get('qbc', {}).get('blend_alpha', 0.0))

    print(f"[SIM] Starting simulation for {R} rounds...")
    for r in range(1, R+1):
        print(f"\n===== ROUND {r}/{R} =====")
        for client in [A, B]:
            before = int(client.pool_mask_labeled.sum())
            print(f"[ROUND {r}] Client={client.name} | labeled_before={before}")
            orig_n, aug_n = fit_rf_with_aug(client, cfg['rf'], cfg, r, progress=True)

        if meta_enabled and (r % meta_refresh == 0):
            Sxx_global = None; Sxy_global = None
            for client in [A, B]:
                L_idx = np.where(client.pool_mask_labeled)[0]
                if len(L_idx) == 0: continue
                X_L = client.X_pool_base[L_idx]
                y_L = client.y_pool[L_idx].astype(str)
                Sxx, Sxy = client.meta.client_oof_stats(X_L, y_L, client.committee_trees, client.scaler)
                if Sxx_global is None:
                    Sxx_global, Sxy_global = Sxx, Sxy
                else:
                    Sxx_global += Sxx; Sxy_global += Sxy
            if Sxx_global is not None:
                W = A.meta.server_solve(Sxx_global, Sxy_global)
                B.meta.W = W.copy(); A.meta.W = W.copy()
                print(f"[ROUND {r}] Meta ridge solved and broadcast.")

        for client in [A, B]:
            pick, picked_per_class, score_vec, meta_p = acquire_by_qbc_with_meta(
                client, BATCH, per_class_min, use_meta_entropy=True, blend_alpha=blend_alpha
            )
            y_hat, conf = meta_pseudo_labels(client, pick)
            y_true = client.y_pool[pick].astype(str) if len(pick) else np.array([], dtype=str)
            pseudo_acc = float((y_hat==y_true).mean()) if len(pick) else float('nan')
            client.pool_mask_labeled[pick] = True
            after = int(client.pool_mask_labeled.sum())
            U = np.where(~client.pool_mask_labeled)[0]
            if len(U):
                Zt = client.scaler.transform(client.X_pool_base[U])
                y_true_rest = client.y_pool[U].astype(str)
                y_pred = client.rf._clf.predict(Zt).astype(str)
                y_proba = client.rf._clf.predict_proba(Zt)
                bal_acc = float(balanced_accuracy_score(y_true_rest, y_pred))
                f1m = float(f1_score(y_true_rest, y_pred, average='macro'))
                try:
                    auc_ovr = float(roc_auc_score(y_true_rest, y_proba, multi_class='ovr'))
                except Exception:
                    auc_ovr = float('nan')
            else:
                bal_acc=f1m=auc_ovr=float('nan')
            print(f"[ROUND {r}] Client={client.name} | acquired={len(pick)} | pseudo_acc_meta={pseudo_acc:.3f}")
            print(f"[ROUND {r}] Client={client.name} | labeled_after={after} | bal_acc={bal_acc:.3f} | f1_macro={f1m:.3f} | auc_ovr={auc_ovr if not np.isnan(auc_ovr) else 'nan'}")
            ve_med = float(np.median(score_vec)) if len(score_vec) else float('nan')
            csv_logger.log(r, client.name, after, ve_med, bal_acc, f1m, None if np.isnan(auc_ovr) else auc_ovr, pseudo_acc, int(before), int(after - before), len(client.committee_trees or []), None)
            jsonl_logger.log({
                "round": r, "client": client.name,
                "picked": int(len(pick)),
                "picked_per_class": {str(k): int(v) for k,v in picked_per_class.items()},
                "pseudo_acc_meta": pseudo_acc,
                "orig_train_n": int(before), "aug_train_n": int(after - before),
                "metrics_rest": {"bal_acc": bal_acc, "f1_macro": f1m, "auc_ovr": None if np.isnan(auc_ovr) else auc_ovr}
            })
    print("Simulation finished. Logs at:", cfg['logging']['out_dir'])
