import numpy as np
from sklearn.metrics import recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_all(y_true, y_pred, y_proba, classes):
    labels = list(classes)
    bal_acc = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1m = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    try:
        y_true_arr = np.asarray(y_true)
        y_bin = np.zeros((len(y_true_arr), len(labels)), dtype=int)
        for j, c in enumerate(labels):
            y_bin[:, j] = (y_true_arr == c).astype(int)
        auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {'bal_acc': bal_acc, 'f1_macro': f1m, 'auc_ovr': auc, 'cm': cm}
