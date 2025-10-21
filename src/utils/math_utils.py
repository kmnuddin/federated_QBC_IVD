import numpy as np

def softmax(logits, axis=-1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)

def one_hot(y_str, classes):
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    Y = np.zeros((len(y_str), len(classes)), dtype=float)
    for i, c in enumerate(y_str):
        Y[i, cls_to_idx[c]] = 1.0
    return Y
