import numpy as np
def class_medoids(X, y, max_k=5):
    protos = {}
    for c in np.unique(y):
        idx = np.where(y==c)[0]
        if len(idx)==0: continue
        Xc = X[idx]
        D = np.sqrt(((Xc[:,None,:]-Xc[None,:,:])**2).sum(-1))
        sums = D.sum(axis=1)
        order = np.argsort(sums)[:max_k]
        protos[c] = Xc[order]
    return protos
