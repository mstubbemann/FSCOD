import numpy as np


def spec(X, n_features=None):
    W = np.array([[np.exp(-(np.linalg.norm(A - B)**2 / 2))
                   for B in X]
                  for A in X])
    D = np.diag(np.sum(W, axis=1))
    L = D-W
    D_Left = np.diag(D).reshape((D.shape[0], 1))**(-0.5)
    D_right = D_Left.flatten()
    Laplace = D_Left * L * D_right
    norm = 1 / np.linalg.norm(X, axis=0)
    X_normalized = X * norm
    Right_Part = np.matmul(Laplace, X_normalized)
    scores = np.sum(X_normalized * Right_Part, axis=0)
    if n_features is not None:
        indices = np.argpartition(scores, n_features)[:n_features]
        return indices
    else:
        return scores


def rrfs(X,
         k,
         threshold=0.9):
    variances = np.var(X, axis=0)
    f_indices = np.argsort(-variances)
    resulting_order = [f_indices[0]]
    prev_i = f_indices[0]
    for index in f_indices[1:]:
        if len(resulting_order) >= k:
            return resulting_order
        sim = np.abs(np.corrcoef(X[:, prev_i], X[:, index], rowvar=False)[0][1])
        if sim < threshold:
            resulting_order.append(index)
    return resulting_order
