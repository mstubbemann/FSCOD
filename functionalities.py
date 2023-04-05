import numpy as np


def create_lf(X, i):
    return np.sort(X[:, i])


def phi_fj(lf,
           j):
    if j == 1:
        return 0
    else:
        return np.min(lf[j-1:]-lf[:1-j])


def create_lfs(X):
    return np.array([create_lf(X, i)
                     for i in range(X.shape[1])])
