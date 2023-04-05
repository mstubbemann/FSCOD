from multiprocessing import Pool

from ogb.nodeproppred import PygNodePropPredDataset as PD
import pandas as pd
import numpy as np
from torch_geometric.transforms import AddSelfLoops, ToUndirected
from tqdm import tqdm

from functionalities import create_lf, phi_fj

rounds = 1
pools = 3


if __name__ == "__main__":
    ts = 0.9
    pools = 6
    results = []

    data = PD(name="ogbn-arxiv",
              root="data")
    G = data[0]
    G = AddSelfLoops()(G)
    G = ToUndirected()(G)
    X = np.array(G["x"])
    n = len(G["x"])
    d_full = X.shape[1]

    print("First Make Correct Computations")
    print("Select Features Via Pearson Coeficent")
    d_pearson = int(X.shape[1] * ts)
    n_remove = X.shape[1] - d_pearson
    X_cor = X
    coref_matrix = np.abs(np.corrcoef(X_cor, rowvar=False))
    corefs = np.sum(coref_matrix, axis=1)
    # Compute Corefmatrix
    coref_matrix = np.abs(np.corrcoef(X, rowvar=False))
    coref_matrix = coref_matrix - np.diag(np.diag(coref_matrix))
    mults = np.array([1/i for i in range(1, n+1)])

    # Repeatedly move one feature of the two most correlated features
    for j in range(n_remove):
        argmax_coref = np.unravel_index(np.argmax(coref_matrix),
                                        coref_matrix.shape)
        drop_index = argmax_coref[0]
        X_cor = np.delete(X_cor, drop_index, 1)
        coref_value = coref_matrix[argmax_coref[0]
                                   ][argmax_coref[1]]
        coref_matrix = np.delete(coref_matrix, drop_index, 0)
        coref_matrix = np.delete(coref_matrix, drop_index, 1)
        print(j+1, "/", n_remove, " Features removed via Pearson")

    print(X.shape, X_cor.shape)
    print("Compute LFS with correlation coefficent")

    def func(x):
        return create_lf(X_cor, x)

    with Pool(pools) as p:
        LFS_cor = [lf for lf in tqdm(p.imap(func,
                                            range(X_cor.shape[1])),
                                     total=X_cor.shape[1])]

    print("Compute for all Values")

    def func(lf):
        return np.array([phi_fj(lf, j)
                        for j in range(1, n+1)])
    with Pool(pools) as p:
        values_c = np.array([t for t in tqdm(p.imap(func,
                                                    LFS_cor),
                                             total=len(LFS_cor))])
    delta_c = np.mean(mults * values_c, axis=1)
    dim_c = 1 / (delta_c**2)

    print("Now without correlation coefficient")

    def func(x):
        return create_lf(X, x)

    with Pool(pools) as p:
        LFS = [lf for lf in tqdm(p.imap(func,
                                        range(X.shape[1])),
                                 total=X.shape[1])]

    print("Compute for all  Values")

    def func(lf):
        return np.array([phi_fj(lf, j)
                        for j in range(1, n+1)])
    with Pool(pools) as p:
        values_no_c = np.array([t for t in tqdm(p.imap(func,
                                                       LFS),
                                                total=len(LFS))])
    delta_no_c = np.mean(mults * values_no_c, axis=1)
    dim_no_c = 1 / (delta_no_c**2)

    print("##############################")
    print("Now start to make with Samples")
    num_samples = list(np.linspace(0.01, 0.2, num=20))
    for num_s_r in num_samples:
        print("Start with num_s: ", num_s_r)
        num_s = int(num_s_r * n)
        samples = list(n+2 - np.geomspace(n, 2, num_s))
        samples = [int(x) for x in samples]
        samples = sorted(list(set(samples)))
        S = np.array(samples)
        gaps = S[1:] - S[:-1]

        print("With correlation discarding")

        def func(lf):
            return np.array([phi_fj(lf, j)
                            for j in samples])
        with Pool(pools) as p:
            maxes_c = np.array([t for t in tqdm(p.imap(func,
                                                       LFS_cor),
                                                total=len(LFS_cor))])
        # Fill Gaps
        max_deltas_c = []
        min_deltas_c = []
        for f in maxes_c:
            max_d = [0, f[0]]
            for g, v in zip(gaps, f[1:]):
                for _ in range(g):
                    max_d.append(v)
            min_d = [0]
            for g, v in zip(gaps, f):
                for _ in range(g):
                    min_d.append(v)
            min_d.append(f[-1])
            max_deltas_c.append(max_d)
            min_deltas_c.append(min_d)

        min_deltas_c = np.sum(mults * min_deltas_c, axis=1)/n
        max_deltas_c = np.sum(mults * max_deltas_c, axis=1)/n
        min_dim_c = 1/(max_deltas_c ** 2)
        max_dim_c = 1/(min_deltas_c ** 2)
        mean_dim_c = (max_dim_c + min_dim_c)/2

        sorting_c = np.argsort(mean_dim_c)
        dim_sorted_c = dim_c[sorting_c]
        errors_c = np.array([dim_sorted_c[i] > dim_sorted_c[j] for j in range(d_pearson)
                             for i in range(j)], dtype=np.int32)
        errors_c_mean = np.mean(errors_c)

        print("Without correlation discarding")

        with Pool(pools) as p:
            maxes_no_c = np.array([t for t in tqdm(p.imap(func,
                                                          LFS),
                                                   total=len(LFS))])
        # Fill Gaps
        max_deltas_no_c = []
        min_deltas_no_c = []
        for f in maxes_no_c:
            max_d = [0, f[0]]
            for g, v in zip(gaps, f[1:]):
                for _ in range(g):
                    max_d.append(v)
            min_d = [0]
            for g, v in zip(gaps, f):
                for _ in range(g):
                    min_d.append(v)
            min_d.append(f[-1])
            max_deltas_no_c.append(max_d)
            min_deltas_no_c.append(min_d)

        min_deltas_no_c = np.sum(mults * min_deltas_no_c, axis=1)/n
        max_deltas_no_c = np.sum(mults * max_deltas_no_c, axis=1)/n
        min_dim_no_c = 1/(max_deltas_no_c ** 2)
        max_dim_no_c = 1/(min_deltas_no_c ** 2)
        mean_dim_no_c = (max_dim_no_c + min_dim_no_c)/2

        sorting_no_c = np.argsort(mean_dim_no_c)
        dim_sorted_no_c = dim_no_c[sorting_no_c]
        errors_no_c = np.array([dim_sorted_no_c[i] > dim_sorted_no_c[j] for j in range(d_full)
                                for i in range(j)], dtype=np.int32)
        errors_no_c_mean = np.mean(errors_no_c)
        results.append({"num_s": num_s_r,
                        "type": "FDS",
                        "error": errors_no_c_mean})
        results.append({"num_s": num_s_r,
                        "type": "FDSC",
                        "error": errors_c_mean})
        D = pd.DataFrame(results)
        D.to_csv("data/ogb_real_mistakes.csv")
