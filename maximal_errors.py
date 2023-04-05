from multiprocessing import Pool

from ogb.nodeproppred import PygNodePropPredDataset as PD
import pandas as pd
import numpy as np
import torch
from torch_geometric.datasets import OGB_MAG
from torch_geometric.transforms import AddSelfLoops, ToUndirected
from tqdm import tqdm

from functionalities import create_lf, phi_fj

C = 1
ts = [0.9, 0.1]
rounds = 1
pools = 10

if __name__ == "__main__":
    tasks = ["ogbn-arxiv",
             "ogbn-products",
             "ogbn-mag"]
    ts = 0.9
    pools = 6
    num_samples = list(np.linspace(0.01, 0.2, num=20))
    k = 2
    result = []
    for task_id, task in enumerate(tasks):
        for num_s in num_samples:
            print("##################")
            print("##################")
            print("Start with: ", task)
            print("Compute Graph Data")
            if task == "ogbn-mag":
                data = OGB_MAG(root="data",
                               preprocess="metapath2vec")
                H = data[0]
                G = H.to_homogeneous()
                G["train_mask"] = torch.where(H["paper"]["train_mask"])[0]
                G["val_mask"] = torch.where(H["paper"]["val_mask"])[0]
                G["test_mask"] = torch.where(H["paper"]["test_mask"])[0]
                G["y"] = H["paper"]["y"]
            else:
                data = PD(name=task, root="data")
                G = data[0]
            G = AddSelfLoops()(G)
            G = ToUndirected()(G)
            X = np.array(G["x"])
            n = len(G["x"])
            num_s = int(num_s * n)
            samples = list(n+2 - np.geomspace(n, 2, num_s))
            samples = [int(x) for x in samples]
            samples = sorted(list(set(samples)))
            S = np.array(samples)
            gaps = S[1:] - S[:-1]
            d_full = X.shape[1]

            print("Select Features Via Pearson Coeficent")
            d_pearson = int(X.shape[1] * ts)
            n_remove = X.shape[1] - d_pearson
            X_cor = X
            coref_matrix = np.abs(np.corrcoef(X_cor, rowvar=False))
            corefs = np.sum(coref_matrix, axis=1)
            # Compute Corefmatrix
            coref_matrix = np.abs(np.corrcoef(X, rowvar=False))
            coref_matrix = coref_matrix - np.diag(np.diag(coref_matrix))

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

            print("Make Computiations with Preprocessed Feature Discarding")

            def func(x):
                return create_lf(X_cor, x)

            with Pool(pools) as p:
                LFS = [lf for lf in tqdm(p.imap(func,
                                                range(X_cor.shape[1])),
                                         total=X_cor.shape[1])]

            print("Compute for Support Values")

            def func(lf):
                return np.array([phi_fj(lf, j)
                                for j in samples])
            with Pool(pools) as p:
                maxes_c = np.array([t for t in tqdm(p.imap(func,
                                                           LFS),
                                                    total=len(LFS))])

            # Fill Gaps
            max_deltas_c = []
            min_deltas_c = []
            for f in maxes_c:
                max_d = [f[0]]
                for g, v in zip(gaps, f[1:]):
                    for _ in range(g):
                        max_d.append(v)
                min_d = []
                for g, v in zip(gaps, f):
                    for _ in range(g):
                        min_d.append(v)
                min_d.append(f[-1])
                max_deltas_c.append(max_d)
                min_deltas_c.append(min_d)

            print("Make Computations without Preprocessed Feature Discarding")

            def func(x):
                return create_lf(X, x)

            with Pool(pools) as p:
                LFS = [lf for lf in tqdm(p.imap(func,
                                                range(X.shape[1])),
                                         total=X.shape[1])]

            print("Compute for Support Values")

            def func(lf):
                return np.array([phi_fj(lf, j)
                                for j in samples])
            with Pool(pools) as p:
                maxes_no_c = np.array([t for t in tqdm(p.imap(func,
                                                              LFS),
                                                       total=len(LFS))])

            # Fill Gaps
            max_deltas_no_c = []
            min_deltas_no_c = []
            for f in maxes_no_c:
                max_d = [f[0]]
                for g, v in zip(gaps, f[1:]):
                    for _ in range(g):
                        max_d.append(v)
                min_d = []
                for g, v in zip(gaps, f):
                    for _ in range(g):
                        min_d.append(v)
                min_d.append(f[-1])
                max_deltas_no_c.append(max_d)
                min_deltas_no_c.append(min_d)

            mults = np.array([1/j for j in range(2, n+1)])
            for min_delta, max_delta, type in ([min_deltas_no_c,
                                                max_deltas_no_c,
                                                "FDS"],
                                               [min_deltas_c,
                                                max_deltas_c,
                                                "FDSC"]):
                min_delta = np.sum(mults * min_delta, axis=1)/n
                max_delta = np.sum(mults * max_delta, axis=1)/n
                min_dim = 1/(max_delta ** 2)
                max_dim = 1/(min_delta ** 2)
                mean_dim = (max_dim + min_dim)/2
                indices = np.argsort(mean_dim)
                min_dim = min_dim[indices]
                max_dim = max_dim[indices]
                if type == "FDS":
                    dim = d_full
                else:
                    dim = d_pearson
                mistakes = [max_dim[i] > min_dim[j] for j in range(dim)
                            for i in range(j)]
                mistakes = np.array(mistakes, dtype=np.int32)
                mistakes = np.mean(mistakes)
                result.append({"task": task,
                               "type": type,
                               "mistakes": mistakes,
                               "num_samples": num_s})
                D = pd.DataFrame(result)
                D.to_csv("data/ogb_mistakes.csv")
