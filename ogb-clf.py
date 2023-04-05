import random
import os
from multiprocessing import Pool
import copy

import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid, OGB_MAG
import pytorch_lightning as pl
from torch_geometric.transforms import SIGN, AddSelfLoops, ToUndirected
from ogb.nodeproppred import PygNodePropPredDataset as PD
from tqdm import tqdm

from gnn.ogbn_sign import Net, GraphModule
from functionalities import create_lf, phi_fj
from baselines import rrfs

ts = [0.9, 0.1]


if __name__ == "__main__":
    tasks = ["ogbn-arxiv",
             "ogbn-products",
             "ogbn-mag"]
    ts = [0.9, 0.1]
    pools = 6
    num_samples = 10000
    iterations = 10
    batch_size = 256
    k = 2
    inception_dim = 512
    weight_decay = 0.0001
    lr = 0.001
    epochs = 1000
    gpus = 1
    input_features = [128, 100, 128]
    input_dropouts = [0.1, 0.3, 0]
    dropouts = [0.5, 0.4, 0.5]
    classes = [40, 47, 349]
    result = []
    for task_id, (task,
                  put_dim,
                  input_dropout,
                  dropout,
                  cls) in enumerate(zip(tasks,
                                        input_features,
                                        input_dropouts,
                                        dropouts,
                                        classes)):
        for round in range(iterations):
            seed = (task_id+1) * round * 42
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)
            random.seed(seed)
            rng = np.random.default_rng(seed)
            torch_geometric.seed.seed_everything(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)

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
            elif task in {"Cora", "CiteSeer", "PubMed"}:
                data = Planetoid(name=task, root="data/")
                G = data[0]
            else:
                data = PD(name=task, root="data")
                G = data[0]
            G = AddSelfLoops()(G)
            G = ToUndirected()(G)
            X = np.array(G["x"])
            n = len(G["x"])
            d = X.shape[1]
            d_fs = int(d * ts[1])
            samples = list(n+2 - np.geomspace(n, 2, num_samples))
            samples = [int(x) for x in samples]
            samples = sorted(list(set(samples)))
            S = np.array(samples)
            gaps = S[1:] - S[:-1]

            print("Select Features Via Pearson Coeficent")
            d_pearson = int(X.shape[1] * ts[0])
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
                coref_value = coref_matrix[argmax_coref[0]][argmax_coref[1]]
                coref_matrix = np.delete(coref_matrix, drop_index, 0)
                coref_matrix = np.delete(coref_matrix, drop_index, 1)
                print(j+1, "/", n_remove, " Features removed via Pearson")
            corr_ratio = d_pearson/X.shape[1]

            print("Select Features via Concentration of Measure")
            print("Compute LFS")

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

            multiplications = np.array(
                [1/j for j in range(2, n+1)])
            min_values = multiplications * min_deltas_c
            min_values = np.sum(min_values, axis=1)/n

            max_values = multiplications * max_deltas_c
            max_values = np.sum(max_values, axis=1)/n

            min_dim = 1/(max_values ** 2)
            max_dim = 1/(min_values ** 2)
            mean_dim = (max_dim + min_dim) / 2

            features = np.argpartition(mean_dim,
                                       d_fs)[:d_fs]

            features = np.array(features)
            fs_ratio = len(features)/d
            X_fs = X_cor[:, np.sort(features)]
            print("Founded features:", features)
            print("Now Fs selected features without preprocessing")
            print("Compute LFS")

            def func(i):
                return create_lf(X, i)

            with Pool(pools) as p:
                LFS = [lf for lf in tqdm(p.imap(func, range(X.shape[1])),
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

            min_values = multiplications * min_deltas_no_c
            min_values = np.sum(min_values, axis=1)

            max_values = multiplications * max_deltas_no_c
            max_values = np.sum(max_values, axis=1)

            min_dim = 1/(max_values ** 2)
            max_dim = 1/(min_values ** 2)
            mean_dim = (max_dim + min_dim) / 2

            features = np.argpartition(mean_dim,
                                       d_fs)[:d_fs]

            features = np.array(features)
            X_fs_full = X[:, np.sort(features)]

            print("Check if we instead continued pearson correlation")
            X_cor_2 = X_cor
            d_remove = d_pearson - d_fs
            coref_matrix = np.abs(np.corrcoef(X_cor_2, rowvar=False))
            coref_matrix = coref_matrix - np.diag(np.diag(coref_matrix))
            # Repeatedly move one feature of the two most correlated features
            for j in range(d_remove):
                argmax_coref = np.unravel_index(np.argmax(coref_matrix),
                                                coref_matrix.shape)
                drop_index = argmax_coref[0]
                X_cor_2 = np.delete(X_cor_2, drop_index, 1)
                coref_matrix = np.delete(coref_matrix, drop_index, 0)
                coref_matrix = np.delete(coref_matrix, drop_index, 1)
                print(j+1, "/", d_remove, " Features removed via Pearson")

            print("Check performance if we replace fs with variance")
            variances = -np.var(X, axis=0)
            variances = np.argpartition(variances, d_fs)[:d_fs]
            variances = np.sort(variances)
            X_var = X[:, variances]

            print("Make RRFS Baseline")
            indices = np.sort(rrfs(X, d_fs, coref_value))
            X_rrfs = X[:, indices]

            print("Make random feature selection")
            all_features = X.shape[1]
            random_features = rng.choice(all_features, d_fs, replace=False)
            random_features.sort()
            random_features = np.array(random_features)
            X_random = X[:, random_features]

            curr_result = {"task": task,
                           "round": round,
                           "d": d,
                           "d_pearson": d_pearson,
                           "d_fs": d_fs}
            print("Now start classifications!")
            for name, F in [["Full", X],
                            ["Pearson", X_cor],
                            ["FS", X_fs],
                            ["FSF", X_fs_full],
                            ["PearsonF", X_cor_2],
                            ["Var", X_var],
                            ["RRFS", X_rrfs],
                            ["Random", X_random]]:
                print("Start ", name, " classification")
                H = copy.copy(G)
                H["x"] = torch.tensor(F)
                H = SIGN(k)(H)
                Xs = [H["x"]] + [H["x" + str(i)] for i in range(1, k+1)]
                datamodule = GraphModule(Xs=Xs,
                                         data=data,
                                         G=H,
                                         name=task,
                                         batch_size=batch_size)
                stopper = EarlyStopping(monitor="val_accuracy",
                                        patience=15,
                                        check_on_train_epoch_end=False,
                                        mode="max")
                net = Net(input_features=[X.shape[1] for X in Xs],
                          inception_dim=512,
                          classes=cls,
                          input_dropout=input_dropout,
                          k=k,
                          dropout=dropout,
                          weight_decay=weight_decay,
                          lr=lr)
                trainer = pl.Trainer(deterministic=True,
                                     callbacks=[stopper],
                                     gpus=gpus,
                                     logger=False,
                                     enable_checkpointing=False,
                                     max_epochs=epochs)
                trainer.fit(model=net,
                            datamodule=datamodule)
                trainer.test(model=net,
                             datamodule=datamodule)
                curr_result[name + "_var_accuracy"] = net.val_accuracy
                curr_result[name + "_test_accuracy"] = net.test_accuracy
            result.append(curr_result)
            DF = pd.DataFrame(result)
            DF.to_csv("data/ogb_test.csv")
