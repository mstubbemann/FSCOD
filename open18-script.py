import warnings

import pandas as pd
import numpy as np
import openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

from baselines import spec, rrfs
from functionalities import create_lfs, phi_fj

C = 1
ts = [0.9, 0.1]
rounds = 10


if __name__ == "__main__":
    # Get the benchmark suite
    result = []
    suite = openml.study.get_suite("OpenML-CC18")

    for i, task_id in enumerate(suite.tasks):  # iterate over all tasks
        breaked = False
        task_result = []
        for round in range(rounds):
            if breaked:
                break
            rng = np.random.default_rng(i + round + 42)
            print("Preprocess Data")
            task = openml.tasks.get_task(task_id)  # download the OpenML task
            X, y = task.get_X_and_y()  # get the data

            # Skip if NaN
            if np.isnan(X).any():
                breaked = True
                break
            # Skip if  binary
            if ((X == 0) | (X == 1)).all():
                breaked = True
                break

            split = task.download_split()
            split = split.split[0]
            all_features = X.shape[1]
            if all_features < 50:
                breaked = True
                break
            for s in split:
                c_s = split[s][0][0]
                X_c = X[c_s]
                y_c = y[c_s]
                test_s = split[s][0][1]
                X_test = X[test_s]
                y_test = y[test_s]
                d_full = X.shape[1]
                d_fs = int(d_full * ts[1])
                n = X_c.shape[0]
                print("Test on Full Feature Set")
                warnings.filterwarnings("error",
                                        category=ConvergenceWarning)
                try:
                    LR = LogisticRegression(
                        C=C, max_iter=1000, random_state=i + round)
                    LR.fit(X_c, y_c)
                    y_pred = LR.predict(X_test)
                    full_ac = accuracy_score(y_test, y_pred)
                    breaked = False
                except ConvergenceWarning:
                    breaked = True
                    break

                warnings.filterwarnings("default",
                                        category=ConvergenceWarning)
                print("Select Features Via Pearson Coeficent")
                n_select = int(d_full * ts[0])
                n_remove = X.shape[1] - n_select
                X_cor = X_c
                X_test_cor = X_test
                # Compute Corefmatrix
                coref_matrix = np.abs(np.corrcoef(X_c, rowvar=False))
                coref_matrix = coref_matrix - np.diag(np.diag(coref_matrix))
                # Repeatedly move one feature of the two most correlated features
                for j in range(n_remove):
                    argmax_coref = np.unravel_index(np.argmax(coref_matrix),
                                                    coref_matrix.shape)
                    coref_value = coref_matrix[argmax_coref[0]
                                               ][argmax_coref[1]]
                    drop_index = argmax_coref[0]
                    X_cor = np.delete(X_cor, drop_index, 1)
                    X_test_cor = np.delete(X_test_cor, drop_index, 1)
                    coref_matrix = np.delete(coref_matrix, drop_index, 0)
                    coref_matrix = np.delete(coref_matrix, drop_index, 1)
                    print(j+1, "/", n_remove, " Features removed via Pearson")

                # Check ratio after feature selection via Pearson
                d_pearson = X_cor.shape[1]
                pearson_ratio = d_pearson / d_full

                print("Make LR with uncorrelated features")
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_cor, y_c)
                y_pred = LR.predict(X_test_cor)
                corr_ac = accuracy_score(y_test, y_pred)

                print("Start Feature Selection!")
                print("Create LFS")
                LFS = create_lfs(X_cor)
                print("Start selecting features")
                print("Compute Maximums")

                def func(lf):
                    return np.array([phi_fj(lf, j)
                                    for j in range(2, X_cor.shape[0] + 1)])
                maxes_c = np.array([func(lf) for lf in LFS])
                multiplications = np.array([1/j
                                           for j in range(2, X_cor.shape[0]+1)])
                values = multiplications * maxes_c
                values = np.sum(values, axis=1)/X_cor.shape[0]
                features = np.argpartition(1 / (values**2), d_fs)[:d_fs]

                print("Now start training with selected features")
                features = list(features)
                features.sort()
                features = np.array(features)
                fs_ratio = len(features)/d_full
                X_cor_fs = X_cor[:, features]
                X_test_fs = X_test_cor[:, features]
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_cor_fs, y_c)
                y_pred = LR.predict(X_test_fs)
                fs_ac = accuracy_score(y_test, y_pred)

                print("Now start training with only ",
                      "selected features without preprocessing")
                print("Create LFS")
                LFS = create_lfs(X_c)
                print("Start selecting features")
                print("Compute Maximums")

                def func(lf):
                    return np.array([phi_fj(lf, j)
                                    for j in range(2, X_c.shape[0] + 1)])
                maxes_c = np.array([func(lf) for lf in LFS])
                multiplications = np.array([1/j
                                           for j in range(2, X_c.shape[0]+1)])
                values = multiplications * maxes_c
                values = np.sum(values, axis=1)/X_c.shape[0]
                features = np.argpartition(1 / (values**2), d_fs)[:d_fs]

                print("Now start training with selected features")
                features = list(features)
                features.sort()
                features = np.array(features)
                X_fs = X_c[:, features]
                X_test_fs = X_test[:, features]
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_fs, y_c)
                y_pred = LR.predict(X_test_fs)
                fs_ac_only = accuracy_score(y_test, y_pred)

                print("Check performance if we replace fs with variance")
                variances = -np.var(X_c, axis=0)
                variances = np.argpartition(variances, d_fs)[:d_fs]
                variances = np.sort(variances)
                X_var = X_c[:, variances]
                X_test_var = X_test[:, variances]

                print("Make LR with variance checking")
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_var, y_c)
                y_pred = LR.predict(X_test_var)
                var_ac = accuracy_score(y_test, y_pred)

                print("Check if we instead continued pearson correlation")

                d_remove = d_pearson - d_fs
                coref_matrix = np.abs(np.corrcoef(X_cor, rowvar=False))
                coref_matrix = coref_matrix - np.diag(np.diag(coref_matrix))
                # Repeatedly move one feature of the two most correlated features
                for j in range(d_remove):
                    argmax_coref = np.unravel_index(np.argmax(coref_matrix),
                                                    coref_matrix.shape)
                    drop_index = argmax_coref[0]
                    X_cor = np.delete(X_cor, drop_index, 1)
                    X_test_cor = np.delete(X_test_cor, drop_index, 1)
                    coref_matrix = np.delete(coref_matrix, drop_index, 0)
                    coref_matrix = np.delete(coref_matrix, drop_index, 1)
                    print(j+1, "/", d_remove, " Features removed via Pearson")

                print("Make Second LR with uncorrelated features")
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_cor, y_c)
                y_pred = LR.predict(X_test_cor)
                corr_ac_1 = accuracy_score(y_test, y_pred)

                print("Make RRFS Baseline")
                indices = np.sort(rrfs(X_c, d_fs, coref_value))
                X_rrfs = X_c[:, indices]
                X_rrfs_test = X_test[:, indices]
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_rrfs, y_c)
                y_pred = LR.predict(X_rrfs_test)
                rrfs_ac = accuracy_score(y_test, y_pred)

                print("Make Spec Baseline")
                indices = np.sort(spec(X_c, d_fs))
                X_spec = X_c[:, indices]
                X_spec_test = X_test[:, indices]
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_spec, y_c)
                y_pred = LR.predict(X_spec_test)
                spec_ac = accuracy_score(y_test, y_pred)

                print("Make Random Baseline")
                random_features = rng.choice(all_features,
                                             d_fs,
                                             replace=False)
                random_features.sort()
                random_features = np.array(random_features)
                X_c_random = X_c[:, random_features]
                X_test_random = X_test[:, random_features]
                print("Try LR!")
                LR = LogisticRegression(
                    C=C, max_iter=1000, random_state=i + round)
                LR.fit(X_c_random, y_c)
                y_pred = LR.predict(X_test_random)
                random_ac = accuracy_score(y_test, y_pred)

                task_result.append({"round": round,
                                    "task": task_id,
                                    "split": s,
                                    "full_acc": full_ac,
                                    "fs_acc": fs_ac,
                                    "fs_ac_only": fs_ac_only,
                                    "random_acc": random_ac,
                                    "corr_ac": corr_ac,
                                    "corr_ac_1": corr_ac_1,
                                    "var_ac": var_ac,
                                    "rrfs_ac": rrfs_ac,
                                    "spec_ac": spec_ac,
                                    "d_full": d_full,
                                    "d_fs": d_fs,
                                    "d_corr": d_pearson,
                                    "corr_ratio": pearson_ratio,
                                    "fs_ratio": fs_ratio,
                                    "full_ratio": fs_ratio,
                                    "C": C})

            print("Task ", i + 1, "of ", len(suite.tasks), " tasks.")
            print("round ", round + 1, "of ", rounds, " rounds.")
        if not breaked:
            result.extend(task_result)
            DF = pd.DataFrame(result)
            DF.to_csv("data/open18.csv")
        else:
            continue
