#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import sys
import fastdtw
import numpy as np
from itertools import product

from sklearn.decomposition import PCA
from sklearn.model_selection import ParameterGrid

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.sparse.linalg import eigsh

from .base import DistilBaseModel
from .metrics import metrics, regression_metrics, classification_metrics
from .forest import ForestCV
from .helpers import tiebreaking_vote

from ..utils import parmap

# --
# Helpers


def _fastdtw_metric(a, b):
    return fastdtw.fastdtw(a, b)[0]


def distance_matrix(X, metric):
    global _dtw_dist_row_all

    if metric != "dtw":
        dist = squareform(pdist(X, metric=metric))
    else:

        def _dtw_dist_row_all(t):
            return [_fastdtw_metric(t, tt) for tt in X]

        dist = np.vstack(parmap(_dtw_dist_row_all, list(X), verbose=X.shape[0] > 1000))

    return dist


def whiten(X_train, X_test):
    pca = PCA(whiten=True).fit(np.vstack([X_train, X_test]))
    return pca.transform(X_train), pca.transform(X_test)


def precomputed_knn1_cv(train_dist, test_dist, y_train, target_metric):
    np.fill_diagonal(train_dist, np.inf)  # don't predict self

    pred_cv = y_train[train_dist.argmin(axis=-1)]  # !! Should break ties better
    fitness_cv = metrics[target_metric](y_train, pred_cv)

    # !! Only for debugging
    pred_test = y_train[test_dist.argmin(axis=-1)]
    # fitness_test = metrics[target_metric](y_test, pred_test)

    return pred_test, {
        "fitness_cv": fitness_cv,
        # "fitness_test" : fitness_test,
    }


def knn1_cv(X_train, X_test, y_train, target_metric, metric, whitened, dists=None):
    if whitened:
        X_train, X_test = whiten(X_train, X_test)

    train_dist = cdist(X_train, X_train, metric=metric)
    test_dist = cdist(X_test, X_train, metric=metric)
    return precomputed_knn1_cv(
        train_dist, test_dist, y_train, target_metric=target_metric
    )


# Diffusion
def run_diffusion(sim, n_neighbors=16, alpha=0.9, sym_mode="mean", k=None):

    # Convert similarity matrix to graph
    sim = sim.clip(min=0)
    np.fill_diagonal(sim, 0)
    thresh = np.sort(sim, axis=0)[-n_neighbors].reshape(1, -1)  # could use np.partition
    sim[sim < thresh] = 0
    sim[sim >= thresh] = 1

    # make knn graph -- how?
    if sym_mode == "max":
        adj = np.maximum(sim, sim.T)
    elif sym_mode == "min":
        adj = np.minimum(sim, sim.T)
    elif sym_mode == "mean":
        adj = (sim + sim.T) / 2
    else:
        raise Exception

    # symmetric normalization
    degrees = adj.sum(axis=1)
    degrees[degrees == 0] = 1e-6
    degrees = degrees ** -0.5
    D_sqinv = np.diag(degrees)

    adj_norm = D_sqinv.dot(adj).dot(D_sqinv)
    adj_norm = (adj_norm + adj_norm.T) / 2

    if k is None:
        k = adj_norm.shape[0]  # Full decomposition

    eigval, eigvec = eigsh(adj_norm, k=k)  # !! Should do something for scalability
    eigval = eigval.astype(np.float64)

    h_eigval = 1 / (1 - alpha * eigval)
    diffusion_scores = eigvec.dot(np.diag(h_eigval)).dot(eigvec.T)
    return diffusion_scores


def diffusion_cv(sim, y_train, param_grid, target_metric, verbose=10, ens_size=1):
    global _eval_grid_point
    n_train = y_train.shape[0]

    def _eval_grid_point(grid_point):
        try:
            diffusion_scores = run_diffusion(sim, **grid_point)

            score_train = diffusion_scores[:n_train, :n_train]
            np.fill_diagonal(score_train, -np.inf)
            cv_preds = y_train[score_train.argmax(axis=-1)]
            cv_score = metrics[target_metric](y_train, cv_preds)
            return grid_point, cv_score, cv_preds
        except:
            return grid_point, -1

    res = parmap(_eval_grid_point, ParameterGrid(param_grid), verbose=verbose)
    ranked = sorted(res, key=lambda x: -x[1])  # !! bigger is better

    assert ens_size > 0
    if ens_size == 1:
        best_params, cv_score, _ = ranked[0]
        diffusion_scores = run_diffusion(sim, **best_params)
        score_test = diffusion_scores[n_train:, :n_train]
    else:
        ens_size = min(len(ranked), ens_size)

        best_params = [b[0] for b in ranked[:ens_size]]
        diffusion_scores = [run_diffusion(sim, **b[0]) for b in ranked[:ens_size]]

        score_train = np.stack([d[:n_train, :n_train] for d in diffusion_scores]).mean(
            axis=0
        )
        np.fill_diagonal(score_train, -np.inf)
        cv_preds = y_train[score_train.argmax(axis=-1)]
        cv_score = metrics[target_metric](y_train, cv_preds)

        score_test = np.stack([d[n_train:, :n_train] for d in diffusion_scores]).mean(
            axis=0
        )

    pred_test = score_test.argmax(axis=-1)
    return y_train[pred_test], best_params, cv_score


# --
# Neighbors


class NeighborsCV(DistilBaseModel):
    def __init__(
        self,
        target_metric,
        metrics,
        diffusion=True,
        forest=True,
        whitens=[True, False],
        ensemble_size=3,
        diffusion_ensemble_size=3,
        verbose=True,
    ):

        self.target_metric = target_metric
        self.is_classification = target_metric in classification_metrics

        self.metrics = metrics
        self.whitens = whitens
        self.verbose = verbose
        self.ensemble_size = ensemble_size
        self.diffusion_ensemble_size = diffusion_ensemble_size

        self.diffusion = diffusion
        self.forest = forest

        self.preds = {}
        self.fitness = {}

        self._y_train = None

    def fit(self, X_train, y_train, U_train):
        X_test = U_train["X_test"]

        self._y_train = y_train

        # KNN models
        self._fit_knn(X_train, X_test, y_train)

        # Diffusion model, w/ best metric from KNN
        if self.diffusion:
            knn_settings = list(self.fitness.keys())
            metric, whitened = knn_settings[
                np.argmax([self.fitness[k]["fitness_cv"] for k in knn_settings])
            ]  # best settings
            self._fit_diffusion(X_train, X_test, y_train, metric, whitened)

        # Random Forest model
        if self.forest:
            self._fit_rf(X_train, X_test, y_train)

        return self

    def predict(self, X):
        # Ensembles K best models.  Handles ties correctly.
        # Shouldn't be ensembling when some of the models are absolute garbage
        # Should maybe drop models that do worse than chance.
        # Alternatively, we chould determine ensembling methods via CV

        print("!! NeighborsCV dos not use the passed argument", file=sys.stderr)

        all_fitness_cv = [v["fitness_cv"] for _, v in self.fitness.items()]

        if self.ensemble_size < len(all_fitness_cv):
            thresh = np.sort(all_fitness_cv)[-self.ensemble_size]  # bigger is better
        else:
            thresh = np.min(all_fitness_cv)

        ens_scores = np.vstack(
            [
                self.preds[k]
                for k, v in self.fitness.items()
                if v["fitness_cv"] >= thresh
            ]
        )
        if self.is_classification:
            ens_pred = tiebreaking_vote(ens_scores, self._y_train)
        else:
            ens_pred = ens_scores.mean(axis=0)  # Might linear regression be better?

        return ens_pred

    def _fit_knn(self, X_train, X_test, y_train):
        # Fit a series of KNN models

        global _dtw_dist_row_train

        print(X_train.shape, X_test.shape)

        X_all = np.vstack([X_train, X_test])
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        knn_settings = list(product(self.metrics, self.whitens))

        if ("dtw", True) in knn_settings:
            knn_settings.remove(("dtw", True))  # doesn't make sense

        for knn_setting in knn_settings:
            metric, whitened = knn_setting
            if metric != "dtw":
                pred, scores = knn1_cv(
                    X_train,
                    X_test,
                    y_train,
                    target_metric=self.target_metric,
                    metric=metric,
                    whitened=whitened,
                )

            else:

                def _dtw_dist_row_train(t):
                    return [
                        _fastdtw_metric(t, tt) for tt in X_train
                    ]  # !! Don't compute distance between test obs

                full_dist = np.vstack(
                    parmap(_dtw_dist_row_train, list(X_all), verbose=1)
                )
                train_dist, test_dist = full_dist[:n_train], full_dist[n_train:]
                pred, scores = precomputed_knn1_cv(
                    train_dist, test_dist, y_train, target_metric=self.target_metric
                )

            self.preds[knn_setting] = np.squeeze(pred)
            self.fitness[knn_setting] = scores

            if self.verbose:
                print(knn_setting, self.fitness[knn_setting], file=sys.stderr)

    def _fit_diffusion(self, X_train, X_test, y_train, metric, whitened):
        # Fit a diffusion model

        global _dtw_dist_row_all

        X_all = np.vstack([X_train, X_test])
        n_train = X_train.shape[0]
        n_test = X_test.shape[0]

        if whitened:
            X_all = PCA(whiten=True).fit_transform(X_all)  # !! Could be slow

        dist = distance_matrix(X_all, metric=metric)
        sim = 1 - dist / dist.max()

        param_grid = {
            "n_neighbors": [8, 16, 32, 64, 128, 256, 512],
            "sym_mode": ["min", "max", "mean"],
        }

        pred_test, _, scores = diffusion_cv(
            sim=sim,
            y_train=y_train,
            param_grid=param_grid,
            target_metric=self.target_metric,
            ens_size=self.diffusion_ensemble_size,
        )

        self.preds[("diff", metric, whitened)] = np.squeeze(pred_test)
        self.fitness[("diff", metric, whitened)] = {
            "fitness_cv": scores,
            # "fitness_test" : metrics[self.target_metric](y_test, pred_test),
        }

        if self.verbose:
            print(
                ("diff", metric, whitened),
                self.fitness[("diff", metric, whitened)],
                file=sys.stderr,
            )

    def _fit_rf(self, X_train, X_test, y_train):
        # Fit a RandomForest model

        forest = ForestCV(target_metric=self.target_metric)
        forest = forest.fit(X_train, y_train)
        pred_test = forest.predict(X_test)

        self.preds["rf"] = pred_test
        self.fitness["rf"] = {
            "fitness_cv": forest.best_fitness,
            # "fitness_test" : metrics[self.target_metric](y_test, pred_test)
        }

        if self.verbose:
            print("rf", self.fitness["rf"], file=sys.stderr)

    @property
    def details(self):
        return {
            "neighbors_fitness": dict([(str(k), v) for k, v in self.fitness.items()])
        }
