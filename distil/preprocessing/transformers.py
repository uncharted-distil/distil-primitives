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

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.cluster import MiniBatchKMeans
from typing import Dict, Optional

from distil.primitives.utils import MISSING_VALUE_INDICATOR, SINGLETON_INDICATOR
from sklearn.decomposition import PCA
from distil.modeling.metrics import metrics, classification_metrics, regression_metrics

# --
# Categorical


class BinaryEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, random_seed=None):
        super().__init__()
        self.lookup = None
        self.random_seed = (
            random_seed if random_seed is not None else int(np.random.randint)
        )

    def fit(self, X):
        levels = list(set(X.squeeze())) + [
            MISSING_VALUE_INDICATOR
        ]  # !! How to sort? Randomly? Alphabetically?
        # use th
        random_state = np.random.RandomState(self.random_seed)
        levels = random_state.permutation(levels)

        vals = range(len(levels))
        max_width = len(np.binary_repr(max(vals)))
        vals = [np.binary_repr(v, width=max_width) for v in vals]
        vals = [np.array(list(v)).astype(int) for v in vals]

        self.lookup = dict(zip(levels, vals))
        return self

    def transform(self, X, y=None):
        assert self.lookup is not None
        squeezed = X.squeeze()
        # single row element will become scalar after squeeze
        if not isinstance(squeezed, pd.Series):
            squeezed = [squeezed]
        return np.vstack(
            [
                self.lookup.get(xx, self.lookup[MISSING_VALUE_INDICATOR])
                for xx in squeezed
            ]
        )

    def fit_transform(self, X, y=None, **kwargs):
        _ = self.fit(X)
        return self.transform(X)


# --
# Text


class SVMTextEncoder(BaseEstimator, TransformerMixin):
    # number of jobs to execute in parallel
    NUM_JOBS = 3
    # number of folds to apply to svm fit
    NUM_FOLDS = 3

    # !! add tuning
    def __init__(self, metric, random_seed):
        super().__init__()

        self._vect = TfidfVectorizer(ngram_range=[1, 2], max_features=30000)
        self._random_seed = random_seed

        if metric in classification_metrics:
            self._model = LinearSVC(class_weight="balanced", random_state=random_seed)
            self.mode = "classification"
        elif metric in regression_metrics:
            self._model = LinearSVR(random_state=random_seed)
            self.mode = "regression"
        else:
            raise AttributeError("metric not in classification or regression metrics")

    def fit(self, X, y):
        raise NotImplemented

    def transform(self, X):
        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR).values

        Xv = self._vect.transform(X)
        if self.mode == "classification":
            out = self._model.decision_function(Xv)
        else:
            out = self._model.predict(Xv)

        if len(out.shape) == 1:
            out = out.reshape(-1, 1)

        return out

    def fit_transform(self, X, y=None, **kwargs):
        assert y is not None, "SVMTextEncoder.fit_transform requires y"

        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR).values
        Xv = self._vect.fit_transform(X)
        self._model = self._model.fit(Xv, y)

        if self.mode == "classification":
            # Aim for NUM_FOLDS and stratified k-fold.  If that doesn't work, fallback to uniform sampling.
            num_folds = min(self.NUM_FOLDS, y.value_counts().min())
            if num_folds < 2:
                cv = KFold(n_splits=self.NUM_FOLDS, random_state=self._random_seed)
                out = cross_val_predict(
                    self._model,
                    Xv,
                    y,
                    method="decision_function",
                    n_jobs=self.NUM_JOBS,
                    cv=cv,
                )
            else:
                out = cross_val_predict(
                    self._model,
                    Xv,
                    y,
                    method="decision_function",
                    n_jobs=self.NUM_JOBS,
                    cv=num_folds,
                )
        else:
            out = cross_val_predict(
                self._model, Xv, y, n_jobs=self.NUM_JOBS, cv=self.NUM_FOLDS
            )

        if len(out.shape) == 1:
            out = out.reshape(-1, 1)

        return out


class TfidifEncoder(BaseEstimator, TransformerMixin):
    # !! add tuning
    def __init__(self):
        super().__init__()

        self._vect = TfidfVectorizer(ngram_range=[1, 2], max_features=300)
        self._pca = PCA(n_components=16)
        self.label_map: Optional[Dict[int, str]] = None

    def fit(self, X, y):
        raise NotImplemented

    def transform(self, X):

        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR)

        if self.label_map:
            self.label_map_inv = {v: k for k, v in self.label_map.items()}
            # fillna is mostly needed if subset of data was trained on
            X = X.map(self.label_map_inv).fillna(0).values
        else:
            X = self._vect.transform(X).toarray()
            X = self._pca.transform(X)
        out = X

        if len(out.shape) == 1:
            out = out.reshape(-1, 1)

        return out

    def fit_transform(self, X, y=None, **kwargs):

        X = pd.Series(X.squeeze()).fillna(MISSING_VALUE_INDICATOR)
        if len(X.unique()) / len(X) < 0.5:  # TODO should be pulled from metadata
            factor = pd.factorize(X)
            X = factor[0]
            self.label_map = {k: v for k, v in enumerate(factor[1])}
        else:
            X = self._vect.fit_transform(X).toarray()
            X = self._pca.fit_transform(X)

        out = X

        if len(out.shape) == 1:
            out = out.reshape(-1, 1)

        return out


# --
# Timeseries


def run_lengths_hist(T_train, T_test):
    # !! Super simple -- ignores values

    train_rls = [
        np.diff(np.where(np.diff(T_train[i]))[0]).astype(int)
        for i in range(len(T_train))
    ]
    test_rls = [
        np.diff(np.where(np.diff(T_test[i]))[0]).astype(int) for i in range(len(T_test))
    ]

    thresh = np.percentile(np.hstack(train_rls), 95).astype(int)

    H_train = np.vstack(
        [np.bincount(r[r <= thresh], minlength=thresh + 1) for r in train_rls]
    )
    H_test = np.vstack(
        [np.bincount(r[r <= thresh], minlength=thresh + 1) for r in test_rls]
    )

    return H_train, H_test


# --
# Sets


def set2hist(
    S_train,
    S_test,
    n_clusters=64,
    n_jobs=32,
    kmeans_sample=100000,
    batch_size=1000,
    verbose=False,
):

    S_train, S_test = list(S_train), list(S_test)
    n_train_obs, n_test_obs = len(S_train), len(S_test)

    train_offsets = np.cumsum([t.shape[0] for t in S_train])
    train_offsets = np.hstack([[0], train_offsets])

    test_offsets = np.cumsum([t.shape[0] for t in S_test])
    test_offsets = np.hstack([[0], test_offsets])

    dim = S_train[0].shape[1]
    assert len(set([t.shape[1] for t in S_train])) == 1
    assert len(set([t.shape[1] for t in S_test])) == 1

    S_train_flat, S_test_flat = np.vstack(S_train), np.vstack(S_test)
    T_all = np.vstack([S_train_flat, S_test_flat])

    kmeans_sample = min(kmeans_sample, T_all.shape[0])
    sel = np.random.choice(T_all.shape[0], kmeans_sample, replace=False)
    km = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size, verbose=verbose
    ).fit(T_all[sel])

    cl_train = km.predict(S_train_flat)
    H_train = np.vstack(
        [
            np.histogram(
                cl_train[train_offsets[i] : train_offsets[i + 1]],
                bins=range(n_clusters + 1),
            )[0]
            for i in range(n_train_obs)
        ]
    )

    cl_test = km.predict(S_test_flat)
    H_test = np.vstack(
        [
            np.histogram(
                cl_test[test_offsets[i] : test_offsets[i + 1]],
                bins=range(n_clusters + 1),
            )[0]
            for i in range(n_test_obs)
        ]
    )

    return H_train, H_test


# --
# Graphs


def cf_remap_graphs(X_train, X_test):

    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train.columns = ("user", "item")
    X_test.columns = ("user", "item")

    uusers = np.unique(np.hstack([X_train.user, X_test.user]))
    user_lookup = dict(zip(uusers, range(len(uusers))))
    X_train.user = X_train.user.apply(user_lookup.get)
    X_test.user = X_test.user.apply(user_lookup.get)

    uitems = np.unique(np.hstack([X_train.item, X_test.item]))
    item_lookup = dict(zip(uitems, range(len(uitems))))
    X_train.item = X_train.item.apply(item_lookup.get)
    X_test.item = X_test.item.apply(item_lookup.get)

    n_users = len(uusers)
    n_items = len(uitems)

    return X_train, X_test, n_users, n_items
