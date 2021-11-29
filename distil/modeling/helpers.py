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
from scipy.optimize import minimize

from .metrics import metrics

# ---------------------------------------------------------
# F1 macro adjusters


def _adjusted_f1_macro(probs, y, bias):
    return metrics["f1Macro"](y, (probs + bias).argmax(axis=-1))


def _adjust_f1_inner(probs_oob, y_train, alpha):
    def _make_adj(bias):
        score = _adjusted_f1_macro(probs_oob, y_train, bias)
        norm = alpha * np.abs(bias).mean()
        return (-score) + norm

    bias = np.zeros((1, probs_oob.shape[1]))
    bias = minimize(_make_adj, bias, method="Powell").x
    return bias


def adjust_f1_macro(model, Xf_test, y_train, y_test, alpha=0.01, subset=-1):
    # !! y_train and y_test must be sequential numeric

    probs_oob = model.oob_decision_function_

    if (subset > 0) and (subset < probs_oob.shape[0]):
        sel = np.sort(np.random.choice(probs_oob.shape[0], subset, replace=False))
        bias = _adjust_f1_inner(probs_oob[sel], y_train[sel], alpha=alpha)
    else:
        bias = _adjust_f1_inner(probs_oob, y_train, alpha=alpha)

    probs_test = model.predict_proba(Xf_test)

    print("_make_adj:train_null", _adjusted_f1_macro(probs_oob, y_train, 0))
    print("_make_adj:train_bias", _adjusted_f1_macro(probs_oob, y_train, bias))
    print("_make_adj:test_null", _adjusted_f1_macro(probs_test, y_test, 0))
    print("_make_adj:test_bias", _adjusted_f1_macro(probs_test, y_test, bias))
    print("--")
    return _adjusted_f1_macro(probs_test, y_test, bias)


# --
# Ensembling


def _vote(p, tiebreaker):
    cnts = pd.value_counts(p)

    # if unique maximum, return it
    # otherwise, break ties w/ tiebreaker

    if (cnts == cnts.max()).sum() == 1:
        return cnts.index[0]
    else:
        tie = cnts[cnts == cnts.max()].index
        for t in tiebreaker:
            if t in tie:
                return t


def tiebreaking_vote(preds, y_train):
    # Vote, breaking ties according to class prevalance
    labels = pd.unique(y_train.squeeze())
    return tiebreaking_vote_pre(preds, labels)


def tiebreaking_vote_pre(preds, labels):
    # Tiebreaking vote but uses a pre-calculated list of labels
    if preds.shape[0] == 1:
        # shortcut if there is only set of predictions
        return preds[0]
    else:
        # CDB: this is a major bottle neck at scale
        return np.array([_vote(p, labels) for p in preds.T])
