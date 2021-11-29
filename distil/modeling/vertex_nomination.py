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
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import linalg

from .base import DistilBaseModel
from .forest import ForestCV
from .svm import SupportVectorCV
from .metrics import metrics, classification_metrics


class VertexNominationCV(DistilBaseModel):
    def __init__(self, target_metric, num_components=8, random_seed=None):
        self.target_metric = target_metric
        self.num_components = num_components
        self.random__seed = random_seed
        self.feats = None

    def fit(self, X_train, y_train, U_train=None):
        graph = U_train["graph"]
        X_train = X_train.copy()
        assert X_train.shape[1] == 1

        X_train.columns = ("nodeID",)

        # --
        # Featurize

        df = pd.DataFrame([graph.nodes[i] for i in graph.nodes]).set_index("nodeID")
        adj = nx.adjacency_matrix(graph).astype(np.float64)
        U, _, _ = linalg.svds(adj, k=self.num_components)

        self.feats = pd.DataFrame(np.hstack([df.values, U])).set_index(df.index)

        Xf_train = self.feats.loc[X_train.nodeID].values

        # --
        # Choose the best model

        print("VertexNominationCV: ForestCV", file=sys.stderr)
        forest = False
        try:
            forest = ForestCV(
                target_metric=self.target_metric,
                random_seed=self.random__seed,
                grid_search=True,
            )
            forest = forest.fit(Xf_train, y_train)
        except:
            pass

        print("VertexNominationCV: SupportVectorCV", file=sys.stderr)
        svm = False
        try:
            svm = SupportVectorCV(
                target_metric=self.target_metric, random_seed=self.random__seed
            )
            svm = svm.fit(Xf_train, y_train)
        except:
            pass

        self.model = forest
        self.best_params = forest.best_params
        self.best_fitness = forest.best_fitness

        if hasattr(svm, "best_fitness"):
            if svm.best_fitness > forest.best_fitness:
                self.model = svm.model
                self.best_params = svm.best_params
                self.score_cv = svm.best_fitness

        return self

    def predict(self, X, U):
        # X = X.copy()
        # assert X.shape[1] == 1
        # X.columns = ('nodeID',)
        #
        # Xf = self.feats.loc[X.nodeID].values
        # return self.model.predict(Xf)

        graph = U["graph"]
        X = X.copy()
        assert X.shape[1] == 1

        X.columns = ("nodeID",)

        # --
        # Featurize
        df = pd.DataFrame([graph.nodes[i] for i in graph.nodes]).set_index("nodeID")

        adj = nx.adjacency_matrix(graph).astype(np.float64)
        U, _, _ = linalg.svds(adj, k=self.num_components)

        feats = pd.DataFrame(np.hstack([df.values, U])).set_index(df.index)

        Xf = feats.reindex(X.nodeID).values
        Xf = (
            pd.DataFrame(Xf).fillna(0).values
        )  # force nan to zero if mismatch between graph and nodes.
        return self.model.predict(Xf)
