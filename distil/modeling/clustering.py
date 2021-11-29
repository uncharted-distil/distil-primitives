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

from sklearn.cluster import KMeans

from .base import DistilBaseModel
from .metrics import metrics


class ClusteringCV(DistilBaseModel):
    def __init__(self, target_metric, n_clusters, all_float):

        self.target_metric = target_metric
        self.n_clusters = n_clusters
        self.all_float = all_float
        self.n_init = 100

    def fit(self, X_train, y_train, U_train=None):
        assert X_train.shape[0] == 0
        assert self.all_float

        print("!! ClusteringCV.fit does nothing")
        return self

    def predict(self, X):
        self.model = KMeans(n_clusters=self.n_clusters, n_init=self.n_init)
        return self.model.fit_predict(X)
