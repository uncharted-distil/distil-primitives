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

from distil.modeling.base import DistilBaseModel
from distil.modeling.forest import ForestCV

from joblib import Parallel, delayed

from numba import jit
from tqdm import tqdm

# --
# Helpers
import logging
from distil.third_party.torchvggish import vggish, vggish_input


logger = logging.getLogger(__name__)


@jit
def _mem_to_arr(w):
    return np.array(w, dtype="int16")


def audioarray2mel(data, sample_rate):
    assert data.shape[1] > 0, data.shape

    ret_val = vggish_input.waveform_to_examples(data / 32768.0, sample_rate)

    return ret_val


# --
# Model


class AudiosetModel(DistilBaseModel):
    def __init__(self, model_path, target_metric=None):
        self.target_metric = target_metric
        self.model_path = model_path

        self.embedding_model = vggish(self.model_path)
        self.embedding_model.eval()

    def _featurize(self, A):
        jobs = [delayed(audioarray2mel)(xx.data, xx.sample_rate) for xx in A]
        mel_feats = Parallel(n_jobs=64, backend="loky", verbose=10)(jobs)

        feature_vecs = []
        for i in tqdm(
            range(len(mel_feats)), desc="passing mel features through embedding model"
        ):
            feature_vec = self.embedding_model.forward(mel_feats[i]).data.numpy()
            if feature_vec.shape[0] == 128:
                feature_vec = feature_vec
            else:
                feature_vec = feature_vec.max(axis=0)

            feature_vecs.append(list(feature_vec))

        return np.array(feature_vecs)

    def fit(self, X_train, y_train, U_train=None):
        assert self.target_metric is not None, "define a target metric"

        vec_maxpool = self._featurize(X_train)
        self.model = ForestCV(target_metric=self.target_metric)
        self.model = self.model.fit(vec_maxpool, y_train)
        return self

    def predict(self, X):
        vec_maxpool = self._featurize(X)
        return self.model.predict(vec_maxpool)
