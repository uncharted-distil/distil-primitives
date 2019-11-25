import numpy as np

from distil.modeling.base import DistilBaseModel
from distil.modeling.forest import ForestCV

from joblib import Parallel, delayed

from numba import jit
# --
# Helpers
import logging
from torchvggish import vggish, vggish_input

# Initialise model and download weights
embedding_model = vggish()
embedding_model.eval()

logger = logging.getLogger(__name__)


@jit
def _mem_to_arr(w):
    return np.array(w, dtype='int16')


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

    def _featurize(self, A):
        jobs = [delayed(audioarray2mel)(xx.data, xx.sample_rate) for xx in A]
        mel_feats = Parallel(n_jobs=64, backend='loky', verbose=10)(jobs)

        mels = []
        for i in range(len(mel_feats)):
            mels.append(embedding_model.forward(mel_feats[i]).data.numpy())

        return np.vstack([f.max(axis=0) for f in mels])

    def fit(self, X_train, y_train, U_train=None):
        assert self.target_metric is not None, 'define a target metric'

        vec_maxpool = self._featurize(X_train)
        self.model = ForestCV(target_metric=self.target_metric)
        self.model = self.model.fit(vec_maxpool, y_train)
        return self

    def predict(self, X):
        vec_maxpool = self._featurize(X)
        return self.model.predict(vec_maxpool)
