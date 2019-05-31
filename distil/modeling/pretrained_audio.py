import os
import sys
import numpy as np
from tqdm import tqdm

from distil.third_party.audioset import vggish_input
from distil.third_party.audioset import vggish_params
from distil.third_party.audioset import vggish_postprocess
from distil.third_party.audioset import vggish_slim

import tensorflow as tf

from .base import DistilBaseModel
from .forest import ForestCV
from .metrics import metrics
from ..utils import parmap

from tensorflow.errors import InvalidArgumentError
from joblib import Parallel, delayed

from numba import jit
import gc
# --
# Helpers
import logging

logger = logging.getLogger(__name__)

@jit
def _mem_to_arr(w):
    return np.array(w, dtype='int16')

def audioarray2mel(data, sample_rate):
    assert data.shape[1] > 0, data.shape

    ret_val = vggish_input.waveform_to_examples(data / 32768.0, sample_rate)

    return ret_val


def audio2vec(X, model_path):
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, model_path)

        feat_ = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        emb_ = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        all_feats = []
        for xx in tqdm(X):

            [feats] = sess.run([emb_], feed_dict={feat_: xx})
            all_feats.append(feats)

        return all_feats

# --
# Model

class AudiosetModel(DistilBaseModel):

    def __init__(self, model_path, target_metric=None):
        self.target_metric = target_metric
        self.model_path = model_path


    def _featurize(self, A):

        jobs = [delayed(audioarray2mel)(xx.data, xx.sample_rate) for xx in A]
        mel_feats = Parallel(n_jobs=32, backend='multiprocessing', verbose=10)(jobs)

        vec_feats = audio2vec(mel_feats, self.model_path)

        return np.vstack([f.max(axis=0) for f in vec_feats])

    def fit(self, X_train, y_train, U_train=None):
        assert self.target_metric is not None, 'define a target metric'

        vec_maxpool = self._featurize(X_train)
        self.model = ForestCV(target_metric=self.target_metric)
        self.model = self.model.fit(vec_maxpool, y_train)
        return self

    def predict(self, X):
        vec_maxpool = self._featurize(X)
        return self.model.predict(vec_maxpool)