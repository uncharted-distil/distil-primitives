import sys
import numpy as np

from .base import DistilBaseModel
from .metrics import metrics

class CommunityDetection(DistilBaseModel):

    def __init__(self, overlapping):
        self.overlapping   = overlapping

    def fit(self, X_train, y_train, U_train=None):
        print('!! CommunityDetection: using null model', file=sys.stderr)
        return self

    def predict(self, X):
        return -np.arange(X.shape[0])

    @property
    def details(self):
        return {
            "null_model" : True
        }
