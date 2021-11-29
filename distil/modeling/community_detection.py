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

from .base import DistilBaseModel
from .metrics import metrics


class CommunityDetection(DistilBaseModel):
    def __init__(self, overlapping):
        self.overlapping = overlapping

    def fit(self, X_train, y_train, U_train=None):
        print("!! CommunityDetection: using null model", file=sys.stderr)
        return self

    def predict(self, X):
        return -np.arange(X.shape[0])

    @property
    def details(self):
        return {"null_model": True}
