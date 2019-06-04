import os
from typing import List, Set, Any
import logging

from d3m import container, utils 
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ('TimeSeriesReshaperPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    pass

class Params(params.Params):
    pass

class TimeSeriesReshaperPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.List, container.ndarray, Params, Hyperparams]):
    """
    A primitive that reshapes time series.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a4375c5b-a439-4e3d-8ce5-2e8859097a51',
            'version': '0.1.0',
            'name': "Time series reshaper",
            'python_path': 'd3m.primitives.data_transformation.data_cleaning.DistilTimeSeriesReshaper',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/timeseries_reshaper.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )


    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:
        base.PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)
        self._sparse = False
        self._truncate_length = -1

    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state['sparse'] = self._sparse
        state['truncate_length'] = self._truncate_length
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._sparse = state['sparse']
        self._truncate_length = state['truncate_length']

    def set_training_data(self, *, inputs: container.List) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        # flag series as sparse
        self._sparse = (np.hstack(self._inputs) == 0).mean() > 0.5

        # compute min length to truncate mismatches
        ts_lengths = set([t.shape[0] for t in self._inputs])
        if len(ts_lengths) > 1:
            self._truncate_length = min(ts_lengths)

        return base.CallResult(None)

    def produce(self, *, inputs: container.List, timeout: float = None, iterations: int = None) -> base.CallResult[container.ndarray]:
        logger.debug(f'Producing {__name__}')

        outputs = [ df.copy() for df in inputs ]
        if self._sparse:
            logger.debug(f'Reformatting sparse timeseries')
            outputs = self._run_lengths_hist(outputs)
        if self._truncate_length > 0:
            logger.debug(f'Truncating timeseries to length {self._truncate_length}')
            outputs = self._truncate(outputs, self._truncate_length)

        outputs = np.vstack(outputs).astype(np.float64)

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

    @classmethod
    def _run_lengths_hist(cls, inputs: List) -> container.ndarray:
        train_rls = [np.diff(np.where(np.diff(inputs[i]))[0]).astype(int) for i in range(len(inputs))]
        thresh = np.percentile(np.hstack(train_rls), 95).astype(int)
        outputs = np.vstack([np.bincount(r[r <= thresh], minlength=thresh + 1) for r in train_rls])
        return outputs

    @classmethod
    def _truncate(cls, inputs: List, length: int) -> container.ndarray:
        inputs = [t[-length:] for t in inputs]
        return inputs