import os
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import unsupervised_learning, transformer, base

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans



from distil.primitives.utils import MISSING_VALUE_INDICATOR

logger = logging.getLogger(__name__)

__all__ = ('KMeansPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    n_clusters = hyperparams.Hyperparameter[int](
        default = 8,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Number of clusters to generate",
    )


class Params(params.Params):
    pass

class KMeansPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that scales standards.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '3b09024e-a83b-418c-8ff4-cf3d30a9609e',
            'version': '0.1.0',
            'name': "K means",
            'python_path': 'd3m.primitives.clustering.k_means.DistilKMeans',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/k_means.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.K_MEANS_CLUSTERING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:
        base.PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)

    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state['columns'] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._cols = state['columns']

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        # find candidate columns
        cols = list(self.hyperparams['use_columns'])
        if cols is None or len(cols) is 0:
            cols = []
            for idx, c in enumerate(self._inputs.columns):
                if (self._inputs[c].dtype == int or self._inputs[c].dtype == float or self._inputs[c].dtype == bool):
                    cols.append(idx)

        logger.debug(f'Found {len(cols)} cols to use for clustering')
        self._cols = cols
        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        numerical_inputs = inputs.iloc[:,self._cols]
        k_means = KMeans(n_clusters = self.hyperparams['n_clusters'])
        result_df = container.DataFrame(k_means.fit_predict(numerical_inputs), generate_metadata=True)

        logger.debug(f'\n{result_df}')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return
