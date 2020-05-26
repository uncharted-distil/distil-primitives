import logging
import os

import numpy as np
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import unsupervised_learning, base
from distil.primitives import utils as distil_utils
from distil.utils import CYTHON_DEP
from sklearn.cluster import KMeans
from typing import Optional, List

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
    cluster_col_name = hyperparams.Hyperparameter[str](
        default = '__cluster',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The name of created cluster column in the returned dataframe",
    )


class Params(params.Params):
    _columns: Optional[List[int]]

class KMeansPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A wrapper for scikit learn k-means that takes in a dataframe as input and returns a dataframe of (d3mIndex, cluster numbers) tuples as its
    output.  It will ignore columns with a string structural type.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '3b09024e-a83b-418c-8ff4-cf3d30a9609e',
            'version': '0.1.1',
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
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
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
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

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
        self._cols = distil_utils.get_operating_columns_structural_type(self._inputs, self.hyperparams['use_columns'], (np.float64, np.int64), False)
        logger.debug(f'Found {len(self._cols)} cols to use for clustering')
        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        numerical_inputs = inputs.iloc[:,self._cols]
        k_means = KMeans(n_clusters = self.hyperparams['n_clusters'])
        result = k_means.fit_predict(numerical_inputs)
        result_df = container.DataFrame({self.hyperparams['cluster_col_name']: result}, generate_metadata=True)
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params(_columns = self._cols)

    def set_params(self, *, params: Params) -> None:
        self._cols = params['_columns']
