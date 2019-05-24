import os
import logging
from typing import List, Tuple, Sequence

from distil.modeling.neighbors import NeighborsCV
from distil.modeling.metrics import metrics, regression_metrics, classification_metrics

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase

import pandas as pd
import numpy as np


_all__ = ('TimeSeriesNeighboursPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='f1Macro',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class Params(params.Params):
    pass


class TimeSeriesNeighboursPrimitive(PrimitiveBase[container.ndarray, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that filters collaboratives.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a8649c58-25ae-4578-87da-bc6cbc68c98e',
            'version': '0.1.0',
            'name': "Timeseries neighbours",
            'python_path': 'd3m.primitives.learner.random_forest.DistilTimeSeriesNeighboursPrimitive',
            'source': {
                'name': 'distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/timeseries_neighbours.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:
        base.PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)

    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state['model'] = self._model
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._model = state['model']

    def set_training_data(self, *, inputs: container.ndarray, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        metrics = ['euclidean', 'cityblock', 'dtw'] # TODO: should remove dtw in sparse case
        diffusion = self.hyperparams['metric'] in classification_metrics
        ensemble_size = 3

        self._model = NeighborsCV(self.hyperparams['metric'], metrics, diffusion=diffusion, forest=True, ensemble_size=3)
        self._model.fit(self._inputs, self._outputs.values, {'X_test': self._inputs})

        return base.CallResult(None)

    def produce(self, *, inputs: container.ndarray, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # create dataframe to hold d3mIndex and result
        result = self._model.predict(inputs)
        result_df = container.DataFrame({self._outputs.index.name: self._outputs.index, self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return
