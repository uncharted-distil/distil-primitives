import os
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from distil.modeling.forest import ForestCV
from distil.modeling.metrics import classification_metrics, regression_metrics

import pandas as pd
import numpy as np

__all__ = ('EnsembleForestV2',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The D3M scoring metric to use during the fit phase.  This can be any of the regression, classification or " +
                    "clustering metrics."
    )
    small_dataset_threshold = hyperparams.Hyperparameter[int](
        default=2000,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls the application of the 'small_dataset_fits' and 'large_dataset_fits' parameters - if the input dataset has " +
                    "fewer rows than the threshold value, 'small_dateset_fits' will be used when fitting.  Otherwise, 'num_large_fits' is used."
    )
    small_dataset_fits = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The number of random forests to fit when using small datasets."
    )
    large_dataset_fits = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The number of random forests to fit when using large datasets."
    )

class Params(params.Params):
    pass

class EnsembleForestV2Primitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    Generates an ensemble of random forests, with the number of internal models created controlled by the size of the
    input dataframe.  Returns a dataframe consisting of predictions only.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'e0ad06ce-b484-46b0-a478-c567e1ea7e02',
            'version': '0.2.0',
            'name': "EnsembleForestV2",
            'python_path': 'd3m.primitives.learner.random_forest.DistilEnsembleForestV2',
            'source': {
                'name': 'distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/ensemble_forest_v2.py',
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
                metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.LEARNER,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)
        self._model = ForestCV(self.hyperparams['metric'])

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._model
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._model = state['models']

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._model.num_fits = self.hyperparams['large_dataset_fits'] \
            if self._inputs.shape[0] > self.hyperparams['small_dataset_threshold'] else self.hyperparams['small_dataset_fits']

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        self._model.fit(self._inputs.values, self._outputs.values)
        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # create dataframe to hold the result
        result = self._model.predict(inputs.values)
        result_df = container.DataFrame({self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')
        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return