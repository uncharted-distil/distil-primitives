import os
import logging
from typing import List

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase

import pandas as pd
import numpy as np

from exline.modeling.bert_models import BERTPairClassification


_all__ = ('BertClassification',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class Params(params.Params):
    pass


class BertClassificationPrimitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that berts.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7c305f3a-442a-41ad-b9db-8c437753b119',
            'version': '0.1.0',
            'name': "Bert models",
            'python_path': 'd3m.primitives.learner.random_forest.ExlineBertClassification',
            'source': {
                'name': 'exline',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/cdbethune/d3m-exline/exline/primitives/bert_classification.py',
                    'https://github.com/cdbethune/d3m-exline',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/cdbethune/d3m-exline.git@{git_commit}#egg=d3m-exline'.format(
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
        self._model = BERTPairClassification(self.hyperparams['metric'])

    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state['model'] = self._model
        state['columns'] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._model = state['model']
        self._cols = state['columns']

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        self._model.fit(self._inputs, self._outputs)
        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # create dataframe to hold d3mIndex and result
        result = self._model.predict(inputs)
        result_df = container.DataFrame({inputs.index.name: inputs.index, self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')
        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return