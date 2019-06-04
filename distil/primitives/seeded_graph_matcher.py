import os
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils 
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult

import pandas as pd
import numpy as np
import networkx as nx
from scipy import sparse

from distil.modeling.sgm import SGMGraphMatcher

__all__ = ('SeededGraphMatcher',)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Params(params.Params):
    pass

class DistilSeededGraphMatchingPrimitive(PrimitiveBase[container.List, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that matches seeded graphs.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8baea8e6-9d3a-46d7-acf1-04fd593dcd37',
            'version': '0.1.0',
            'name': "SeededGraphMatcher",
            'python_path': 'd3m.primitives.data_transformation.seeded_graph_matcher.DistilSeededGraphMatcher',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:fred@qntfy.com',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/seeded_graph_matcher.py',
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

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)
        self._model = SGMGraphMatcher(target_metric='accuracy')

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._model
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._model = state['models']

    def set_training_data(self, *, inputs: container.List, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        X_train, y_train, U_train = self._inputs
        X_train = X_train.value
        self._model.fit(X_train, y_train, U_train)

        return CallResult(None)

    def produce(self, *, inputs: container.List, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        X_train, _, _ = inputs
        X_train = X_train.value
        result = self._model.predict(X_train)

        # create dataframe to hold d3mIndex and result
        result_df = container.DataFrame({X_train.index.name: X_train.index, self._outputs.columns[0]: result})

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return