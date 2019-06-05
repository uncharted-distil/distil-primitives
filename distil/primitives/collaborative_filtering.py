import os
import logging
from typing import List, Tuple, Mapping
from collections import defaultdict
import random

from d3m import container, utils 
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase

import pandas as pd
import numpy as np

from sklearn import preprocessing
import torch

from distil.modeling.collaborative_filtering import SGDCollaborativeFilter
from distil.primitives import utils


_all__ = ('CollaborativeFilteringPrimtive',)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The D3M scoring metric to optimize for when fitting.'
    )
    user_col = hyperparams.Hyperparameter[int](
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The index of the column containing the user / agent IDs.'
    )
    item_col = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The index of the column containing the item IDs.'
    )
    force_cpu = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Force CPU execution regardless of GPU availability.'
    )

class Params(params.Params):
    pass


class CollaborativeFilteringPrimitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A collaborative filtering primitive based on pytorch.  Will use available GPU resources, or run in a CPU mode at a significant
    performance penalty.  Takes a dataframe containing user IDs, item IDs, and ratings as training input, and produces a dataframe
    containing rating predictions as output.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a242314d-7955-483f-aed6-c74cd2b880df',
            'version': '0.1.0',
            'name': "Collaborative filtering",
            'python_path': 'd3m.primitives.learner.random_forest.DistilCollaborativeFiltering',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/collaborative_filtering.py',
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
                metadata_base.PrimitiveAlgorithmType.STOCHASTIC_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.COLLABORATIVE_FILTERING,
        },
    )


    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:
        base.PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)


    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state['model'] = self._model
        state['encoders'] = self._encoders
        return state


    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._model = state['model']
        self._encoders = state['encoders']


    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._encoders: Mapping[str, preprocessing.LabelEncoder] = defaultdict(preprocessing.LabelEncoder)


    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        if torch.cuda.is_available():
            if self.hyperparams['force_cpu']:
                logger.info("Detected CUDA support - forcing use of CPU")
                device = "cpu"
            else:
                logger.info("Detected CUDA support - using GPU")
                device = "cuda"
        else:
            logger.info("CUDA does not appear to be supported - using CPU.")
            device = "cpu"

        # extract columns
        inputs = self._inputs.iloc[:, [self.hyperparams['user_col'], self.hyperparams['item_col']]]

        # need int encoded inputs for the underlying model
        encoded_inputs = inputs.apply(lambda x: self._encoders[x.name].fit_transform(x))

        graph, n_users, n_items = self._remap_graphs(inputs)
        self._model = SGDCollaborativeFilter(n_users, n_items, self.hyperparams['metric'], device=device)
        self._model.fit(encoded_inputs, self._outputs.values)

        return base.CallResult(None)


    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')
        # extract and encode user and item columns
        inputs = inputs.iloc[:, [self.hyperparams['user_col'], self.hyperparams['item_col']]]
        inputs = inputs.apply(lambda x: self._encoders[x.name].transform(x))
        # predict ratings
        result = self._model.predict(inputs)
        # create dataframe to hold result
        result_df = container.DataFrame({self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')
        return base.CallResult(result_df)


    def get_params(self) -> Params:
        return Params()


    def set_params(self, *, params: Params) -> None:
        return


    @classmethod
    def _remap_graphs(cls, data: container.DataFrame) -> Tuple[container.DataFrame, int, int]:
        assert data.shape[1] == 2

        data = data.copy()

        data.columns = ('user', 'item')

        uusers       = np.unique([data.user, data.user])
        user_lookup  = dict(zip(uusers, range(len(uusers))))
        data.user = data.user.apply(user_lookup.get)

        uitems       = np.unique(data.item)
        item_lookup  = dict(zip(uitems, range(len(uitems))))
        data.item = data.item.apply(item_lookup.get)

        n_users = len(uusers)
        n_items = len(uitems)

        return data, n_users, n_items