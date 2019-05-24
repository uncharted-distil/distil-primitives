import os
from typing import List, Set, Any, Sequence
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import compose

logger = logging.getLogger(__name__)

__all__ = ('OneHotEncoderPrimitive',)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

    max_one_hot = hyperparams.Hyperparameter[int](
        default=16,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Max number of unique labels a column can have for encoding.  If the value is surpassed, the column is skipped.",
    )

class Params(params.Params):
    pass

class OneHotEncoderPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    One-hot encodes categorical columns that equal or fall below a caller specified cardinality.  The source columns will be replaced by the
    encoding columns.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'd3d421cb-9601-43f0-83d9-91a9c4199a06',
            'version': '0.1.0',
            'name': "One-hot encoder",
            'python_path': 'd3m.primitives.data_transformation.one_hot_encoder.DistilOneHotEncoder',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/one_hot_encoder.py',
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
                metadata_base.PrimitiveAlgorithmType.ENCODE_ONE_HOT,
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
        state['models'] = self._encoder
        state['columns'] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._encoder = state['models']
        self._cols = state['columns']

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        # use caller supplied columns if set
        cols = set(self.hyperparams['use_columns'])
        categorical_cols = set(self._inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                                                                       'https://metadata.datadrivendiscovery.org/types/OrdinalData')))
        if len(cols) > 0:
            cols = categorical_cols & cols
        else:
            cols = categorical_cols

        filtered_cols: List[int] = []
        for c in cols:
            num_labels = len(set(self._inputs.iloc[:,c]))
            if num_labels <= self.hyperparams['max_one_hot']:
                filtered_cols.append(c)

        logger.debug(f'Found {len(cols)} columns to encode')

        self._cols = list(cols)
        self._encoder = None
        if len(cols) is 0:
            return base.CallResult(None)

        input_cols = self._inputs.iloc[:,self._cols]
        self._encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._encoder.fit(input_cols)

        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        # encode using the previously identified categorical columns
        input_cols = inputs.iloc[:,self._cols]
        result = self._encoder.transform(input_cols)

        # remove the source columns and append the result to the copy
        outputs = inputs.copy()
        for i in range(result.shape[1]):
            outputs[('__onehot_' + str(i))] = result[:,i]
        outputs.drop(outputs.columns[self._cols], axis=1, inplace=True)

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

    @classmethod
    def _detect_text(cls, X: container.DataFrame, thresh: int = 8) -> bool:
        """ returns true if median entry has more than `thresh` tokens"""
        X = X[X.notnull()]
        n_toks = X.apply(lambda xx: len(str(xx).split(' '))).values
        return np.median(n_toks) >= thresh