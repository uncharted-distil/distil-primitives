import os
import logging
from typing import List

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

from distil.primitives import utils
from distil.primitives.utils import CATEGORICALS

import pandas as pd
import numpy as np

from distil.preprocessing.transformers import BinaryEncoder


__all__ = ('BinaryEncoderPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

    min_binary = hyperparams.Hyperparameter[int](
        default=17,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Min number of unique labels a column can have for binary encoding.  If a column has fewer, it will be skipped.",
    )

class Params(params.Params):
    pass

class BinaryEncoderPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    Performs a binary encoding of categorical columns that are above a caller specified cardinality.  The source columns will be replaced by the
    encoding columns.  Some information is lost in comparison to a one-hot encoding, but the number of dimensions used is reduced.
    Categorical columns currently include those with the semantic type Categorical, Ordinal or DateTime.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'd38e2e28-9b18-4ce4-b07c-9d809cd8b915',
            'version': '0.1.0',
            'name': "Binary encoder",
            'python_path': 'd3m.primitives.data_transformation.encoder.DistilBinaryEncoder',
            'source': {
                'name': 'distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/binary_encoder.py',
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
                metadata_base.PrimitiveAlgorithmType.ENCODE_BINARY,
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
        state['models'] = self._encoders
        state['columns'] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._encoders = state['models']
        self._cols = state['columns']

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug('Fitting binary encoder')

        # find columns to operate on
        cols = utils.get_operating_columns(self._inputs, self.hyperparams['use_columns'], CATEGORICALS)

        filtered_cols: List[int] = []
        for c in cols:
            num_labels = len(set(self._inputs.iloc[:,c]))
            if num_labels > self.hyperparams['min_binary']:
                filtered_cols.append(c)
        self._cols = list(filtered_cols)

        logger.debug(f'Found {len(cols)} columns to encode')

        # add the binary encoded columns and remove the source
        self._encoders: List[BinaryEncoder] = []
        for c in self._cols:
            encoder = BinaryEncoder()
            categorical_inputs = self._inputs.iloc[:,c]
            encoder.fit(categorical_inputs)
            self._encoders.append(encoder)

        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        # add the binary encoded columns and remove the source columns
        outputs = inputs.copy()
        encoded_cols = container.DataFrame()
        for i, c in enumerate(self._cols):
            categorical_inputs = outputs.iloc[:,c]
            result = self._encoders[i].transform(categorical_inputs)
            for j in range(result.shape[1]):
                bin_idx = i * result.shape[1] + j
                encoded_cols[(f'__binary_{bin_idx}')] = result[:,j]
        encoded_cols.metadata = encoded_cols.metadata.generate(encoded_cols)

        for c in range(encoded_cols.shape[1]):
            encoded_cols.metadata = encoded_cols.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, c), 'http://schema.org/Integer')

        outputs = outputs.append_columns(encoded_cols)
        outputs = outputs.remove_columns(self._cols)

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return