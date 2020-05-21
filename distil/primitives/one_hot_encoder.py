import os
from typing import List, Set, Any, Sequence
import logging

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

import pandas as pd
import numpy as np

from distil.primitives import utils as distil_utils
from distil.primitives.utils import CATEGORICALS
from distil.utils import CYTHON_DEP

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
    encoder: preprocessing.OneHotEncoder
    cols: List[int]

class OneHotEncoderPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    One-hot encodes categorical columns that equal or fall below a caller specified cardinality.  The source columns will be replaced by the
    encoding columns.  Categorical columns currently include those with the semantic type Categorical, Ordinal or DateTime.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'd3d421cb-9601-43f0-83d9-91a9c4199a06',
            'version': '0.2.1',
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
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
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
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        # figure out columns to operate on
        cols = distil_utils.get_operating_columns(self._inputs, self.hyperparams['use_columns'], CATEGORICALS)

        filtered_cols: List[int] = []
        for c in cols:
            num_labels = len(set(self._inputs.iloc[:,c]))
            if num_labels <= self.hyperparams['max_one_hot']:
                filtered_cols.append(c)

        logger.debug(f'Found {len(filtered_cols)} columns to encode')

        self._cols = list(filtered_cols)
        self._encoder = None
        if len(self._cols) is 0:
            return base.CallResult(None)

        input_cols = self._inputs.iloc[:,self._cols]
        self._encoder = preprocessing.OneHotEncoder(sparse=False, handle_unknown='ignore')
        self._encoder.fit(input_cols)

        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        # map encoded cols to source column names
        feature_names = self._encoder.get_feature_names()
        encoded_cols_source = []
        # feature names are xA_YY where A is the source column index and YY is the value
        for name in feature_names:
            # take the first part of the name (xA) and remove the x
            encoded_feature_index = int(name.split('_')[0][1:])
            feature_index = self._cols[encoded_feature_index]
            encoded_cols_source.append(inputs.metadata.query((metadata_base.ALL_ELEMENTS, feature_index))['name'])

        # encode using the previously identified categorical columns
        input_cols = inputs.iloc[:,self._cols]
        result = self._encoder.transform(input_cols)

        # append the encoding columns and generate metadata
        outputs = inputs.copy()
        encoded_cols: container.DataFrame = container.DataFrame()

        for i in range(result.shape[1]):
            encoded_cols[f'__onehot_{str(i)}'] = result[:,i]
        encoded_cols.metadata = encoded_cols.metadata.generate(encoded_cols)

        for c in range(encoded_cols.shape[1]):
            encoded_cols.metadata = encoded_cols.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, c), 'http://schema.org/Float')
            col_dict = dict(encoded_cols.metadata.query((metadata_base.ALL_ELEMENTS, c)))
            col_dict['source_column'] = encoded_cols_source[c]
            encoded_cols.metadata = encoded_cols.metadata.update((metadata_base.ALL_ELEMENTS, c), col_dict)

        outputs = outputs.append_columns(encoded_cols)

        # drop the source columns
        outputs = outputs.remove_columns(self._cols)

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params(
            encoder = self._encoder,
            cols = self._cols
        )

    def set_params(self, *, params: Params) -> None:
        self._encoder = params['encoder']
        self._cols = params['cols']
        return