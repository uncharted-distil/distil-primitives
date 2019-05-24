import os
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from common_primitives import utils as common_utils

import pandas as pd
import numpy as np

from sklearn_pandas import CategoricalImputer
from sklearn import compose

from distil.preprocessing.utils import MISSING_VALUE_INDICATOR


__all__ = ('CategoricalImputerPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

    strategy = hyperparams.Enumeration[str](
        default='most_frequent',
        values=('most_frequent', 'constant'),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Replacement strategy.  'most_frequent' will replace missing values with the mode of the column, 'constant' uses 'fill_value'",
    )

    fill_value = hyperparams.Hyperparameter[str](
        default=MISSING_VALUE_INDICATOR,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="Value to replace missing values with.  Only applied when strategy is set to 'fill_value'"
    )

class CategoricalImputerPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive that imputes missing categorical values.  It can either replace with a constant value, or use the column mode.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '0a9936f3-7784-4697-82f0-2a5fcc744c16',
            'version': '0.1.0',
            'name': "Categorical imputer",
            'python_path': 'd3m.primitives.data_transformation.imputer.DistilCategoricalImputer',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/categorical_imputer.py',
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
                metadata_base.PrimitiveAlgorithmType.IMPUTATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:

        logger.debug(f'Running {__name__}')

        # use caller supplied columns if set
        cols = set(self.hyperparams['use_columns'])
        categorical_cols = set(inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/CategoricalData',
                                                                                 'https://metadata.datadrivendiscovery.org/types/OrdinalData')))
        if len(cols) > 0:
            cols = categorical_cols & cols
        else:
            cols = categorical_cols

        logger.debug(f'Found {len(cols)} categorical columns to evaluate')

        if len(cols) is 0:
            return base.CallResult(inputs)

        imputer = CategoricalImputer(strategy=self.hyperparams['strategy'], fill_value=self.hyperparams['fill_value'], missing_values='')
        outputs = inputs.copy()
        for c in cols:
            input_col = inputs.iloc[:,c]
            imputer.fit(input_col)
            result = imputer.transform(input_col)
            outputs.iloc[:,c] = result

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)
