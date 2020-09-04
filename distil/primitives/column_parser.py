import hashlib
import os
import typing

import numpy as np
import pandas as pd

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP

import version

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    error_handling = hyperparams.Enumeration[str](
        default='coerce',
        values=('ignore', 'raise', 'coerce'),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Setting to deal with error when converting a column to numeric value.",
    )

class ColumnParserPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    A primitive which parses columns and sets the appropriate dtypes according to it's respective metadata.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'e8e78214-9770-4c26-9eae-a45bd0ede91a',
            'version': version.__version__,
            'name': 'Column Parser',
            'python_path': 'd3m.primitives.data_transformation.DistilColumnParser',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:vkorapaty@uncharted.software',
                'uris': ['https://gitlab.com/uncharted-distil/distil-primitives']
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION
        }
    )

    def produce(self, *,
                inputs: container.DataFrame,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.DataFrame]:
        cols = self._get_columns(inputs.metadata)
        outputs = inputs.copy()

        for col in cols:
            column_metadata = inputs.metadata.query((metadata_base.ALL_ELEMENTS, col))
            semantic_types = column_metadata.get('semantic_types', [])
            if 'http://schema.org/Boolean' in semantic_types or 'http://schema.org/Float' in semantic_types or 'http://schema.org/Integer' in semantic_types:
                outputs[outputs.columns[col]] = pd.to_numeric(outputs[outputs.columns[col]], errors=self.hyperparams['error_handling'])

        return base.CallResult(outputs)

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return True

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can parsed. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use
