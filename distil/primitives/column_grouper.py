import os
from typing import List, Set, Any, Sequence
import logging

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ('ColumnGrouperPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](''),
        default=('https://metadata.datadrivendiscovery.org/types/Attribute',),
        min_size=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Semantic types to use for grouping. If any of them matches, by default.",
    )
    grouper_columns = hyperparams.List(
        elements=hyperparams.Hyperparameter[str](''),
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
    )
    target_columns = hyperparams.List(
        elements=hyperparams.Hyperparameter[str](''),
        default=[],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
    )


class Params(params.Params):
    pass


class ColumnGrouperPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame,
                                                                     container.DataFrame,
                                                                     Hyperparams]):
    """
     List Encoder takes columns that are made up out of lists and replaces them by expanding the list
    across multiple columns.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '162bd18e-e7c6-4f8c-bd5a-458b9da8ed66',
            'version': '0.1.0',
            'name': "Column Grouper",
            'python_path': 'd3m.primitives.data_transformation.column_grouper.DistilColumnGrouper',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/column_grouper.py',
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
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:
        base.PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)

    def produce(self, *, inputs: container.DataFrame,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.DataFrame]:

        logger.debug(f'Producing {__name__}')

        grouper_columns = self.hyperparams.get('grouper_columns')
        if len(grouper_columns) == 0:
            grouper_columns = self._get_column_index(inputs, 'https://metadata.datadrivendiscovery.org/types/GroupingKey')

        target_col = self.hyperparams['target_columns']
        if len(target_col) == 0:
            target_col = self._get_column_index(inputs, 'https://metadata.datadrivendiscovery.org/types/Target')

        inputs[target_col] = inputs[target_col].astype(int)
        outputs = inputs.groupby(list(grouper_columns)).mean()[target_col]

        return base.CallResult(outputs)

    def _get_column_index(self, inputs, semantic_type):
        for column_index, col_name in enumerate(inputs.columns):
            column_metadata = inputs.metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            semantic_types = column_metadata.get('semantic_types', [])
            if semantic_type in semantic_types:
                return [col_name]
        raise AttributeError(f"No columns found with semantic type {semantic_type}")