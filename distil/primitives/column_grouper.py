import os
from typing import List, Set, Any, Sequence
import logging

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

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
        default=['timeseries_file'],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
    )
    target_columns = hyperparams.List(
        elements=hyperparams.Hyperparameter[str](''),
        default=['label'],
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
    )


class Params(params.Params):
    pass


class ColumnGrouperPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[
                                 container.DataFrame, container.DataFrame, Params, Hyperparams]):
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

        grouper_columns = hyperparams['grouper_col']
        target_col = hyperparams['target_col']

        inputs[target_col] = inputs[target_col].astype(int)
        outputs = inputs.groupby([list(grouper_columns)]).mean()[target_col]

        return base.CallResult(outputs)
