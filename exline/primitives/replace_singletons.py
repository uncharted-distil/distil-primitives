import os
import logging

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from exline.primitives import utils
from exline.primitives.utils import SINGLETON_INDICATOR, CATEGORICALS

import typing
import numpy as np
import pandas as pd

__all__ = ('ReplaceSingletonsPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )


class ReplaceSingletonsPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    Replaces category members with a count of one with a shared singleton token value.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7cacc8b6-85ad-4c8f-9f75-360e0faee2b8',
            'version': '0.1.0',
            'name': "Replace singeltons",
            'python_path': 'd3m.primitives.data_transformation.data_cleaning.ExlineReplaceSingletons',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/primitives/replace_singletons.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=d3m-exline'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ENCODE_BINARY,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Running {__name__}')

        # set values that only occur once to a special token
        outputs = inputs.copy()

        # determine columns to operate on
        cols = utils.get_operating_columns(inputs, self.hyperparams['use_columns'], CATEGORICALS)

        for c in cols:
            vcs = pd.value_counts(list(inputs.iloc[:,c]))
            singletons = set(vcs[vcs == 1].index)
            if singletons:
                outputs.iloc[:,c][outputs.iloc[:,c].isin(singletons)] = SINGLETON_INDICATOR

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)