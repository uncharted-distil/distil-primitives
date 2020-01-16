import os
import logging

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.primitives import utils as distil_utils
from distil.primitives.utils import SINGLETON_INDICATOR, CATEGORICALS
from distil.utils import CYTHON_DEP

import typing
import numpy as np
import pandas as pd

__all__ = ('OutputDataframePrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    output_path = hyperparams.Hyperparameter[str](
        default='tmp',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='the output path'
    )

class OutputDataframePrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):
    """
    Outputs a dataframe to the specified output path dir
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7cacc8b6-85ad-4c8f-9f75-360e0faee2b9',
            'version': '0.1.1',
            'name': "Output Dataframe",
            'python_path': 'd3m.primitives.data_transformation.data_cleaning.OutputDataframe',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/primitives/output_dataframe.py',
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
                metadata_base.PrimitiveAlgorithmType.ENCODE_BINARY,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Running {__name__}')

        out_path = self.hyperparams['output_path']

        inputs.to_csv(out_path)

        logger.debug(f'\n{inputs}')

        return base.CallResult(inputs)