import os
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import pandas as pd
import numpy as np
from PIL import Image

from distil.modeling.metrics import classification_metrics, regression_metrics

from distil.utils import Img2Vec

__all__ = ('ImageTransferPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    metric = hyperparams.Hyperparameter[str](
        default='',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    fast = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Params(params.Params):
    pass


class ImageTransferPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that images.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '782e261e-8e23-4184-9258-5a412c9b32d4',
            'version': '0.1.0',
            'name': "Image Transfer",
            'python_path': 'd3m.primitives.data_transformation.encoder.DistilImageTransfer',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/image_transfer.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [
                {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
                },
                {
                    "type": "FILE",
                    "key": "resnet18-5c106cde",
                    "file_uri": "http://public.datadrivendiscovery.org/resnet18-5c106cde.pth",
                    "file_digest": "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
                }
            ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )


    def __init__(self, *,
                 hyperparams: Hyperparams, 
                 random_seed: int=0,
                 volumes: Dict[str, str] = None) -> None:

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        self.volumes = volumes
        self.img2vec = Img2Vec(model_path=self.volumes["resnet18-5c106cde"])
       

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)

        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)


    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs


    def _transform_inputs(self, inputs):
        result = inputs.copy()
        result['image_vec'] = (
            result['filename']
                .apply(lambda image_file: self.img2vec.get_vec(Image.fromarray(image_file))) #self.img2vec.get_vec(image_file))
        )

        df = pd.DataFrame(result['image_vec'].values.tolist())
        df.columns = ['v{}'.format(i) for i in range(0, df.shape[1])]
        df.index = result['d3mIndex']
        df.index.name = 'd3mIndex'

        return container.DataFrame(df, generate_metadata=True)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        # create dataframe to hold d3mIndex and result

        self.features_df = self._transform_inputs(self._inputs)

        logger.debug(self.features_df)
        logger.debug(self.features_df.metadata)

        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        return base.CallResult(self._transform_inputs(inputs))

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

