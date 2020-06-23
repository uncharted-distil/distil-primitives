import logging
import os
import typing
from typing import Dict, Optional

import pandas as pd
from PIL import Image
from d3m import container, utils, exceptions
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult
from distil.utils import CYTHON_DEP
from distil.utils import Img2Vec
import version

__all__ = ('ImageTransferPrimitive',)

logger = logging.getLogger(__name__)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    filename_col = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The filname column index for image data.'
    )

class ImageTransferPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that converts an input image to a vector of 512 numerical features.
    """

    _VOLUME_KEY = 'resnet18-5c106cde'
    _image_semantic = ('https://schema.org/ImageObject',)

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '782e261e-8e23-4184-9258-5a412c9b32d4',
            'version': version.__version__,
            'name': "Image Transfer",
            'python_path': 'd3m.primitives.feature_extraction.image_transfer.DistilImageTransfer',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/image_transfer.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [CYTHON_DEP,
                {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
                },
                {
                    "type": "FILE",
                    "key": _VOLUME_KEY,
                    "file_uri": "http://public.datadrivendiscovery.org/resnet18-5c106cde.pth",
                    "file_digest": "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8",
                }
            ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,

        },
    )

    # class instance to avoid unnecessary re-init
    _model: Optional[Img2Vec] = None

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int=0,
                 volumes: Optional[Dict[str, str]]=None) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        if volumes is None:
            raise ValueError('volumes cannot be None')

    def _img_to_vec(self, image_array):
        image_array = image_array.squeeze()
        return self._model.get_vec(Image.fromarray(image_array).convert('RGB'))

    def _transform_inputs(self, inputs):
        result = inputs.copy()

        result['image_vec'] = (
            result[self.filename_col]
                .apply(lambda image_file: self._img_to_vec(image_file))) #self.img2vec.get_vec(image_file))

        df = pd.DataFrame(result['image_vec'].values.tolist())
        df.columns = ['v{}'.format(i) for i in range(0, df.shape[1])]

        return container.DataFrame(df, generate_metadata=True)

    def _get_filename_column_index(self, inputs_metadata):
        filename_col_index = self.hyperparams['filename_col']
        image_indices = inputs_metadata.list_columns_with_semantic_types(self._image_semantic)
        if filename_col_index is not None:
            # if filename_col_index not in image_indices:
            #     raise exceptions.InvalidArgumentValueError('column with index ' + str(filename_col_index) + ' does not have image semantic')
            return filename_col_index
        elif len(image_indices) > 0:
            return image_indices[0]
        raise exceptions.InvalidArgumentValueError('inputs does not have image semantic')


    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        if self._model is None:
            model_path = self.volumes[self._VOLUME_KEY]
            logger.info(f'Loading pre-trained model from {model_path}')
            self._model = Img2Vec(model_path)
        filename_col_index = self._get_filename_column_index(inputs.metadata)
        self.filename_col = inputs.columns[filename_col_index]

        return base.CallResult(self._transform_inputs(inputs))

