import logging
import os
from typing import Dict, Optional

import pandas as pd
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.primitives import utils as primitive_utils
from distil.utils import CYTHON_DEP
import version

__all__ = ('AudioTransferPrimitive',)

logger = logging.getLogger(__name__)

Inputs = container.List
Outputs = container.DataFrame

# lazy load pretrained audio due to lengthy import time
_pretrained_audio = primitive_utils.lazy_load("distil.modeling.pretrained_audio")

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

class AudioTransferPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive that converts an input audio waveform to a vector of VGGish features.

    """

    _VOLUME_KEY = 'vggish_model'
    _audio_semantic = ('http://schema.org/AudioObject',)

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f2f149c8-a984-4f5b-8a9b-2f13ee0cf16d',
            'version': version.__version__,
            'name': "Audio Transfer",
            'python_path': 'd3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/audio_transfer.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                    "type": "FILE",
                    "key": _VOLUME_KEY,
                    "file_uri": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth",
                    "file_digest": "10086976245803799d9194e9a73d9b6c1549c71d1b80106f5cade5608a561f4b",
                }, {
                    'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                    'package': 'build-essential',
                    'version': '12.4ubuntu1',
                }, {
                    'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                    'package': 'libcap-dev',
                    'version': '1:2.25-1.1',
                }, {
                    # "python-prctl" requires "build-essential" and "libcap-dev". We list it here instead of
                    # "setup.py" to not have to list these system dependencies for every common primitive (because
                    # we cannot assure this primitive annotation gets installed first).
                    'type': metadata_base.PrimitiveInstallationType.PIP,
                    'package': 'python-prctl',
                    'version': '1.7',
                },
                CYTHON_DEP,
                {
                    'type': metadata_base.PrimitiveInstallationType.PIP,
                    'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    )
                },
            ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        },
    )

    _audio_set: Optional[_pretrained_audio.AudiosetModel] = None

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int=0,
                 volumes: Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        if volumes is None:
            raise ValueError('volumes cannot be None')

    def _transform_inputs(self, inputs):
        feats = []
        for col_name in self.use_column_names:
            feats += self._audio_set._featurize(inputs[self.use_column_names[i]])
        audio_vecs = pd.DataFrame(feats.tolist())
        audio_vecs.columns = ['v{}'.format(i) for i in range(0, audio_vecs.shape[1])]

        return container.DataFrame(audio_vecs) # TODO: fix index setup

    def _get_use_column_indices(self, inputs_metadata):
        use_columns_indices = self.hyperparams['use_columns']
        audio_indices = inputs_metadata.list_columns_with_semantic_types(self._audio_semantic)
        if use_columns_indices is not None and len(use_columns_indices) > 0:
            return use_columns_indices
        elif len(audio_indices) > 0:
            return audio_indices
        raise exceptions.InvalidArgumentValueError('inputs does not have audio semantic')

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

         # lazy init the audio set model
        if self._audio_set is None:
            model_path = self.volumes[self._VOLUME_KEY]
            logger.debug(f'Loading pretrained model from {model_path}')
            self._audio_set = _pretrained_audio.AudiosetModel(model_path=model_path)

        use_column_indices = self._get_use_column_indices(inputs.metadata)
        self.use_column_names = inputs.columns[use_column_indices]
        outputs = self._transform_inputs(inputs)
        logger.debug(f'Audio transfer completed on {len(outputs.columns)} samples')

        return base.CallResult(outputs)
