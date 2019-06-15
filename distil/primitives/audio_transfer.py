import os
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult
import pandas as pd
import numpy as np
from PIL import Image

from distil.modeling.metrics import classification_metrics, regression_metrics


from distil.modeling.pretrained_audio import AudiosetModel

__all__ = ('AudioTransferPrimitive',)

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


class AudioTransferPrimitive(unsupervised_learning.UnsupervisedLearnerPrimitiveBase[container.List, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that converts an input audio waveform to a vector of VGGish features.

    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f2f149c8-a984-4f5b-8a9b-2f13ee0cf16d',
            'version': '0.1.0',
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
                    'type': metadata_base.PrimitiveInstallationType.PIP,
                    'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                }, {
                    "type": "FILE",
                    "key": "vggish_model",
                    "file_uri": "http://public.datadrivendiscovery.org/vggish_model.ckpt",
                    "file_digest": "0962b1914e3e053922d957c45bc84a78c985765641dc6bceeeb3a7d8dfecfdf6",
                }, {
                    # "python-prctl" requires "build-essential" and "libcap-dev". We list it here instead of
                    'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                    'package': 'build-essential',
                    'version': '12.4ubuntu1',
                }, {
                    'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                    'package': 'libcap-dev',
                    'version': '1:2.25-1.1',
                }
            ],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        },
    )


    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int=0,
                 volumes: Dict[str, str] = None) -> None:

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        self.volumes = volumes
        self.audio_set = AudiosetModel(model_path=self.volumes["vggish_model"])

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)

        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)


    def set_training_data(self, *, inputs: container.List) -> None:
        self._inputs = inputs


    def _transform_inputs(self, inputs):
        logger.warning('start audio transfer')

        import time

        t0 = time.time()
        logger.warning(inputs)

        feats = self.audio_set._featurize(inputs.audio)

        audio_vecs = pd.DataFrame(feats.tolist(), inputs.index)
        audio_vecs.columns = ['v{}'.format(i) for i in range(0, audio_vecs.shape[1])]

        df = container.DataFrame(audio_vecs) # TODO: fix index setup
        df.index.name = 'd3mIndex'

        logger.warning(df)

        return df

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:


        return CallResult(None)

    def produce(self, *, inputs: container.List, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        return base.CallResult(self._transform_inputs(inputs))

    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

