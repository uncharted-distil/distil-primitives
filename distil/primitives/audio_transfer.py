#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import logging
import os
from typing import Dict, Optional

import pandas as pd
from d3m import container, utils, exceptions
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.primitives import utils as primitive_utils
from distil.utils import CYTHON_DEP
import version

__all__ = ("AudioTransferPrimitive",)

logger = logging.getLogger(__name__)

Inputs = container.DataFrame
Outputs = container.DataFrame

# lazy load pretrained audio due to lengthy import time
_pretrained_audio = primitive_utils.lazy_load("distil.modeling.pretrained_audio")


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )


class AudioTransferPrimitive(
    transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]
):
    """
    A primitive that converts an input audio waveform to a vector of VGGish features.

    """

    _VOLUME_KEY = "vggish_model"
    _AUDIO_SEMANTIC_TYPE = ("http://schema.org/AudioObject",)

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "f2f149c8-a984-4f5b-8a9b-2f13ee0cf16d",
            "version": version.__version__,
            "name": "Audio Transfer",
            "python_path": "d3m.primitives.feature_extraction.audio_transfer.DistilAudioTransfer",
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/audio_transfer.py",
                    "https://github.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                {
                    "type": "FILE",
                    "key": _VOLUME_KEY,
                    "file_uri": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth",
                    "file_digest": "10086976245803799d9194e9a73d9b6c1549c71d1b80106f5cade5608a561f4b",
                },
                CYTHON_DEP,
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        },
    )

    _audio_set: Optional[_pretrained_audio.AudiosetModel] = None

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        volumes: Dict[str, str] = None,
    ) -> None:
        super().__init__(
            hyperparams=hyperparams, random_seed=random_seed, volumes=volumes
        )
        if volumes is None:
            raise ValueError("volumes cannot be None")

    def _transform_inputs(self, inputs):
        feats = []
        for col_name in self.use_column_names:
            feats += self._audio_set._featurize(inputs[col_name]).tolist()
        audio_vecs = pd.DataFrame(feats)
        audio_vecs.columns = ["v{}".format(i) for i in range(0, audio_vecs.shape[1])]

        return container.DataFrame(audio_vecs)  # TODO: fix index setup

    def _get_use_column_indices(self, inputs_metadata):
        use_columns_indices = self.hyperparams["use_columns"]
        audio_indices = inputs_metadata.list_columns_with_semantic_types(
            self._AUDIO_SEMANTIC_TYPE
        )
        if use_columns_indices is not None and len(use_columns_indices) > 0:
            return use_columns_indices
        elif len(audio_indices) > 0:
            return audio_indices
        else:
            # Rather than failing, we'll just take the first column as a last ditch effort.  This lines
            # up with the legacy approach of using the audio loader's `produce_collection` call, which just
            # returns a single column dataframe containing the audio data.
            return [0]

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[container.DataFrame]:
        logger.debug(f"Producing {__name__}")

        # lazy init the audio set model
        if self._audio_set is None:
            model_path = self.volumes[self._VOLUME_KEY]
            logger.debug(f"Loading pretrained model from {model_path}")
            self._audio_set = _pretrained_audio.AudiosetModel(model_path=model_path)

        use_column_indices = self._get_use_column_indices(inputs.metadata)
        self.use_column_names = inputs.columns[use_column_indices]
        outputs = self._transform_inputs(inputs)
        logger.debug(f"Audio transfer completed on {len(outputs.columns)} samples")

        return base.CallResult(outputs)
