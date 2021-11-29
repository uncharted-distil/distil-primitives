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

import unittest
import os
from os import path
import sys

from d3m import container
from d3m.metadata import base as metadata_base
from common_primitives import dataset_to_dataframe
from distil.primitives.audio_transfer import AudioTransferPrimitive
from distil.primitives.audio_reader import AudioDatasetLoaderPrimitive
import utils as test_utils


class AudioTransferPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "audio_dataset_1"))

    def test_basic(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        audio_loader_hyperparams = (
            AudioDatasetLoaderPrimitive.metadata.get_hyperparams()
        )
        audio_loader_primitive = AudioDatasetLoaderPrimitive(
            hyperparams=audio_loader_hyperparams.defaults()
        )
        audio_df = audio_loader_primitive.produce_collection(inputs=dataset).value

        audio_transfer_hyperparams = AudioTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = AudioTransferPrimitive.metadata.get_volumes()
        volumes = {
            primitive_volumes[0]["key"]: os.getenv("D3MSTATICDIR")
            + "/"
            + primitive_volumes[0]["file_digest"]
        }
        audio_transfer_primitive = AudioTransferPrimitive(
            hyperparams=audio_transfer_hyperparams.defaults().replace(
                {"use_columns": [0]}
            ),
            volumes=volumes,
        )

        result = audio_transfer_primitive.produce(inputs=audio_df).value

    def test_no_hyperparams_semantic_type(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        dataframe_hyperparams_class = (
            dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        )
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataframe_hyperparams_class.defaults()
        )
        audio_df = dataframe_primitive.produce(inputs=dataset).value

        audio_df.metadata = audio_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0), "http://schema.org/AudioObject"
        )

        audio_transfer_hyperparams = AudioTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = AudioTransferPrimitive.metadata.get_volumes()
        volumes = {
            primitive_volumes[0]["key"]: os.getenv("D3MSTATICDIR")
            + "/"
            + primitive_volumes[0]["file_digest"]
        }
        audio_transfer_primitive = AudioTransferPrimitive(
            hyperparams=audio_transfer_hyperparams.defaults(), volumes=volumes
        )

        result = audio_transfer_primitive.produce(inputs=audio_df).value

    def test_no_hyperparams_no_semantic_type(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        auto_loader_hyperparams_class = (
            AudioDatasetLoaderPrimitive.metadata.get_hyperparams()
        )
        audio_loader_hyperparams = auto_loader_hyperparams_class.defaults()
        audio_loader_primitive = AudioDatasetLoaderPrimitive(
            hyperparams=audio_loader_hyperparams
        )
        audio_df = audio_loader_primitive.produce(inputs=dataset).value

        audio_transfer_hyperparams = AudioTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = AudioTransferPrimitive.metadata.get_volumes()
        volumes = {
            primitive_volumes[0]["key"]: os.getenv("D3MSTATICDIR")
            + "/"
            + primitive_volumes[0]["file_digest"]
        }
        audio_transfer_primitive = AudioTransferPrimitive(
            hyperparams=audio_transfer_hyperparams.defaults(), volumes=volumes
        )

        result = audio_transfer_primitive.produce(inputs=audio_df).value


if __name__ == "__main__":
    unittest.main()
