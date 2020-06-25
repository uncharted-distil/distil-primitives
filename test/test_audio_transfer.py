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

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'audio_dataset_1'))

    def test_basic(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        audio_loader_hyperparams = AudioDatasetLoaderPrimitive.metadata.get_hyperparams()
        audio_loader_primitive = AudioDatasetLoaderPrimitive(hyperparams=audio_loader_hyperparams.defaults())
        audio_df = audio_loader_primitive.produce_collection(inputs=dataset).value

        audio_transfer_hyperparams = AudioTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = AudioTransferPrimitive.metadata.get_volumes()
        volumes = {primitive_volumes[0]['key']: os.getenv('D3MSTATICDIR') + '/' + primitive_volumes[0]['file_digest']}
        audio_transfer_primitive = AudioTransferPrimitive(hyperparams=audio_transfer_hyperparams.defaults().replace({'use_columns': [0]}), volumes=volumes)

        result = audio_transfer_primitive.produce(inputs=audio_df).value

    def test_no_hyperparams_semantic_type(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults())
        df = dataframe_primitive.produce(inputs=dataset).value

        audio_df.metadata = audio_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS,0), 'http://schema.org/AudioObject')

        audio_transfer_hyperparams = AudioTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = AudioTransferPrimitive.metadata.get_volumes()
        volumes = {primitive_volumes[0]['key']: os.getenv('D3MSTATICDIR') + '/' + primitive_volumes[0]['file_digest']}
        audio_transfer_primitive = AudioTransferPrimitive(hyperparams=audio_transfer_hyperparams.defaults(), volumes=volumes)

        result = audio_transfer_primitive.produce(inputs=audio_df).value

    def test_no_hyperparams_no_semantic_type(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        audio_loader_hyperparams = AudioDatasetLoaderPrimitive.metadata.get_hyperparams()
        audio_loader_primitive = AudioDatasetLoaderPrimitive(hyperparams=audio_loader_hyperparams)
        audio_df = audio_loader_primitive.produce(inputs=df).value
        images.metadata = images.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, images.metadata.get_column_index_from_column_name('audio')), 'http://schema.org/AudioObject')

        audio_transfer_hyperparams = AudioTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = AudioTransferPrimitive.metadata.get_volumes()
        volumes = {primitive_volumes[0]['key']: os.getenv('D3MSTATICDIR') + '/' + primitive_volumes[0]['file_digest']}
        audio_transfer_primitive = AudioTransferPrimitive(hyperparams=audio_transfer_hyperparams.defaults(), volumes=volumes)

        result = audio_transfer_primitive.produce(inputs=audio_df).value


if __name__ == '__main__':
    unittest.main()
