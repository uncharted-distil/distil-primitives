"""
   Copyright © 2020 Uncharted Software Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from d3m.metadata import base as metadata_base

from distil.primitives.satellite_image_loader import DataFrameSatelliteImageLoaderPrimitive
import utils as test_utils

class DataFrameSatelliteImageLoaderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'satellite_image_dataset'))

    def test_band_mapping_append(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/GroupingKey')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/FileName')
        dataset.metadata = dataset.metadata.update(('0', ), {'location_base_uris': 'file:///home/ubuntu/git-projects/distil-primitives/test/satellite_image_dataset/media/'})
        dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), {'location_base_uris': ['file:///home/ubuntu/git-projects/distil-primitives/test/satellite_image_dataset/media/']})
        dataframe = test_utils.get_dataframe(dataset, 'learningData')

        hyperparams_class = \
            DataFrameSatelliteImageLoaderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults()
        loader = DataFrameSatelliteImageLoaderPrimitive(hyperparams=hyperparams)
        result_dataframe = loader.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe.shape), [1, 8])
        self.assertListEqual(list(result_dataframe.iloc[0, 7].shape), [12, 120, 120])

    def test_band_mapping_replace(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/GroupingKey')
        dataset.metadata = dataset.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/FileName')
        dataset.metadata = dataset.metadata.update(('0', ), {'location_base_uris': 'file:///home/ubuntu/git-projects/distil-primitives/test/satellite_image_dataset/media/'})
        dataset.metadata = dataset.metadata.update(('learningData', metadata_base.ALL_ELEMENTS, 1), {'location_base_uris': ['file:///home/ubuntu/git-projects/distil-primitives/test/satellite_image_dataset/media/']})
        dataframe = test_utils.get_dataframe(dataset, 'learningData')

        hyperparams_class = \
            DataFrameSatelliteImageLoaderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace({
            "return_result": "replace"
        })
        loader = DataFrameSatelliteImageLoaderPrimitive(hyperparams=hyperparams)
        result_dataframe = loader.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe.shape), [1, 7])
        self.assertListEqual(list(result_dataframe['image_file'][0].shape), [12, 120, 120])

if __name__ == '__main__':
    unittest.main()
