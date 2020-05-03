"""
   Copyright © 2018 Uncharted Software Inc.

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
from d3m.primitives.data_preprocessing.satellite_image_reader import DataFrameSatelliteImageReaderPrimitive
from d3m.metadata import base as metadata_base


class DataFrameSatelliteImageReaderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'satellite_image_dataset'))

    def test_band_mapping(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, 'learningData')

        hyperparams_class = \
            DataFrameSatelliteImageReaderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'grouping_key_column': 2
            }
        )
        loader = DataFrameSatelliteImageReaderPrimitive(hyperparams=hyperparams)
        result_dataframe = loader.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [2, 5, 3])
        self.assertListEqual(list(result_dataframe['name']), ['bravo', 'echo', 'charlie'])
        expected_ranks = [1.342857, 0.861607, 0.0]
        for i, r in enumerate(result_dataframe['rank']):
            self.assertAlmostEqual(r, expected_ranks[i], places=6)


if __name__ == '__main__':
    unittest.main()
