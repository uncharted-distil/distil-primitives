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

from distil.primitives.prediction_expansion import PredictionExpansionPrimitive
import utils as test_utils

class PredictionExpansionPrimitiveTestCase(unittest.TestCase):

    _dataset_path_multi = path.abspath(path.join(path.dirname(__file__), 'satellite_image_dataset'))
    _dataset_path_single = path.abspath(path.join(path.dirname(__file__), 'tabular_dataset_3'))

    def test_prediction_expansion(self) -> None:
        dataset_multi = test_utils.load_dataset(self._dataset_path_multi)
        dataframe_multi = test_utils.get_dataframe(dataset_multi, 'learningData')

        dataset_single = test_utils.load_dataset(self._dataset_path_single)
        dataset_single.metadata = dataset_single.metadata.add_semantic_type(('learningData', metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')
        dataframe_single = test_utils.get_dataframe(dataset_single, 'learningData')


        hyperparams_class = \
            PredictionExpansionPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults()
        primitive = PredictionExpansionPrimitive(hyperparams=hyperparams)
        result_dataframe = primitive.produce(inputs=dataframe_single, reference=dataframe_multi).value

        # verify the output
        self.assertListEqual(list(result_dataframe.shape), [24, 2])
        self.assertEqual(result_dataframe.iloc[0, 1], 'whiskey')
        self.assertEqual(result_dataframe.iloc[13, 1], 'tango')
