"""
   Copyright © 2019 Uncharted Software Inc.

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
import sys

from d3m import container

from d3m.metadata import base as metadata_base
from distil.primitives.binary_encoder import BinaryEncoderPrimitive
from distil.primitives import utils
import utils as test_utils

class BinaryEncoderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'tabular_dataset'))

    def test_single_row(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, 'learningData')

        # create the imputer
        hyperparams_class = \
            BinaryEncoderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        encoder = BinaryEncoderPrimitive(hyperparams=hyperparams_class.defaults().replace({
            "min_binary": 3
        }))

        encoder.set_training_data(inputs=dataframe)
        encoder.fit()
        result = encoder.produce(inputs=dataframe.head(1)).value
        self.assertEqual(len(result.index), 1)

if __name__ == '__main__':
    unittest.main()
