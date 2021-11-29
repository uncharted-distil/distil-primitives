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
from os import path
import sys

from d3m import container

from d3m.metadata import base as metadata_base
from distil.primitives.one_hot_encoder import OneHotEncoderPrimitive
from distil.primitives import utils
import utils as test_utils


class OneHotEncoderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "tabular_dataset"))

    def test_defaults(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = OneHotEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = OneHotEncoderPrimitive(hyperparams=hyperparams_class.defaults())

        encoder.set_training_data(inputs=dataframe)
        encoder.fit()
        result = encoder.produce(inputs=dataframe).value
        self.assertEqual(len(result.index), 5)
        self.assertEqual(
            result.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/Attribute",)
            ),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
        self.assertSequenceEqual(
            list(result.columns), ["d3mIndex"] + [f"__onehot_{i}" for i in range(10)]
        )
        self.assertSequenceEqual(
            result.dtypes.tolist(), [object] + [float for i in range(10)]
        )

    def test_get_set_params(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = OneHotEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = OneHotEncoderPrimitive(hyperparams=hyperparams_class.defaults())
        encoder.set_training_data(inputs=dataframe)
        encoder.fit()

        hyperparams = encoder.hyperparams
        params = encoder.get_params()
        encoder = OneHotEncoderPrimitive(hyperparams=hyperparams)
        encoder.set_params(params=params)

        result = encoder.produce(inputs=dataframe).value

        self.assertEqual(len(result.index), 5)
        print(result.columns)
        self.assertSequenceEqual(
            list(result.columns), ["d3mIndex"] + [f"__onehot_{i}" for i in range(10)]
        )
        self.assertSequenceEqual(
            result.dtypes.tolist(), [object] + [float for i in range(10)]
        )


if __name__ == "__main__":
    unittest.main()
