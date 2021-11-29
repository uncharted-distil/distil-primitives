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
import numpy as np
from d3m import container
from d3m.metadata import base as metadata_base
from distil.primitives.list_to_dataframe import ListEncoderPrimitive
from distil.primitives import utils
import utils as test_utils


class ListEncoderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "tabular_dataset_4"))

    def test_defaults(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe = ListEncoderPrimitiveTestCase._convert_lists(dataframe)

        # create the imputer
        hyperparams_class = ListEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = ListEncoderPrimitive(hyperparams=hyperparams_class.defaults())

        encoder.set_training_data(inputs=dataframe)
        encoder.fit()
        result = encoder.produce(inputs=dataframe).value
        self._assert_result(result)

    def test_col_num(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe = ListEncoderPrimitiveTestCase._convert_lists(dataframe)

        # create the imputer
        hyperparams_class = ListEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = ListEncoderPrimitive(
            hyperparams=hyperparams_class.defaults().replace({"use_columns": [1, 2]})
        )
        encoder.set_training_data(inputs=dataframe)
        encoder.fit()
        result = encoder.produce(inputs=dataframe).value
        self._assert_result(result)

    def test_get_set_params(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe = ListEncoderPrimitiveTestCase._convert_lists(dataframe)

        # create the imputer
        # create the imputer
        hyperparams_class = ListEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = ListEncoderPrimitive(
            hyperparams=hyperparams_class.defaults().replace({"use_columns": [1, 2]})
        )
        encoder.set_training_data(inputs=dataframe)
        encoder.fit()

        hyperparams = encoder.hyperparams
        params = encoder.get_params()
        encoder = ListEncoderPrimitive(hyperparams=hyperparams)
        encoder.set_params(params=params)

        result = encoder.produce(inputs=dataframe).value
        print(result)

        self._assert_result(result)

    def _assert_result(self, result: container.DataFrame) -> None:
        self.assertEqual(result["bravo_0"].iloc[0], 1)
        self.assertEqual(result["bravo_1"].iloc[0], 2)
        self.assertEqual(result["bravo_0"].iloc[4], 70)
        self.assertEqual(result["bravo_1"].iloc[4], 80)

    @staticmethod
    def _convert_lists(dataframe: container.DataFrame) -> container.DataFrame:
        # convert colum contents to numpy array of values similar to what extract semantic types would do
        for index, row in dataframe.iterrows():
            row["bravo"] = container.ndarray([int(i) for i in row["bravo"].split(",")])
        return dataframe


if __name__ == "__main__":
    unittest.main()
