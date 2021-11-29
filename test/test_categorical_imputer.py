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
from distil.primitives.categorical_imputer import CategoricalImputerPrimitive
from distil.primitives import utils
import utils as test_utils


class CategoricalImputerPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "tabular_dataset"))

    def test_defaults(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = CategoricalImputerPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        imputer = CategoricalImputerPrimitive(hyperparams=hyperparams_class.defaults())

        result = imputer.produce(inputs=dataframe).value
        self.assertEqual(result["alpha"].iloc[2], "whiskey")
        self.assertEqual(result["bravo"].iloc[2], "whiskey")
        self.assertEqual(result["charlie"].iloc[2], "whiskey")
        self.assertEqual(result["delta"].iloc[2], utils.MISSING_VALUE_INDICATOR)

    def test_constant(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = CategoricalImputerPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        imputer = CategoricalImputerPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {
                    "strategy": "constant",
                    "fill_value": "empty",
                    "use_columns": [1, 2, 3, 4],
                }
            )
        )

        result = imputer.produce(inputs=dataframe).value
        self.assertEqual(result["alpha"].iloc[2], "empty")
        self.assertEqual(result["bravo"].iloc[2], "empty")
        self.assertEqual(result["charlie"].iloc[2], "whiskey")
        self.assertEqual(result["delta"].iloc[2], "empty")

    def test_mode(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = CategoricalImputerPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        imputer = CategoricalImputerPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"strategy": "most_frequent", "use_columns": [1]}
            )
        )

        result = imputer.produce(inputs=dataframe).value
        self.assertEqual(result["alpha"].iloc[2], "whiskey")

    def test_no_missing(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = CategoricalImputerPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        imputer = CategoricalImputerPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"strategy": "most_frequent", "use_columns": [3]}
            )
        )

        result = imputer.produce(inputs=dataframe).value
        self.assertEqual(result["charlie"].iloc[2], "whiskey")

    def test_all_missing(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = CategoricalImputerPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        imputer = CategoricalImputerPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"strategy": "constant", "use_columns": [4], "fill_value": "empty"}
            )
        )

        result = imputer.produce(inputs=dataframe).value
        self.assertEqual(result["delta"].iloc[2], "empty")

    def test_tie(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the imputer
        hyperparams_class = CategoricalImputerPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        imputer = CategoricalImputerPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"strategy": "most_frequent", "fill_value": "empty", "use_columns": [2]}
            )
        )

        result = imputer.produce(inputs=dataframe).value
        self.assertEqual(result["bravo"].iloc[2], "whiskey")


if __name__ == "__main__":
    unittest.main()
