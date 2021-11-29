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
import csv
import sys

from d3m import container
from d3m.metadata import base as metadata_base

from distil.primitives.time_series_formatter import TimeSeriesFormatterPrimitive
from distil.primitives import utils

import utils as test_utils


class TimeSeriesFormatterPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(
        path.join(path.dirname(__file__), "timeseries_resource_dataset")
    )
    _dataset_2_path = path.abspath(
        path.join(path.dirname(__file__), "timeseries_resource_dataset_2")
    )
    _resource_id = "learningData"

    def test_basic(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)

        # create the time series dataset
        hyperparams_class = TimeSeriesFormatterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        ts_formatter = TimeSeriesFormatterPrimitive(
            hyperparams=hyperparams_class.defaults()
        )
        timeseries_dataset = ts_formatter.produce(inputs=dataset).value
        timeseries_df = timeseries_dataset[self._resource_id]

        # verify that ID and digest is present - runtime will fail without when it tries to execute the sub-pipeline
        root_metadata = timeseries_dataset.metadata.query(())
        self.assertIn("id", root_metadata)
        self.assertIn("digest", root_metadata)

        # verify that we have the expected shape
        self.assertEqual(timeseries_df.shape[0], 664)
        self.assertEqual(timeseries_df.shape[1], 5)

        # check that learning metadata was copied
        name = timeseries_dataset.metadata.query_column_field(
            0, "name", at=(self._resource_id,)
        )
        self.assertEqual("d3mIndex", name)
        name = timeseries_dataset.metadata.query_column_field(
            1, "name", at=(self._resource_id,)
        )
        self.assertEqual("timeseries_file", name)
        name = timeseries_dataset.metadata.query_column_field(
            2, "name", at=(self._resource_id,)
        )
        self.assertEqual("label", name)
        name = timeseries_dataset.metadata.query_column_field(
            3, "name", at=(self._resource_id,)
        )
        self.assertEqual("time", name)
        name = timeseries_dataset.metadata.query_column_field(
            4, "name", at=(self._resource_id,)
        )
        self.assertEqual("value", name)

        # verify that the d3mIndex is now marked as a multi key
        self.assertIn(
            "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
            timeseries_dataset.metadata.query_column_field(
                0, "semantic_types", at=(self._resource_id,)
            ),
        )
        self.assertIn(
            "http://schema.org/Integer",
            timeseries_dataset.metadata.query_column_field(
                0, "semantic_types", at=(self._resource_id,)
            ),
        )

        # verify that the grouping key was added
        self.assertIn(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
            timeseries_dataset.metadata.query_column_field(
                1, "semantic_types", at=(self._resource_id,)
            ),
        )
        self.assertIn(
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            timeseries_dataset.metadata.query_column_field(
                1, "semantic_types", at=(self._resource_id,)
            ),
        )
        self.assertIn(
            "http://schema.org/Text",
            timeseries_dataset.metadata.query_column_field(
                1, "semantic_types", at=(self._resource_id,)
            ),
        )

        # verify that the label column is of type unknown
        self.assertIn(
            "https://metadata.datadrivendiscovery.org/types/UnknownType",
            timeseries_dataset.metadata.query_column_field(
                2, "semantic_types", at=(self._resource_id,)
            ),
        )

        # verify that data columns have correct semantic types
        self.assertIn(
            "https://metadata.datadrivendiscovery.org/types/Time",
            timeseries_dataset.metadata.query_column_field(
                3, "semantic_types", at=(self._resource_id,)
            ),
        )
        self.assertIn(
            "http://schema.org/Integer",
            timeseries_dataset.metadata.query_column_field(
                3, "semantic_types", at=(self._resource_id,)
            ),
        )

        # verify that data columns have correct semantic types
        self.assertIn(
            "http://schema.org/Float",
            timeseries_dataset.metadata.query_column_field(
                4, "semantic_types", at=(self._resource_id,)
            ),
        )
        self.assertIn(
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            timeseries_dataset.metadata.query_column_field(
                4, "semantic_types", at=(self._resource_id,)
            ),
        )

    def test_hyperparams(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)

        # create the time series dataset
        hyperparams_class = TimeSeriesFormatterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        ts_formatter = TimeSeriesFormatterPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"main_resource_id": "learningData", "file_col_index": 1}
            )
        )
        timeseries_df = ts_formatter.produce(inputs=dataset).value[self._resource_id]

        # verify that we have the expected shape
        self.assertEqual(timeseries_df.shape[0], 664)
        self.assertEqual(timeseries_df.shape[1], 5)


if __name__ == "__main__":
    unittest.main()
