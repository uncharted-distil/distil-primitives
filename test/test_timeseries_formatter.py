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
import csv
import sys

from d3m import container
#from d3m.primitives.data_transformation.data_cleaning import DistilTimeSeriesFormatter
from distil.primitives.timeseries_formatter import TimeSeriesFormatterPrimitive
from d3m.metadata import base as metadata_base

from distil.primitives import utils

class TimeSeriesFormatterPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'dataset'))

    def test_basic(self) -> None:
        dataset = self._load_timeseries()

        # create the time series dataset
        hyperparams_class = \
            TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_formatter = TimeSeriesFormatterPrimitive(hyperparams=hyperparams_class.defaults())
        timeseries_df = ts_formatter.produce(inputs=dataset).value
        timeseries_df.metadata = timeseries_df.metadata.generate(timeseries_df)

        # verify that we have the expected shape
        self.assertEqual(timeseries_df.shape[0], 664)
        self.assertEqual(timeseries_df.shape[1], 5)

        # check that learning metadata was copied
        name = timeseries_df.metadata.query_column_field(0, 'name')
        self.assertEqual("d3mIndex", name)
        name = timeseries_df.metadata.query_column_field(1, 'name')
        self.assertEqual("timeseries_file", name)
        name = timeseries_df.metadata.query_column_field(2, 'name')
        self.assertEqual("label", name)
        name = timeseries_df.metadata.query_column_field(3, 'name')
        self.assertEqual("time", name)
        name = timeseries_df.metadata.query_column_field(4, 'name')
        self.assertEqual("value", name)

        # verify that the d3mIndex is now marked as a multi key
        self.assertIn("https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey", timeseries_df.metadata.query_column_field(0, 'semantic_types'))

        # verify that the grouping key was added
        self.assertIn("https://metadata.datadrivendiscovery.org/types/GroupingKey", timeseries_df.metadata.query_column_field(1, 'semantic_types'))


        print(utils.metadata_to_str(timeseries_df.metadata), file=sys.__stdout__)
        print(timeseries_df, file=sys.__stdout__)

    def test_hyperparams(self) -> None:
        dataset = self._load_timeseries()

        # create the time series dataset
        hyperparams_class = \
            TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_formatter = TimeSeriesFormatterPrimitive(hyperparams=hyperparams_class.defaults().replace(
            {
                'main_resource_id': 'learningData',
                'file_col_index': 1
            })
        )
        timeseries_df = ts_formatter.produce(inputs=dataset).value
        timeseries_df.metadata = timeseries_df.metadata.generate(timeseries_df)

        # verify that we have the expected shape
        self.assertEqual(timeseries_df.shape[0], 664)
        self.assertEqual(timeseries_df.shape[1], 5)

    @classmethod
    def _load_timeseries(cls) -> container.Dataset:
        dataset_doc_path = path.join(cls._dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))

        return dataset

if __name__ == '__main__':
    unittest.main()
