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

from common_primitives.column_parser import ColumnParserPrimitive
from d3m import container
from d3m.metadata import base as metadata_base

from distil.primitives.time_series_binner import TimeSeriesBinnerPrimitive
from distil.primitives import utils
import utils as test_utils

class TimeSeriesBinnerPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'timeseries_resource_dataset_2'))
    # _dataset_path = '/Users/vkorapaty/data/datasets/seed_datasets_current/LL1_736_population_spawn_MIN_METADATA/TRAIN/dataset_TRAIN'

    def _load_semantics_into_data(self, df, group_index=None, time_index=None, value_indices=[]):

        if group_index:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, group_index), 'https://metadata.datadrivendiscovery.org/types/GroupingKey')
        if time_index:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, time), 'https://metadata.datadrivendiscovery.org/types/Time')
        for value_index in value_indices:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, value_index), 'https://metadata.datadrivendiscovery.org/types/Target')

    def _load_data(self, dataframe_name):
        dataset = test_utils.load_dataset(self._dataset_path)
        timeseries_df = test_utils.get_dataframe(dataset, dataframe_name)
        self._load_semantics_into_data(timeseries_df, group_index=1, value_indices=[3])
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'exclude_columns': [1]
        }))
        return cpp.produce(inputs=timeseries_df).value

    def test_hyperparams_integer_bin(self) -> None:
        timeseries_df = self._load_data('singleGroupData')

        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'grouping_key_col': 1,
            'time_col': 2,
            'value_cols': [3],
            'binning_starting_value': 'min'
        }))
        result = ts_binner.produce(inputs=timeseries_df).value
        expected_rows = 3
        self.assertEqual(result.shape[0], expected_rows)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(result['count'].iloc[0], 85920)
        self.assertEqual(result['count'].iloc[1], 85574)
        self.assertEqual(result['count'].iloc[2], 88357)
        self.assertEqual(result['day'].iloc[0], 8)
        self.assertEqual(result['day'].iloc[1], 11)
        self.assertEqual(result['day'].iloc[2], 16)
        for i in range(expected_rows):
            self.assertEqual(result['species'][i], 'cas9_VBBA')


    def test_no_hyperparams_integer_bin(self) -> None:
        timeseries_df = self._load_data('singleGroupData')

        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults())
        result = ts_binner.produce(inputs=timeseries_df).value
        expected_rows = 3
        self.assertEqual(result.shape[0], expected_rows)
        self.assertEqual(result.shape[1], 4)
        self.assertEqual(result['count'].iloc[0], 85920)
        self.assertEqual(result['count'].iloc[1], 85574)
        self.assertEqual(result['count'].iloc[2], 88357)
        self.assertEqual(result['day'].iloc[0], 8)
        self.assertEqual(result['day'].iloc[1], 11)
        self.assertEqual(result['day'].iloc[2], 16)
        for i in range(expected_rows):
            self.assertEqual(result['species'][i], 'cas9_VBBA')


if __name__ == '__main__':
    unittest.main()
