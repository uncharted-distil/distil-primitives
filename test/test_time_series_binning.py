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
import math
import pandas as pd

from common_primitives.column_parser import ColumnParserPrimitive
from d3m import container, exceptions
from d3m.metadata import base as metadata_base

from distil.primitives.time_series_binner import TimeSeriesBinnerPrimitive
from distil.primitives import utils
import utils as test_utils

class TimeSeriesBinnerPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'timeseries_resource_dataset_2'))
    # _dataset_path = '/Users/vkorapaty/data/datasets/seed_datasets_current/LL1_736_population_spawn_MIN_METADATA/TRAIN/dataset_TRAIN'

    def test_hyperparams_integer_bin(self) -> None:
        timeseries_df = self._load_data('singleGroupData', value_indices=[3], parsing_hyperparams={
                'exclude_columns': [1]
            }
        )

        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'grouping_key_col': 1,
            'time_col': 2,
            'value_cols': [3],
            'binning_starting_value': 'min'
        }))
        result = ts_binner.produce(inputs=timeseries_df).value
        self._compare_dataframes(result, [3, 4], ['species', 'day', 'count'], [['cas9_VBBA', 'cas9_VBBA', 'cas9_VBBA'], [8, 11, 16], [85920, 85574, 88357]])


    def test_no_hyperparams_integer_bin(self) -> None:
        timeseries_df = self._load_data('singleGroupData', value_indices=[3], parsing_hyperparams={
                'exclude_columns': [1]
            })

        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults())
        result = ts_binner.produce(inputs=timeseries_df).value
        self._compare_dataframes(result, [4, 4], ['species', 'day', 'count'], [['cas9_VBBA', 'cas9_VBBA', 'cas9_VBBA'], [4, 10, 15, 16], [28810, 113989, 86925, 30127]])


    def test_timestamp_downsampling_bin(self) -> None:
        timeseries_df = self._load_data('singleGroupDataTimestamp', value_indices=[3], date_time_index=2, parsing_hyperparams={
            'exclude_columns': [1, 2],
            'parse_semantic_types': ('http://schema.org/Integer', 'http://schema.org/DateTime',)
        })
        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults().replace({'granularity': 'years'}))
        timeseries_df['day'] = pd.to_datetime(timeseries_df['day'])
        result = ts_binner.produce(inputs=timeseries_df).value
        df = pd.DataFrame({'year': [2020, 2021],
                   'month': [12, 12],
                   'day': [31, 31]})
        self._compare_dataframes(result, [2, 4], ['species', 'day', 'count'], [['cas9_VBBA', 'cas9_VBBA'], pd.to_datetime(df), [142799, 117052]])


    def test_multigroupvalue_bin(self) -> None:
        timeseries_df = self._load_data('multiGroupData', value_indices=[3, 4], parsing_hyperparams={
            'exclude_columns': [1]
        })

        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'binning_size': 7,
            'binning_starting_value': 'min'
        }))
        result = ts_binner.produce(inputs=timeseries_df).value
        self._compare_dataframes(result, [6, 5], ['species', 'day', 'count', 'offspring'], [['cas9_VBBA', 'cas9_VBBA', 'cas9_FAB', 'cas9_FAB', 'cas9_JAC', 'cas9_JAC'], [10, 16, 297, 303, 62, 67], [142799, 117052, 76340, 60210, 160706, 97558], [22870, 17614, 29558, 22999, 33174, 8254]])


    def test_multigroupvalue_timestamp_upsampling_bin(self) -> None:
        timeseries_df = self._load_data('multiGroupDataTimestamp', value_indices=[3, 4], date_time_index=2, parsing_hyperparams={
            'exclude_columns': [1, 2],
            'parse_semantic_types': ('http://schema.org/Integer', 'http://schema.org/DateTime',)
        })
        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults().replace({
            'binning_operation': 'mean'
        }))
        timeseries_df['day'] = pd.to_datetime(timeseries_df['day'])
        result = ts_binner.produce(inputs=timeseries_df).value
        expected_dates = pd.DataFrame({'date': ['2020-04-30','2020-05-31','2020-06-30','2020-07-31','2020-08-31','2020-09-30','2020-10-31','2020-11-30','2020-12-31','2021-01-31','2021-02-28','2021-03-31','2021-04-30','2021-05-31','2021-06-30','2021-07-31','2021-08-31','2021-09-30','2023-01-31',
        '2023-02-28','2023-03-31','2023-04-30','2023-05-31','2023-06-30','2023-07-31','2023-08-31','2023-09-30','2023-10-31','2023-11-30','2022-02-28','2022-03-31','2022-04-30','2022-05-31','2022-06-30','2022-07-31','2022-08-31','2022-09-30','2022-10-31','2022-11-30','2022-12-31']}).astype('datetime64[ns]')
        expected_species = ['cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA','cas9_VBBA', \
        'cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB','cas9_FAB', 'cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC','cas9_JAC']
        expected_count = [28810.0,28869.0,28241.0,float('NaN'),28399.0,float('NaN'),float('NaN'),28480.0,float('NaN'),float('NaN'),28695.0,float('NaN'),28899.0,float('NaN'),29331.0,float('NaN'),float('NaN'),30127.0,15280.0,15290.0,15230.0,15270.0,15270.0,15090.0,15040.0,15000.0,float('NaN'),float('NaN'),15080.0,31988.0,32016.0,float('NaN'),32479.0,32171.0,32052.0,float('NaN'),32490.0,float('NaN'),32589.0,32479.0]
        expected_offspring = [5415.0,4151.0,7647.0,float('NaN'),4314.0,float('NaN'),float('NaN'),1343.0,float('NaN'),float('NaN'),3415.0,float('NaN'),4522.0,float('NaN'),6536.0,float('NaN'),float('NaN'),3141.0,5125.0,7654.0,7895.0,5432.0,3452.0,3654.0,6758.0,6789.0,float('NaN'),float('NaN'),5798.0,4567.0,2345.0,float('NaN'),8764.0,7644.0,9854.0,float('NaN'),2345.0,float('NaN'),2345.0,3564.0]
        self._compare_dataframes(result, [40, 5], ['species', 'day', 'count', 'offspring'], [expected_species, expected_dates['date'], expected_count, expected_offspring])


    def test_wrong_semantics(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        timeseries_df = test_utils.get_dataframe(dataset, 'singleGroupData')
        hyperparams_class = \
            TimeSeriesBinnerPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        ts_binner = TimeSeriesBinnerPrimitive(hyperparams=hyperparams_class.defaults())
        self.assertRaises(exceptions.InvalidArgumentValueError, ts_binner.produce, inputs=timeseries_df)
        timeseries_df.metadata = timeseries_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/GroupingKey')
        timeseries_df.metadata = timeseries_df.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Time')
        self.assertRaises(exceptions.InvalidArgumentValueError, ts_binner.produce, inputs=timeseries_df)
        timeseries_df.metadata = timeseries_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'https://metadata.datadrivendiscovery.org/types/Time')
        self.assertRaises(exceptions.InvalidArgumentValueError, ts_binner.produce, inputs=timeseries_df)


    def _compare_dataframes(self, result, expected_shape, expected_columns, expected_values):
        self.assertEqual(result.shape[0], expected_shape[0])
        self.assertEqual(result.shape[1], expected_shape[1])
        for col_index, column_name in enumerate(expected_columns):
            for row_index, row_value in enumerate(expected_values[col_index]):
                if isinstance(row_value, float) and math.isnan(row_value):
                    self.assertEqual(math.isnan(result[column_name].iloc[row_index]), True)
                else:
                    self.assertEqual(result[column_name].iloc[row_index], row_value)


    def _load_data(self, dataframe_name, date_time_index=None, value_indices=[], parsing_hyperparams=None):
        dataset = test_utils.load_dataset(self._dataset_path)
        timeseries_df = test_utils.get_dataframe(dataset, dataframe_name)
        self._load_semantics_into_data(timeseries_df, group_index=1, date_time_index=date_time_index, value_indices=value_indices)
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        if parsing_hyperparams:
            cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace(parsing_hyperparams))
        else:
            ccp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        return cpp.produce(inputs=timeseries_df).value


    def _load_semantics_into_data(self, df, group_index=None, date_time_index=None, value_indices=[]):

        if group_index:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, group_index), 'https://metadata.datadrivendiscovery.org/types/GroupingKey')
        if date_time_index:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, date_time_index), 'http://schema.org/DateTime')
        for value_index in value_indices:
            df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, value_index), 'https://metadata.datadrivendiscovery.org/types/Target')


if __name__ == '__main__':
    unittest.main()
