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

import os
import sys
import logging
import typing

import pandas as pd
from pandas.api.types import is_numeric_dtype
from d3m import container, exceptions, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP


logger = logging.getLogger(__name__)

__all__ = ('TimeSeriesBinner',)

class Hyperparams(hyperparams.Hyperparams):
    grouping_key_col = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The GroupKey column index for the time series data.'
    )
    time_col = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=''
    )
    value_cols = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Columns needed in the dataset; all other columns will be removed."
    )
    granularity = hyperparams.Enumeration[str](
        default='months',
        values=('seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The granularity of the time series timestamp values.",
    )
    binning_operation = hyperparams.Enumeration[str](
        default='sum',
        values=('sum', 'mean', 'min', 'max'),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Setting operation to bin time series data with.",
    )
    binning_size =  hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="If time is numeric, this will be the size to comebine row values.",
    )
    binning_starting_value = hyperparams.Enumeration[str](
        default='zero',
        values=('zero', 'min'),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Where to start binning intervals from. min starts from min of dataset.",
    )

class TimeSeriesBinnerPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):

    _grouping_key_semantic = ('https://metadata.datadrivendiscovery.org/types/GroupingKey',)
    _time_semantic = ('https://metadata.datadrivendiscovery.org/types/Time',)
    _target_semantic = ('https://metadata.datadrivendiscovery.org/types/Target',)
    
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '5fee7a91-b843-4636-a21e-a02bf0fd7f3a',
            'version': '0.1.0',
            'name': 'Time series binner',
            'python_path': 'd3m.primitives.data_transformation.binning.DistilTimeSeriesBinner',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:vkorapaty@uncharted.software',
                'uris': ['https://gitlab.com/uncharted-distil/distil-primitives']
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_NORMALIZATION
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION
        }
    )

    def produce(self, *,
                inputs: container.DataFrame,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.DataFrame]:

        if inputs.shape[0] == 0:
            return base.CallResult(inputs)
        # cols = distil_utils.get_operating_columns(inputs, self.hyperparams['binning_columns'], self._semantic_types)
        init_index = inputs.index
        d3m_index = inputs.columns.get_loc('d3mIndex')
        d3m_col = inputs['d3mIndex']
        group_key_index = self._get_grouping_key_index(inputs.metadata)
        time_index = self._get_time_index(inputs.metadata)
        value_indices = self._get_value_indices(inputs.metadata)
        self.time_col_name = inputs.columns[time_index]
        self.group_col_name = inputs.columns[group_key_index]
        # inputs = inputs.set_index(self.group_col_name)
        self.time_col_dtype = inputs.dtypes[self.time_col_name]
        self.value_columns = inputs.columns[value_indices]
        usable_cols = [self.group_col_name, self.time_col_name] + list(self.value_columns)
        inputs = inputs[usable_cols]

        groups = inputs.groupby(self.group_col_name, sort=False)

        outputs = pd.DataFrame()
        binned_groups = [None] * len(groups)
        group_col_values = []
        i = 0
        for group_name, group in groups:
            # group_col = group[self.group_col_name]
            timeseries_group = group.drop(columns=[self.group_col_name])

            timeseries_group = self._applyBinningOperation(timeseries_group)

            # timeseries_group.insert(loc=group_key_index, column=self.group_col_name, value=group_name)
            group_col_values += [group_name] * len(timeseries_group)
            binned_groups[i] = timeseries_group
            i += 1
        outputs = pd.concat(binned_groups)

        is_datetime_index = isinstance(outputs.index, pd.DatetimeIndex)
        if is_datetime_index:
            datetime_index = outputs.index
        if len(outputs) <= len(init_index):
            outputs = outputs.set_index(init_index[0:len(outputs)]) #if len(outputs) <= len(init_index) else list(range(0, len(outputs), 1))
            outputs.insert(loc=d3m_index, column='d3mIndex', value=d3m_col[0:len(outputs)])
        else: # assume index and d3mIndex are int
            outputs = outputs.set_index(pd.Index(range(0, len(outputs), 1)))
            d3m_new_col = container.DataFrame({'d3mIndex': range(0, len(outputs), 1)})
            outputs.insert(loc=d3m_index, column='d3mIndex', value=d3m_new_col)
        outputs.insert(loc=group_key_index, column=self.group_col_name, value=group_col_values)
        if is_datetime_index:
            outputs.insert(loc=time_index, column=self.time_col_name, value=datetime_index)
        return base.CallResult(outputs)

    def _get_grouping_key_index(self, inputs_metadata):
        group_key_index = self.hyperparams['grouping_key_col']
        if group_key_index:
            return group_key_index
        grouping_key_indices = inputs_metadata.list_columns_with_semantic_types(self._grouping_key_semantic)
        if len(grouping_key_indices) > 0:
            return grouping_key_indices[0]
        raise exceptions.InvalidArgumentValueError('no column with grouping key')

    def _get_time_index(self, inputs_metadata):
        time_index = self.hyperparams['time_col']
        if time_index:
            return time_index
        time_indices = inputs_metadata.list_columns_with_semantic_types(self._time_semantic)
        if len(time_indices) > 0:
            return time_indices[0]
        raise exceptions.InvalidArgumentValueError('no column with time')

    def _get_value_indices(self, inputs_metadata):
        value_indices = self.hyperparams['value_cols']
        if value_indices and len(value_indices) > 0:
            return value_indices
        value_indices = inputs_metadata.list_columns_with_semantic_types(self._target_semantic)
        if len(value_indices) > 0:
            return value_indices
        raise exceptions.InvalidArgumentValueError('no columns with target')

    def _granularityToRule(self):
        granularity = self.hyperparams['granularity']
        if granularity == 'seconds':
            return 'S'
        elif granularity == 'minutes':
            return 'T'
        elif granularity == 'hours':
            return 'H'
        elif granularity == 'days':
            return 'D'
        elif granularity == 'weeks':
            return 'W'
        elif granularity == 'months':
            return 'M'
        elif granularity == 'years':
            return 'A'
        raise exceptions.InvalidArgumentValueError('Given granularity argument not supported')


    def _applyBinningOperation(self, timeseries_group):
        if is_numeric_dtype(self.time_col_dtype):
            return self._applyIntegerNumericBinning(timeseries_group)
        timeseries_group = timeseries_group.set_index(pd.DatetimeIndex(timeseries_group[self.time_col_name]))
        df = timeseries_group.resample(self._granularityToRule())
        bin_oper = self.hyperparams['binning_operation']
        return getattr(df, bin_oper)()

    def _applyIntegerNumericBinning(self, timeseries_group):
        bin_oper =  self.hyperparams['binning_operation']
        binning_size = self.hyperparams['binning_size']
        (firstTime, right,) = self._get_starting_bin_value(timeseries_group)#timeseries_group[self.time_col_name][0]
        lastTime = timeseries_group[self.time_col_name].iloc[len(timeseries_group) - 1]
        amount_of_binning_numbers = int((lastTime - firstTime) / binning_size) + 1
        amount_of_binning_intervals = amount_of_binning_numbers + 1
        binning_intervals = [i * binning_size + firstTime for i in range(amount_of_binning_intervals)]
        binning_intervals[0] = binning_intervals[0] - int(right)
        timeseries_group['binned'] = pd.cut(x=timeseries_group[self.time_col_name], bins=binning_intervals, right=right)
        # print(timeseries_group, file=sys.__stdout__)
        columnsToOperation = {}
        columnsToOperation[self.time_col_name] = 'max'
        for value in self.value_columns:
            columnsToOperation[value] = bin_oper
        return timeseries_group.groupby('binned').agg(columnsToOperation).reset_index(drop=True)

    def _get_starting_bin_value(self, df):
        if self.hyperparams['binning_starting_value'] == 'zero':
            return (0, True,)
        else:
            return (df[self.time_col_name].iloc[0], False,)
