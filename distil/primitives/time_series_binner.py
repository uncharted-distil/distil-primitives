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

import logging

import pandas as pd
from d3m import container, exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams, params
from distil.utils import CYTHON_DEP


logger = logging.getLogger(__name__)

__all__ = ('TimeSeriesBinner',)

class Hyperparams(hyperparams.HyperParams):
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

class TimeSeriesBinnerPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame, container.DataFrame, Hyperparams]):

    _semantic_types = ('https://metadata.datadrivendiscovery.org/types/GroupingKey',
                        'https://metadata.datadrivendiscovery.org/types/Timeseries',
                        'https://metadata.datadrivendiscovery.org/types/TrueTarget')
    
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '',
            'version': '0.1.0',
            'name': 'Time series binner',
            'python_path': 'd3m.primitive.data_transformation.DistilTimeSeriesBinner',
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
                inputs: container.Dataset,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.Dataset]:

        if inputs.shape[0] == 0:
            return base.callResult(inputs)
        # cols = distil_utils.get_operating_columns(inputs, self.hyperparams['binning_columns'], self._semantic_types)
        init_index = inputs.index
        group_key_index = self.hyperparams['grouping_key_col']
        time_index = self.hyperparams['time_col']
        inputs.set_index(inputs.columns(time_index))
        group_col_name = inputs.columns(group_key_index)
        time_col_dtype = inputs.dtypes[self.hyperparams['time_col']]

        groups = inputs.groupby(group_col_name).groups

        outputs = pd.DataFrame()
        for group_name, group in groups:
            group_col = group[group_col_name]
            timeseries_group = group.drop(group_col_name)

            timeseries_group = self.applyBinningOperation(timeseries_group, time_col_dtype)

            timeseries_group.insert(loc=group_key_index, column=group_col_name, value=group_name)
            pd.concat([outputs, timeseries_group], ignore_index=True)
        
        outputs.set_index(init_index)
        return base.callResults(outputs)


    def granularityToRule(self):
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


    def applyBinningOperation(self, timeseries_group, time_col_dtype):
        if time_col_dtype == 'float' || time_col_type == 'int':
            return self._applyIntegerNumericBinning(timeseries_group)
        df = timeseries_group.resample(self.granularityToRule())
        bin_oper = self.hyperparams['binning_operation']
        if bin_oper == 'mean':
            return df.mean()
        elif bin_oper == 'min':
            return df.min()
        elif bin_oper == 'max':
            return df.max()
        elif bin_oper == 'sum':
            return df.sum()
        raise exceptions.InvalidArgumentValueError('Given binner operation argument not supported')

    def _applyIntegerNumericBinning(self, timeseries_group):
        binning_size = self.hyperparams['binning_size']
        n = int(len(timeseries_group) / binning_size)

        binned_df = pd.DataFrame()
        for i in range(n):

            timeseries_group.iloc[i * binning_size:(i + 1) * binning_size]
