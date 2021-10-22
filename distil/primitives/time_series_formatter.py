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
import typing

import numpy as np
import pandas as pd  # type: ignore
from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
import version

__all__ = ("TimeSeriesFormatterPrimitive",)


class Hyperparams(hyperparams.Hyperparams):
    file_col_index = hyperparams.Hyperparameter[typing.Union[int, None]](
        default=None,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Index of column in input dataset containing time series file names."
        + "If set to None, will use the first csv filename column found.",
    )
    main_resource_id = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="ID of data resource in input dataset containing the reference to timeseries data."
        + "If set to None, will use the entry point.",
    )
    equal_length = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Wheter all files should have the same output length. Cuts everything to the smallest length. ",
    )


class TimeSeriesFormatterPrimitive(
    transformer.TransformerPrimitiveBase[
        container.Dataset, container.Dataset, Hyperparams
    ]
):
    """
    Reads the time series files from a given column in an input dataset resource into a new M x N data resource,
    where each value in timeseries occupies one of M rows. Each row has N columns, representing the union of
    the fields found in the timeseries files and in the main data resource. The loading process assumes that
    each series file has an identical set of timestamps.  The `GroupingKey` semantic type will be added to the
    column that contains the file names, and the time column will be marked with the `Time` semantic type.

    Example output::

        filename    | time      | value     | label     |
        -------------------------------------------------
        f1.csv      | 0         | 0.1       | alpha     |
        f1.csv      | 1         | 0.12      | alpha     |
        f1.csv      | 2         | 0.13      | alpha     |
        f2.csv      | 0         | 0.72      | bravo     |
        f2.csv      | 1         | 0.77      | bravo     |
        f2.csv      | 2         | 0.67      | bravo     |
    """

    _semantic_types = (
        "https://metadata.datadrivendiscovery.org/types/FileName",
        "https://metadata.datadrivendiscovery.org/types/Timeseries",
        "http://schema.org/Text",
        "https://metadata.datadrivendiscovery.org/types/Attribute",
    )
    _media_types = ("text/csv",)
    _resource_id = "learningData"

    __author__ = ("Uncharted Software",)
    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "6a1ce3ee-ee70-428b-b1ff-0490bdb23023",
            "version": version.__version__,
            "name": "Time series formatter",
            "python_path": "d3m.primitives.data_transformation.time_series_formatter.DistilTimeSeriesFormatter",
            "keywords": ["series", "reader", "csv"],
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/time_series_formatter.py",
                    "https://gitlab.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=d3m_utils.current_git_commit(
                            os.path.dirname(__file__)
                        ),
                    ),
                },
            ],
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
            ],
            "supported_media_types": _media_types,
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def produce(
        self,
        *,
        inputs: container.Dataset,
        timeout: float = None,
        iterations: int = None
    ) -> base.CallResult[container.Dataset]:

        # if this is a single resource dataset we don't need to reformat it
        if len(inputs) < 2:
            return base.CallResult(inputs)

        # find the main resource if supplied, infer if not
        main_resource_id, main_resource = base_utils.get_tabular_resource(
            inputs, self.hyperparams["main_resource_id"]
        )
        if main_resource_id is None:
            raise exceptions.InvalidArgumentValueError("no main resource specified")

        # find the csv file column resource if supplied, infer if not
        file_index = self.hyperparams["file_col_index"]
        if file_index is not None:
            if not self._is_csv_file_column(
                inputs.metadata, main_resource_id, file_index
            ):
                raise exceptions.InvalidArgumentValueError(
                    "column idx="
                    + str(file_index)
                    + " from does not contain csv file names"
                )
        else:
            file_index = self._find_csv_file_column(inputs.metadata, main_resource_id)
            if file_index is None:
                raise exceptions.InvalidArgumentValueError(
                    "no column from contains csv file names"
                )

        # generate the long form timeseries data
        base_path = self._get_base_path(inputs.metadata, main_resource_id, file_index)
        csv_paths = [
            os.path.join(base_path, local_path)
            for local_path in inputs[main_resource_id].iloc[:, file_index]
        ]
        new_dfs = [pd.read_csv(path) for path in csv_paths]
        original_dfs = [
            pd.DataFrame(
                np.tile(row, (df.shape[0], 1)),
                columns=inputs[main_resource_id].columns,
                index=df.index,
            )
            for row, df in zip(inputs[main_resource_id].values, new_dfs)
        ]
        combined_dfs = [
            original_df.join(new_df)
            for original_df, new_df in zip(original_dfs, new_dfs)
        ]
        output_data = pd.concat(combined_dfs)
        timeseries_dataframe = container.DataFrame(output_data)
        timeseries_dataframe.reset_index(drop=True, inplace=True)

        # make sure that all timeseries have the same length, most downstream tasks will appreciate this.
        if self.hyperparams["equal_length"]:
            min_length = (
                timeseries_dataframe.groupby(timeseries_dataframe.columns[file_index])
                .count()
                .min()
                .values[0]
            )
            group_count = timeseries_dataframe.groupby(
                timeseries_dataframe.columns[file_index]
            ).cumcount()
            timeseries_dataframe = timeseries_dataframe.assign(group_count=group_count)
            timeseries_dataframe = timeseries_dataframe[
                timeseries_dataframe["group_count"] < min_length
            ]
            timeseries_dataframe = timeseries_dataframe.drop(["group_count"], axis=1)

        # create a dataset to hold the result
        timeseries_dataset = container.Dataset(
            {self._resource_id: timeseries_dataframe}, generate_metadata=True
        )
        timeseries_dataset.metadata = timeseries_dataset.metadata.update(
            (), {"id": inputs.metadata.query(())["id"]}
        )
        timeseries_dataset.metadata = timeseries_dataset.metadata.update(
            (), {"digest": inputs.metadata.query(())["digest"]}
        )

        # copy main resource column metadata to timeseries dataframe
        num_main_resource_cols = inputs.metadata.query(
            (main_resource_id, metadata_base.ALL_ELEMENTS)
        )["dimension"]["length"]
        for i in range(num_main_resource_cols):
            source = inputs.metadata.query(
                (main_resource_id, metadata_base.ALL_ELEMENTS, i)
            )
            timeseries_dataset.metadata = timeseries_dataset.metadata.update_column(
                i, source, at=(self._resource_id,)
            )

        # remove the foreign key entry from the filename column if it exists
        metadata = dict(
            timeseries_dataset.metadata.query(
                (self._resource_id, metadata_base.ALL_ELEMENTS, file_index)
            )
        )
        metadata["foreign_key"] = metadata_base.NO_VALUE
        timeseries_dataset.metadata = timeseries_dataset.metadata.update(
            (self._resource_id, metadata_base.ALL_ELEMENTS, file_index), metadata
        )

        # copy timeseries column metadata to timeseries if its available in the metadata (which is not necssarily true anymore)
        source = self._find_timeseries_metadata(inputs)
        i = 0
        start_idx = 0
        if source is not None:
            for col_info in source["file_columns"]:
                timeseries_dataset.metadata = timeseries_dataset.metadata.update_column(
                    i + num_main_resource_cols, col_info, at=(self._resource_id,)
                )
                i += 1
            # flag all other columns as attributes
            start_idx = i + num_main_resource_cols
        else:
            # loop over the appended time series columns
            start_idx = original_dfs[0].shape[1]

        for i in range(start_idx, timeseries_dataframe.shape[1]):
            timeseries_dataset.metadata = timeseries_dataset.metadata.add_semantic_type(
                (self._resource_id, metadata_base.ALL_ELEMENTS, i),
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
            struct_type = timeseries_dataset.metadata.query(
                (self._resource_id, metadata_base.ALL_ELEMENTS, i)
            )["structural_type"]
            if struct_type == np.float64:
                timeseries_dataset.metadata = (
                    timeseries_dataset.metadata.add_semantic_type(
                        (self._resource_id, metadata_base.ALL_ELEMENTS, i),
                        "http://schema.org/Float",
                    )
                )
            elif struct_type == np.int64:
                timeseries_dataset.metadata = (
                    timeseries_dataset.metadata.add_semantic_type(
                        (self._resource_id, metadata_base.ALL_ELEMENTS, i),
                        "http://schema.org/Integer",
                    )
                )
            else:
                timeseries_dataset.metadata = (
                    timeseries_dataset.metadata.add_semantic_type(
                        (self._resource_id, metadata_base.ALL_ELEMENTS, i),
                        "http://schema.org/Text",
                    )
                )

        # mark the filename column as a grouping key
        timeseries_dataset.metadata = timeseries_dataset.metadata.add_semantic_type(
            (self._resource_id, metadata_base.ALL_ELEMENTS, file_index),
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        )

        # mark the d3mIndex as a primary multi-key since there are now multiple instances of the value present
        primary_index_col = (
            timeseries_dataset.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/PrimaryKey",),
                at=(self._resource_id,),
            )
        )
        timeseries_dataset.metadata = timeseries_dataset.metadata.remove_semantic_type(
            (self._resource_id, metadata_base.ALL_ELEMENTS, primary_index_col[0]),
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        )
        timeseries_dataset.metadata = timeseries_dataset.metadata.add_semantic_type(
            (self._resource_id, metadata_base.ALL_ELEMENTS, primary_index_col[0]),
            "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey",
        )

        return base.CallResult(timeseries_dataset)

    @classmethod
    def _find_csv_file_column(
        cls, inputs_metadata: metadata_base.DataMetadata, res_id: str
    ) -> typing.Optional[int]:
        indices = inputs_metadata.list_columns_with_semantic_types(
            cls._semantic_types, at=(res_id,)
        )
        for i in indices:
            if cls._is_csv_file_column(inputs_metadata, res_id, i):
                return i
        return None

    @classmethod
    def _is_csv_file_column(
        cls, inputs_metadata: metadata_base.DataMetadata, res_id: str, column_index: int
    ) -> bool:
        # check to see if a given column is a file pointer that points to a csv file
        column_metadata = inputs_metadata.query(
            (res_id, metadata_base.ALL_ELEMENTS, column_index)
        )

        if not column_metadata or column_metadata["structural_type"] != str:
            return False

        # check if a foreign key exists
        if "foreign_key" not in column_metadata:
            return False

        ref_col_index = column_metadata["foreign_key"]["column_index"]
        ref_res_id = column_metadata["foreign_key"]["resource_id"]

        return cls._is_csv_file_reference(inputs_metadata, ref_res_id, ref_col_index)

    @classmethod
    def _is_csv_file_reference(
        cls, inputs_metadata: metadata_base.DataMetadata, res_id: int, column_index: int
    ) -> bool:
        # check to see if the column is a csv resource
        column_metadata = inputs_metadata.query(
            (res_id, metadata_base.ALL_ELEMENTS, column_index)
        )

        if not column_metadata or column_metadata["structural_type"] != str:
            return False

        semantic_types = column_metadata.get("semantic_types", [])
        media_types = column_metadata.get("media_types", [])

        semantic_types_set = set(semantic_types)
        _semantic_types_set = set(cls._semantic_types)

        return bool(semantic_types_set.intersection(_semantic_types_set)) and set(
            cls._media_types
        ).issubset(media_types)

    @classmethod
    def _find_timeseries_metadata(
        cls, dataset: container.Dataset
    ) -> typing.Optional[metadata_base.DataMetadata]:
        # loop over the dataset to find the resource that contains the timeseries file col info
        for resource_id, resource in dataset.items():
            metadata = dataset.metadata.query((resource_id, "ALL_ELEMENTS", 0))
            if "file_columns" in metadata:
                return metadata
        return None

    def _get_base_path(
        self,
        inputs_metadata: metadata_base.DataMetadata,
        res_id: str,
        column_index: int,
    ) -> str:
        # get the base uri from the referenced column
        column_metadata = inputs_metadata.query(
            (res_id, metadata_base.ALL_ELEMENTS, column_index)
        )

        ref_col_index = column_metadata["foreign_key"]["column_index"]
        ref_res_id = column_metadata["foreign_key"]["resource_id"]

        return inputs_metadata.query(
            (ref_res_id, metadata_base.ALL_ELEMENTS, ref_col_index)
        )["location_base_uris"][0]

    def _get_ref_resource(
        self,
        inputs_metadata: metadata_base.DataMetadata,
        res_id: str,
        column_index: int,
    ) -> str:
        # get the referenced resource from the referenced column
        column_metadata = inputs_metadata.query(
            (res_id, metadata_base.ALL_ELEMENTS, column_index)
        )
        ref_res_id = column_metadata["foreign_key"]["resource_id"]

        return ref_res_id
