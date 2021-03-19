import logging
import os
import collections
import typing
from typing import List, Optional

import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP
import version
import common_primitives
from common_primitives import dataframe_utils


__all__ = ("VectorBoundsFilterPrimitive",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    mins = hyperparams.Union[typing.Union[float, typing.Sequence[float]]](
        configuration=collections.OrderedDict(
            set=hyperparams.List(
                elements=hyperparams.Hyperparameter[float](-1),
                default=(),
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="A set of minimum values, corresponding to the vector values to filter on",
            ),
            float=hyperparams.Hyperparameter[float](0),
        ),
        default="float",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to filter on",
    )
    maxs = hyperparams.Union[typing.Union[float, typing.Sequence[float]]](
        configuration=collections.OrderedDict(
            set=hyperparams.List(
                elements=hyperparams.Hyperparameter[float](-1),
                default=(),
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="A set of minimum values, corresponding to the vector values to filter on",
            ),
            float=hyperparams.Hyperparameter[float](0),
        ),
        default="float",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to filter on",
    )
    column = hyperparams.Hyperparameter[typing.Optional[int]](
        default=None,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The indicated FloatVector column to operate on",
    )
    inclusive = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="True when values outside the range are removed; False gives the complement.",
    )
    strict = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="True when the filter bounds are strict (ie. less than), false then are not (ie. less than equal to).",
    )


class VectorBoundsFilterPrimitive(
    transformer.TransformerPrimitiveBase[
        container.DataFrame, container.DataFrame, Hyperparams
    ]
):
    """
    A primitive to filter columns with FloatVector semantics. based on the the i'th value of the mins/maxs
    list will indicate the appropriate min/max to filter out the indicated row indices. Note that the amount
    of row indices must match the amount of mins and maxs provided, otherwise the excess given indices won't
    have any filter applied on them.

    The filter assumes the mins and maxs are the same type of data. They can be of type int, list, and two
    dimensional list.

    If row_indices_list is empty, it filters on all indices.
    If the mins/maxs are an int, all values in all vectors will be filtered with those bounds.
    If the mins/maxs are a list, then it expect it to be the same length as the amount of indice lists given.
    i.e each scalar in the mins/maxs will correspond to each set of indices in row_indices_list to filter.
    If the mins/maxs are a two dimensional list, then each vector of filters in the list will correspond to
    each set of row_indices_list. In there, each i'th value in the filter vector will correspond to each i'th
    column in the vector to be filtered.
    i.e if we have the dataframe:
    d3mIndex | values
    0        | 10, 20, 30
    1        | 15, 25, 35
    2        | 40, 20, 50
    And you provide row_indices_list = [[0, 1], [2]],
    mins = [[12, 18, 31], [20, 25, 50]], maxs = [[20, 30, 40], [30, 25, 60]]
    Only row with index 1 will be returned, as row 0 has 10 < 12, and 30 < 31.
    Row 2 was filtered out because 40 > 20 and 50 > 40, 20 < 25.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "c2fa34c0-2d1b-42af-91d2-515da4a27752",
            "version": version.__version__,
            "name": "Vector bound filter",
            "python_path": "d3m.primitives.data_transformation.vector_bounds_filter.DistilVectorBoundsFilter",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/vector_filter.py",
                    "https://github.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                CYTHON_DEP,
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    _floatvector_semantic = (
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        if self.hyperparams["strict"]:
            self._min_comparison_op = lambda x, y: x > y
            self._max_comparision_op = lambda x, y: x < y
        else:
            self._min_comparison_op = lambda x, y: x >= y
            self._max_comparision_op = lambda x, y: x <= y

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[container.DataFrame]:

        vector_column = self._get_floatvector_column(inputs.metadata)
        if vector_column is None:
            return base.CallResult(inputs)

        maxs = self.hyperparams["maxs"]
        mins = self.hyperparams["mins"]

        if type(mins) == float or type(mins) == int:
            return base.CallResult(self._scalar_filter(inputs, vector_column))

        indices = inputs.index.tolist()

        mins = [float("-inf") if i == None else i for i in mins]
        maxs = [float("inf") if i == None else i for i in maxs]

        indices_to_keep = np.empty((inputs.shape[0],))

        try:
            rows = np.stack(inputs.iloc[:, vector_column], axis=0)

            filter_length = rows.shape[1]

            rows = np.logical_and(
                self._min_comparison_op(
                    rows[:, :filter_length],
                    mins,
                ),
                self._max_comparision_op(rows[:, :filter_length], maxs),
            )
            rows_to_keep = rows.sum(axis=1) == filter_length
        except ValueError as error:
            # rows had uneven length
            rows = inputs.iloc[:, vector_column]
            # get length of each vector
            vector_lengths = rows.apply(np.shape).apply(np.take, args=([0]))

            filter_lengths = vector_lengths.values
            # need this to loop over lengths array while keeping vectorised
            # apply function over rows
            count_for_ref = [0]

            def _filter_r(row, filter_lengths, mins, maxs, counter):
                # in case fewer filters than row length
                filterable_range = min(filter_lengths[counter[0]], len(mins))

                mins_for_filter = np.array(mins[:filterable_range])
                maxs_for_filter = np.array(maxs[:filterable_range])

                filtered_row = np.logical_and(
                    self._min_comparison_op(row[:filterable_range], mins_for_filter),
                    self._max_comparision_op(
                        row[:filterable_range],
                        maxs_for_filter,
                    ),
                )
                counter[0] += 1
                return filtered_row

            rows = rows.apply(
                _filter_r,
                args=(filter_lengths, mins, maxs, count_for_ref),
            )
            rows_to_keep = rows.apply(np.sum).values == filter_lengths

        if self.hyperparams["inclusive"]:
            indices_to_keep = [
                indices[j] for j in range(len(indices)) if rows_to_keep[j]
            ]
        else:
            indices_to_keep = [
                indices[j] for j in range(len(indices)) if not rows_to_keep[j]
            ]

        outputs = dataframe_utils.select_rows(inputs, indices_to_keep)

        return base.CallResult(outputs)

    def _scalar_filter(self, inputs, vector_column):
        max_value = self.hyperparams["maxs"]
        min_value = self.hyperparams["mins"]
        indices = inputs.index.tolist()

        if min_value == None:
            float("-inf")
        if max_value == None:
            float("inf")

        try:
            rows = np.stack(inputs.iloc[:, vector_column], axis=0)

            rows = np.logical_and(
                self._min_comparison_op(
                    rows,
                    min_value,
                ),
                self._max_comparision_op(rows, max_value),
            )
            rows_to_keep = rows.sum(axis=1) == rows.shape[1]
        except ValueError as error:
            rows = inputs.iloc[:, vector_column]

            def _filter_r(row, min_val, max_val):
                return np.logical_and(
                    self._min_comparison_op(
                        row,
                        min_val,
                    ),
                    self._max_comparision_op(
                        row,
                        max_val,
                    ),
                )

            rows = rows.apply(
                _filter_r,
                args=(min_value, max_value),
            )
            rows_to_keep = rows.apply(np.sum) == rows.apply(np.shape).apply(
                np.take, args=([0])
            )
        if self.hyperparams["inclusive"]:
            rows_to_keep = [indices[j] for j in range(len(indices)) if rows_to_keep[j]]
        else:
            rows_to_keep = [
                indices[j] for j in range(len(indices)) if not rows_to_keep[j]
            ]
        return dataframe_utils.select_rows(inputs, rows_to_keep)

    def _get_floatvector_column(self, inputs_metadata: metadata_base.DataMetadata):
        fv_column = self.hyperparams["column"]
        if fv_column:
            return fv_column
        fv_columns = inputs_metadata.list_columns_with_semantic_types(
            self._floatvector_semantic
        )
        if len(fv_columns) > 0:
            return fv_columns[0]
        logger.warning(
            "inputs provided contains no specified FloatVector column and lacks columns with FloatVector semantic"
        )
        return None
