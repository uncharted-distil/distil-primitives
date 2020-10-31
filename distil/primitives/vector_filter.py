import logging
import os
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
    row_indices_list = hyperparams.Set(
        elements=hyperparams.Set(
            elements=hyperparams.Hyperparameter[int](-1),
            default=(),
            semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/ControlParameter"
            ],
            description="A set of column indices to filter on",
        ),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set with sets to specify rows to apply filters on.",
    )
    mins = hyperparams.Set(
        elements=hyperparams.Set(
            elements=hyperparams.Hyperparameter[float](-1),
            default=(),
            semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/ControlParameter"
            ],
            description="A set of column indices to filter on",
        ),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to filter on",
    )
    maxs = hyperparams.Set(
        elements=hyperparams.Set(
            elements=hyperparams.Hyperparameter[float](-1),
            default=(),
            semantic_types=[
                "https://metadata.datadrivendiscovery.org/types/ControlParameter"
            ],
            description="A set of column indices to filter on",
        ),
        default=(),
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
        description="True when values outside the range are removed, False when values within the range are removed.",
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
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "c2fa34c0-2d1b-42af-91d2-515da4a27752",
            "version": version.__version__,
            "name": "Vector bound filter",
            "python_path": "d3m.primitives.data_preprocessing.vector_bounds_filter.DistilVectorBoundsFilter",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/distil/primitives/vector_filter.py",
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
            "primitive_family": metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        }
    )

    _floatvector_semantic = (
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    )

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
        indices = self.hyperparams["row_indices_list"]

        # if len(maxs) != len(indices) or len(mins) != len(indices):
        # if min(len(maxs), len(mins), len(indices))
        total_filters_to_apply = 0

        if len(indices) == 0:
            total_filters_to_apply = 1
            if len(maxs) == 0 and len(mins) == 0:
                return base.CallResult(inputs)
            indices = [inputs.index.tolist()]
        elif len(maxs) < len(mins):
            logger.warning("excess min filters present and will be skipped")
            if len(indices) > len(maxs):
                logger.warning("excess indices to filter on will be skipped")
                total_filters_to_apply = len(maxs)
            else:
                total_filters_to_apply = len(indices)
        elif len(mins) < len(maxs):
            logger.warning("excess max filters will be skipped")
            if len(indices) > len(maxs):
                logger.warning("excess indices to filter on will be skipped")
                total_filters_to_apply = len(mins)
            else:
                total_filters_to_apply = len(indices)
        else:
            total_filters_to_apply = len(indices)

        mins = [
            [
                float("-inf") if min_filter[i] == None else min_filter[i]
                for i in range(len(min_filter))
            ]
            for min_filter in mins
        ]
        maxs = [
            [
                float("inf") if max_filter[i] == None else max_filter[i]
                for i in range(len(max_filter))
            ]
            for max_filter in maxs
        ]

        indices_to_keep = np.empty((inputs.shape[0],))
        final_index = 0
        for i in range(total_filters_to_apply):
            rows = np.stack(inputs.iloc[indices[i], vector_column], axis=0)
            if self.hyperparams["inclusive"]:
                if self.hyperparams["strict"]:
                    rows = np.logical_and(
                        rows > np.array(mins[i]), rows < np.array(maxs[i])
                    )
                else:
                    rows = np.logical_and(
                        rows >= np.array(mins[i]), rows <= np.array(maxs[i])
                    )
            else:
                if self.hyperparams["strict"]:
                    rows = np.logical_or(
                        rows < np.array(mins[i]), rows > np.array(maxs[i])
                    )
                else:
                    rows = np.logical_or(
                        rows <= np.array(mins[i]), rows >= np.array(maxs[i])
                    )
            rows_to_keep = rows.sum(axis=1) == rows.shape[1]
            amount_of_kept_rows = rows_to_keep.sum()
            indices_to_keep[final_index : final_index + amount_of_kept_rows] = [
                indices[i][j] for j in range(len(indices[i])) if rows_to_keep[j]
            ]
            final_index += amount_of_kept_rows

        outputs = dataframe_utils.select_rows(inputs, indices_to_keep[0:final_index])

        return base.CallResult(outputs)

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
