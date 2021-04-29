import logging
import os
import collections
import typing
from typing import List, Optional

import numpy as np
import pandas as pd

from d3m import container, utils, exceptions
from d3m.base import utils as d3m_base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP

import version


__all__ = ("VerticalConcatenationPrimitive",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    remove_duplicate_rows = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="If there are two rows with the same d3mIndex, one is retained",
    )
    column_overlap = hyperparams.Enumeration[str](
        default="union",
        values=("union", "exact", "intersection"),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The logic to concat two dataframes.",
    )


class VerticalConcatenationPrimitive(
    transformer.TransformerPrimitiveBase[
        container.List, container.Dataset, Hyperparams
    ]
):
    """
    A primitive to encapsulate the functionality of pandas.concat.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "b93e3e85-c462-4290-8131-abc51d76a6dd",
            "version": version.__version__,
            "name": "DistilVerticalConcat",
            "python_path": "d3m.primitives.data_transformation.concat.DistilVerticalConcat",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/concat.py",
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(
        self, *, inputs: container.List, timeout: float = None, iterations: int = None
    ) -> base.CallResult[container.Dataset]:
        # build the list of dataframes from the list of inputs
        dataframes = []
        metadata = None
        for input in inputs:
            if isinstance(input, container.DataFrame):
                dataframes.append(input)
            try:
                _, main_dr = d3m_base_utils.get_tabular_resource(input, None)
                dataframes.append(main_dr)
                metadata = input.metadata
            except ValueError as error:
                raise exceptions.InvalidArgumentValueError(
                    "Failure to find tabular resource in dataset"
                ) from error

        if self.hyperparams["column_overlap"] == "exact":
            columns_to_handle = dataframes[0].columns
            if np.sum(
                np.array([np.all(df.columns == columns_to_handle) for df in dataframes])
            ) != len(dataframes):
                raise exceptions.InvalidArgumentValueError(
                    "Dataframes don't have same columns, cannot exact concat"
                )
            concated = pd.concat(dataframes, ignore_index=True)
        elif self.hyperparams["column_overlap"] == "union":
            concated = pd.concat(dataframes, ignore_index=True)
        elif self.hyperparams["column_overlap"] == "intersection":
            concated = pd.concat(dataframes, join="inner", ignore_index=True)

        if self.hyperparams["remove_duplicate_rows"]:
            concated.drop_duplicates(subset="d3mIndex", keep="first", inplace=True)

        if metadata is None:
            metadata = container.Dataset({'learningData': concated.head(1)}, generate_metadata=True).metadata
        outputs = container.Dataset({'learningData': concated}, metadata)
        outputs.metadata = outputs.metadata.update(
            (metadata_base.ALL_ELEMENTS,), {"dimension": {"length": concated.shape[0]}}
        )

        return base.CallResult(outputs)
