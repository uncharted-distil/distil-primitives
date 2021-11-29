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

import logging
import os
from typing import List

import pandas as pd
import numpy as np
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning
from distil.utils import CYTHON_DEP
import version

logger = logging.getLogger(__name__)

__all__ = ("ListEncoderPrimitive",)


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )


class Params(params.Params):
    columns: List[int]


class ListEncoderPrimitive(
    unsupervised_learning.UnsupervisedLearnerPrimitiveBase[
        container.DataFrame, container.DataFrame, Params, Hyperparams
    ]
):
    """
     List Encoder takes columns that are made up out of lists and replaces them by expanding the list
    across multiple columns.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "67f53b00-f936-4bb4-873e-4698c4aaa37f",
            "version": version.__version__,
            "name": "List encoder",
            "python_path": "d3m.primitives.data_transformation.list_to_dataframe.DistilListEncoder",
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/list_to_dataframe.py",
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
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._cols: List[int] = []

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs

    def fit(
        self, *, timeout: float = None, iterations: int = None
    ) -> base.CallResult[None]:
        logger.debug(f"Fitting {__name__}")

        # figure out columns to operate on
        cols = list(range(len(self._inputs.columns)))
        if len(self.hyperparams["use_columns"]) > 0:
            cols = list(set(cols) & set(self.hyperparams["use_columns"]))
        filtered_cols: List[int] = []
        for c in cols:
            if isinstance(self._inputs.iloc[0, c], np.ndarray):
                filtered_cols.append(c)

        logger.debug(f"Found {len(filtered_cols)} columns to encode")

        self._cols = list(filtered_cols)
        if len(self._cols) is 0:
            return base.CallResult(None)

        return base.CallResult(None)

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[container.DataFrame]:
        logger.debug(f"Producing {__name__}")

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        # encode using the previously identified categorical columns
        input_cols = inputs.iloc[:, self._cols]
        from itertools import zip_longest

        encoded_cols = container.DataFrame()
        for i in self._cols:
            col_name = inputs.columns[i]
            col = container.DataFrame.from_records(
                zip_longest(*inputs[col_name].values)
            ).T
            col.columns = [f"{col_name}_{x}" for x in range(len(col.columns))]
            encoded_cols = pd.concat([encoded_cols, col], axis=1)

        # append the encoding columns and generate metadata
        outputs = inputs.copy()
        encoded_cols.metadata = encoded_cols.metadata.generate(encoded_cols)

        for c in range(encoded_cols.shape[1]):
            encoded_cols.metadata = encoded_cols.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, c), "http://schema.org/Float"
            )

        outputs = outputs.append_columns(encoded_cols)

        # drop the source columns
        outputs = outputs.remove_columns(self._cols)

        logger.debug(f"\n{outputs}")

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params(columns=self._cols)

    def set_params(self, *, params: Params) -> None:
        self._cols = params["columns"]
        return
