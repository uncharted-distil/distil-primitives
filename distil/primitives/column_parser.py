import os
import typing
import time
import logging

import numpy as np
import pandas as pd

from d3m import container, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP
from common_primitives import utils

import version

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    parsing_semantics = hyperparams.Set(
        elements=hyperparams.Enumeration(
            values=[
                "http://schema.org/Boolean",
                "http://schema.org/Integer",
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/FloatVector",
                "http://schema.org/DateTime",
            ],
            default="http://schema.org/Float",
        ),
        default=(
            "http://schema.org/Boolean",
            "http://schema.org/Integer",
            "http://schema.org/Float",
        ),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of semantic types to parse. One can provide a subset of supported semantic types to limit what the primitive parses.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description='A set of column indices to not operate on. Applicable only if "use_columns" is not provided.',
    )
    error_handling = hyperparams.Enumeration[str](
        default="coerce",
        values=("ignore", "raise", "coerce"),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Setting to deal with error when converting a column to numeric value.",
    )
    fuzzy_time_parsing = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Use fuzzy time parsing.",
    )


class ColumnParserPrimitive(
    transformer.TransformerPrimitiveBase[
        container.DataFrame, container.DataFrame, Hyperparams
    ]
):
    """
    A primitive which parses columns and sets the appropriate dtypes according to it's respective metadata.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "e8e78214-9770-4c26-9eae-a45bd0ede91a",
            "version": version.__version__,
            "name": "Column Parser",
            "python_path": "d3m.primitives.data_transformation.column_parser.DistilColumnParser",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/column_parser.py",
                    "https://gitlab.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                CYTHON_DEP,
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=d3m_utils.current_git_commit(
                            os.path.dirname(__file__)
                        ),
                    ),
                },
            ],
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[container.DataFrame]:

        start = time.time()
        logger.debug(f"Producing {__name__}")

        cols = self._get_columns(inputs.metadata)
        # outputs = container.DataFrame(generate_metadata=False)
        outputs = [None] * inputs.shape[1]

        parsing_semantics = self.hyperparams["parsing_semantics"]

        def fromstring(x: str) -> np.ndarray:
            # if column isn't a string, we'll just pass it through assuming it doesn't need to be parsed
            if type(x) is not str:
                return x

            return np.fromstring(x, dtype=float, sep=",")

        for col_index in range(len(inputs.columns)):
            if col_index in cols:
                column_metadata = inputs.metadata.query(
                    (metadata_base.ALL_ELEMENTS, col_index)
                )
                semantic_types = column_metadata.get("semantic_types", [])
                desired_semantics = set(semantic_types).intersection(parsing_semantics)
                if desired_semantics:
                    if (
                        "https://metadata.datadrivendiscovery.org/types/FloatVector"
                        in desired_semantics
                    ):
                        outputs[col_index] = inputs[inputs.columns[col_index]].apply(
                            fromstring, convert_dtype=False
                        )
                        if outputs[col_index].shape[0] > 0:
                            inputs.metadata = inputs.metadata.update_column(
                                col_index,
                                {"structural_type": type(outputs[col_index][0])},
                            )
                    elif "http://schema.org/DateTime" in desired_semantics:
                        is_fuzzy = self.hyperparams["fuzzy_time_parsing"]
                        outputs[col_index] = inputs[inputs.columns[col_index]].apply(
                            utils.parse_datetime_to_float, fuzzy=is_fuzzy
                        )
                        if outputs[col_index].shape[0] > 0:
                            inputs.metadata = inputs.metadata.update_column(
                                col_index, {"structural_type": float}
                            )
                    else:
                        outputs[col_index] = pd.to_numeric(
                            inputs.iloc[:, col_index],
                            errors=self.hyperparams["error_handling"],
                        )
                        # Update structural type to reflect the results of the to_numeric call.  We can't rely on the semantic type because
                        # error coersion may result in a type becoming a float due to the presence of NaN.
                        if outputs[col_index].shape[0] > 0:
                            updated_type = type(outputs[col_index][0].item())
                            inputs.metadata = inputs.metadata.update_column(
                                col_index, {"structural_type": updated_type}
                            )
                else:
                    # columns without specified semantics need to be concatenated
                    outputs[col_index] = inputs.iloc[:, col_index]
            else:
                # columns not specified still need to be concatenated
                outputs[col_index] = inputs.iloc[:, col_index]

        outputs = container.DataFrame(pd.concat(outputs, axis=1))
        outputs.metadata = inputs.metadata
        end = time.time()
        logger.debug(f"Produce {__name__} completed in {end - start} ms")

        return base.CallResult(outputs)

    def _get_columns(
        self, inputs_metadata: metadata_base.DataMetadata
    ) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return True

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            inputs_metadata,
            self.hyperparams["use_columns"],
            self.hyperparams["exclude_columns"],
            can_use_column,
        )

        if self.hyperparams["use_columns"] and columns_not_to_use:
            self.logger.warning(
                "Not all specified columns can parsed. Skipping columns: %(columns)s",
                {
                    "columns": columns_not_to_use,
                },
            )

        return columns_to_use
