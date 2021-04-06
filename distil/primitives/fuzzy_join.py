import typing
import os
import csv
import collections
import sys

import frozendict  # type: ignore
import pandas as pd  # type: ignore
import numpy as np
import math

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as d3m_base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from fuzzywuzzy import fuzz, process
from dateutil import parser
from distil.utils import CYTHON_DEP
import version

__all__ = ("FuzzyJoinPrimitive",)

Inputs = container.Dataset
Outputs = container.Dataset


class Hyperparams(hyperparams.Hyperparams):
    left_col = hyperparams.Union[typing.Union[str, typing.Sequence[str]]](
        configuration=collections.OrderedDict(
            set=hyperparams.Set(
                elements=hyperparams.Hyperparameter[str](
                    default="",
                    semantic_types=[
                        "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                    ],
                    description="Name of the column.",
                ),
                default=(),
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
            ),
            str=hyperparams.Hyperparameter[str](
                default="",
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="Name of the column.",
            ),
        ),
        default="str",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Columns to join on from left dataframe",
    )
    right_col = hyperparams.Union[typing.Union[str, typing.Sequence[str]]](
        configuration=collections.OrderedDict(
            set=hyperparams.Set(
                elements=hyperparams.Hyperparameter[str](
                    default="",
                    semantic_types=[
                        "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                    ],
                    description="Name of the column.",
                ),
                default=(),
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
            ),
            str=hyperparams.Hyperparameter[str](
                default="",
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="Name of the column.",
            ),
        ),
        default="str",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Columns to join on from right dataframe",
    )
    accuracy = hyperparams.Hyperparameter[float](
        default=0.0,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Requierd accuracy of join ranging from 0.0 to 1.0, where 1.0 is an exact match.",
    )
    accuracy = hyperparams.Union[typing.Union[float, typing.Sequence[float]]](
        configuration=collections.OrderedDict(
            set=hyperparams.Set(
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
        description="Required accuracy of join ranging from 0.0 to 1.0, where 1.0 is an exact match.",
    )


class FuzzyJoinPrimitive(
    transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]
):
    """
    Place holder fuzzy join primitive
    """

    _STRING_JOIN_TYPES = set(
        (
            "https://metadata.datadrivendiscovery.org/types/CategoricalData",
            "http://schema.org/Text",
            "http://schema.org/Boolean",
        )
    )

    _NUMERIC_JOIN_TYPES = set(("http://schema.org/Integer", "http://schema.org/Float"))

    _VECTOR_JOIN_TYPES = set(
        ("https://metadata.datadrivendiscovery.org/types/FloatVector",)
    )

    _DATETIME_JOIN_TYPES = set(("http://schema.org/DateTime",))

    _SUPPORTED_TYPES = (
        _STRING_JOIN_TYPES.union(_NUMERIC_JOIN_TYPES)
        .union(_DATETIME_JOIN_TYPES)
        .union(_VECTOR_JOIN_TYPES)
    )

    __author__ = ("Uncharted Software",)
    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "6c3188bf-322d-4f9b-bb91-68151bf1f17f",
            "version": version.__version__,
            "name": "Fuzzy Join Placeholder",
            "python_path": "d3m.primitives.data_transformation.fuzzy_join.DistilFuzzyJoin",
            "keywords": ["join", "columns", "dataframe"],
            "source": {
                "name": "Uncharted Software",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/fuzzy_join.py",
                    "https://github.com/uncharted-distil/distil-primitives",
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
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.ARRAY_CONCATENATION,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def produce(
        self,
        *,
        left: Inputs,  # type: ignore
        right: Inputs,  # type: ignore
        timeout: float = None,
        iterations: int = None
    ) -> base.CallResult[Outputs]:

        # attempt to extract the main table
        try:
            left_resource_id, left_df = d3m_base_utils.get_tabular_resource(left, None)
        except ValueError as error:
            raise exceptions.InvalidArgumentValueError(
                "Failure to find tabular resource in left dataset"
            ) from error

        try:
            right_resource_id, right_df = d3m_base_utils.get_tabular_resource(
                right, None
            )
        except ValueError as error:
            raise exceptions.InvalidArgumentValueError(
                "Failure to find tabular resource in right dataset"
            ) from error

        accuracy = self.hyperparams["accuracy"]
        if type(accuracy) == float and (accuracy <= 0.0 or accuracy > 1.0):
            raise exceptions.InvalidArgumentValueError(
                "accuracy of " + str(accuracy) + " is out of range"
            )
        else:
            for acc in accuracy:
                if acc <= 0.0 or acc > 1.0:
                    raise exceptions.InvalidArgumentValueError(
                        "accuracy of " + str(acc) + " is out of range"
                    )

        left_col = self.hyperparams["left_col"]
        right_col = self.hyperparams["right_col"]

        if type(left_col) != type(right_col) or (
            type(left_col) == list and len(left_col) != len(right_col)
        ):
            raise exceptions.InvalidArgumentTypeError(
                "both left_col and right_col need to have same data type and if they are lists, the same list lengths"
            )
        if type(left_col) == str:
            # perform join based on semantic type
            join_type = self._get_join_semantic_type(
                left, left_resource_id, left_col, right, right_resource_id, right_col
            )
            joined: pd.Dataframe = None
            if join_type in self._STRING_JOIN_TYPES:
                joined = self._join_string_col(
                    left_df, left_col, right_df, right_col, accuracy
                )
            elif join_type in self._NUMERIC_JOIN_TYPES:
                joined = self._join_numeric_col(
                    left_df, left_col, right_df, right_col, accuracy
                )
            elif join_type in self._VECTOR_JOIN_TYPES:
                joined = self._join_vector_col(
                    left_df, left_col, right_df, right_col, accuracy
                )
            elif join_type in self._DATETIME_JOIN_TYPES:
                joined = self._join_datetime_col(
                    left_df, left_col, right_df, right_col, accuracy
                )
            else:
                raise exceptions.InvalidArgumentValueError(
                    "join not surpported on type " + str(join_type)
                )
        else:
            join_types = [
                self._get_join_semantic_type(
                    left,
                    left_resource_id,
                    left_col[i],
                    right,
                    right_resource_id,
                    right_col[i],
                )
                for i in range(len(left_col))
            ]
            # if self._VECTOR_JOIN_TYPES[0] in join_types:
            #     column = join_types.index(self._VECTOR_JOIN_TYPES[0])
            #     left_df = self._join_vector_col(
            #         left_df, left_col[column], right_df, right_col[column], accuracy
            #     )
            #     del left_col[column]
            #     del right_col[column]
            #     del join_types[column]
            joined = left_df
            column_intersection = set(left_df.columns).intersection(right_df.columns)
            for col_index in range(len(left_col)):
                excess_columns = set(joined.columns).difference(left_df.columns)
                # ensure columns with the same name from left df column doesn't get removed
                for col_inter in column_intersection:
                    if col_inter + "_left" in excess_columns:
                        excess_columns.remove(col_inter + "_left")
                # get rid of columns that will be added again from right df
                if len(excess_columns) > 0:
                    joined.drop(columns=excess_columns, inplace=True)
                if join_types[col_index] in self._STRING_JOIN_TYPES:
                    joined = self._join_string_col(
                        joined,
                        left_col[col_index],
                        right_df,
                        right_col[col_index],
                        accuracy[col_index],
                    )
                elif join_types[col_index] in self._NUMERIC_JOIN_TYPES:
                    joined = self._join_numeric_col(
                        joined,
                        left_col[col_index],
                        right_df,
                        right_col[col_index],
                        accuracy[col_index],
                    )
                elif join_types[col_index] in self._VECTOR_JOIN_TYPES:
                    joined = self._join_vector_col(
                        joined,
                        left_col[col_index],
                        right_df,
                        right_col[col_index],
                        accuracy[col_index],
                    )
                elif join_types[col_index] in self._DATETIME_JOIN_TYPES:
                    joined = self._join_datetime_col(
                        joined,
                        left_col[col_index],
                        right_df,
                        right_col[col_index],
                        accuracy[col_index],
                    )
                else:
                    raise exceptions.InvalidArgumentValueError(
                        "join not surpported on type " + str(join_type)
                    )

        # create a new dataset to hold the joined data
        resource_map = {}
        float_vector_columns = {}
        for resource_id, resource in left.items():  # type: ignore
            if resource_id == left_resource_id:
                for column in joined.columns:
                    # need to avoid bug in container.Dataset, it doesn't like vector columns
                    if type(joined[column][0]) == np.ndarray:
                        float_vector_columns[column] = joined[column]
                        joined[column] = np.NAN
                resource_map[resource_id] = joined
            else:
                resource_map[resource_id] = resource
        result_dataset = container.Dataset(resource_map, generate_metadata=True)
        for key in float_vector_columns.keys():
            df = result_dataset["0"]
            df[key] = float_vector_columns[key]
            float_vec_loc = df.columns.get_loc(key)
            float_vec_col_indices = df.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/FloatVector",)
            )
            if float_vec_loc not in float_vec_col_indices:
                df.metadata = df.metadata.add_semantic_type(
                    (metadata_base.ALL_ELEMENTS, float_vec_loc),
                    "https://metadata.datadrivendiscovery.org/types/FloatVector",
                )

        return base.CallResult(result_dataset)

    def multi_produce(
        self,
        *,
        produce_methods: typing.Sequence[str],
        left: Inputs,
        right: Inputs,  # type: ignore
        timeout: float = None,
        iterations: int = None
    ) -> base.MultiCallResult:  # type: ignore
        return self._multi_produce(
            produce_methods=produce_methods,
            timeout=timeout,
            iterations=iterations,
            left=left,
            right=right,
        )

    def fit_multi_produce(
        self,
        *,
        produce_methods: typing.Sequence[str],
        left: Inputs,
        right: Inputs,  # type: ignore
        timeout: float = None,
        iterations: int = None
    ) -> base.MultiCallResult:  # type: ignore
        return self._fit_multi_produce(
            produce_methods=produce_methods,
            timeout=timeout,
            iterations=iterations,
            left=left,
            right=right,
        )

    @classmethod
    def _get_join_semantic_type(
        cls,
        left: container.Dataset,
        left_resource_id: str,
        left_col: str,
        right: container.Dataset,
        right_resource_id: str,
        right_col: str,
    ) -> typing.Optional[str]:
        # get semantic types for left and right cols
        left_types = cls._get_column_semantic_type(left, left_resource_id, left_col)
        right_types = cls._get_column_semantic_type(right, right_resource_id, right_col)

        # extract supported types
        supported_left_types = left_types.intersection(cls._SUPPORTED_TYPES)
        supported_right_types = right_types.intersection(cls._SUPPORTED_TYPES)

        # check for exact match
        join_types = list(supported_left_types.intersection(supported_right_types))
        if len(join_types) == 0:
            if (
                len(left_types.intersection(cls._NUMERIC_JOIN_TYPES)) > 0
                and len(right_types.intersection(cls._NUMERIC_JOIN_TYPES)) > 0
            ):
                # no exact match, but FLOAT and INT are allowed to join
                join_types = ["http://schema.org/Float"]
            elif (
                len(left_types.intersection(cls._STRING_JOIN_TYPES)) > 0
                and len(right_types.intersection(cls._STRING_JOIN_TYPES)) > 0
            ):
                # no exact match, but any text-based type is allowed to join
                join_types = ["http://schema.org/Text"]

        if len(join_types) > 0:
            return join_types[0]
        return None

    @classmethod
    def _get_column_semantic_type(
        cls, dataset: container.Dataset, resource_id: str, col_name: str
    ) -> typing.Set[str]:
        for col_idx in range(
            dataset.metadata.query((resource_id, metadata_base.ALL_ELEMENTS))[
                "dimension"
            ]["length"]
        ):
            col_metadata = dataset.metadata.query(
                (resource_id, metadata_base.ALL_ELEMENTS, col_idx)
            )
            if col_metadata.get("name", "") == col_name:
                return set(col_metadata.get("semantic_types", ()))
        return set()

    @classmethod
    def _string_fuzzy_match(
        cls, match: typing.Any, choices: typing.Sequence[typing.Any], min_score: float
    ) -> typing.Optional[str]:
        choice, score = process.extractOne(match, choices)
        val = None
        if score >= min_score:
            val = choice
        return val

    @classmethod
    def _join_string_col(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
    ) -> pd.DataFrame:
        # use d3mIndex from left col if present
        right_df = right_df.drop(columns="d3mIndex")

        # pre-compute fuzzy matches
        left_keys = left_df[left_col].unique()
        right_keys = right_df[right_col].unique()
        matches: typing.Dict[str, typing.Optional[str]] = {}
        for left_key in left_keys:
            matches[left_key] = cls._string_fuzzy_match(
                left_key, right_keys, accuracy * 100
            )

        # look up pre-computed fuzzy match for each element in the left column
        left_df.index = left_df[left_col].map(lambda key: matches[key])

        # make the right col the right dataframe index
        right_df = right_df.set_index(right_col)

        # inner join on the left / right indices
        joined = container.DataFrame(
            left_df.join(right_df, lsuffix="_left", rsuffix="_right", how="inner")
        )

        # sort on the d3m index if there, otherwise use the joined column
        if "d3mIndex" in joined:
            joined = joined.sort_values(by=["d3mIndex"])
        else:
            joined = joined.sort_values(by=[left_col])
        joined = joined.reset_index(drop=True)

        return joined

    def _numeric_fuzzy_match(match, choices, accuracy):
        # not sure if this is faster than applying a lambda against the sequence - probably is
        inv_accuracy = 1.0 - accuracy
        min_distance = float("inf")
        min_val = None
        tolerance = float(match) * inv_accuracy
        for i, num in enumerate(choices):
            distance = abs(match - num)
            if distance <= tolerance and distance <= min_distance:
                min_diff = distance
                min_val = num
        return min_val

    @classmethod
    def _join_numeric_col(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
    ) -> pd.DataFrame:
        # use d3mIndex from left col if present
        right_df = right_df.drop(columns="d3mIndex")

        # fuzzy match each of the left join col against the right join col value and save the results as the left
        # dataframe index
        right_df[right_col] = pd.to_numeric(right_df[right_col])
        choices = right_df[right_col].unique()
        left_df[left_col] = pd.to_numeric(left_df[left_col])
        left_df.index = left_df[left_col].map(
            lambda x: cls._numeric_fuzzy_match(x, choices, accuracy)
        )

        # make the right col the right dataframe index
        right_df = right_df.set_index(right_col)

        # inner join on the left / right indices
        joined = container.DataFrame(
            left_df.join(right_df, lsuffix="_left", rsuffix="_right", how="inner")
        )

        # sort on the d3m index if there, otherwise use the joined column
        if "d3mIndex" in joined:
            joined = joined.sort_values(by=["d3mIndex"])
        else:
            joined = joined.sort_values(by=[left_col])
        joined = joined.reset_index(drop=True)

        return joined

    @classmethod
    def _vector_fuzzy_match(cls, match, choices, accuracy):
        tolerance = match * (1 - accuracy)
        min_distance = float("inf")
        min_val = None
        for i in range(choices.shape[0]):
            num = choices[i, :]
            distance = abs(match - num)
            if np.all(distance <= tolerance) and np.all(distance <= min_distance):
                min_diff = distance
                min_val = num
        return min_val

    @classmethod
    def _join_vector_col(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
    ) -> pd.DataFrame:
        new_left_cols = ["lefty" + str(i) for i in range(left_df[left_col][0].shape[0])]
        new_right_cols = [
            "righty" + str(i) for i in range(right_df[right_col][0].shape[0])
        ]
        left_df[new_left_cols] = container.DataFrame(
            left_df[left_col].values.tolist(), columns=new_left_cols
        )
        # new_left_df["d3mIndex"] = left_df["d3mIndex"]
        right_df[new_right_cols] = container.DataFrame(
            right_df[right_col].values.tolist(), columns=new_right_cols
        )
        right_df.drop(columns=right_col, inplace=True)
        # new_right_df["d3mIndex"] = right_df["d3mIndex"]
        joined = left_df
        column_intersection = set(left_df.columns).intersection(right_df.columns)
        for i in range(len(new_left_cols)):
            excess_columns = set(joined.columns).difference(left_df.columns)
            # ensure columns with the same name from left df column doesn't get removed
            for col_inter in column_intersection:
                if col_inter + "_left" in excess_columns:
                    excess_columns.remove(col_inter + "_left")
            # get rid of columns that will be added again from right df
            if len(excess_columns) > 0:
                joined.drop(columns=excess_columns, inplace=True)
            joined = cls._join_numeric_col(
                left_df,
                new_left_cols[i],
                right_df,
                new_right_cols[i],
                accuracy,
            )
            # these columns have no use anymore
            joined.drop(columns=new_left_cols[i], inplace=True)
            left_df.drop(columns=new_left_cols[i], inplace=True)
            right_df.drop(columns=new_right_cols[i], inplace=True)

        return joined

    @classmethod
    def _join_datetime_col(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
    ) -> pd.DataFrame:
        # use d3mIndex from left col if present
        right_df = right_df.drop(columns="d3mIndex")

        # compute a tolerance delta for time matching based on a percentage of the minimum left/right time
        # range
        choices = np.array(
            [np.datetime64(parser.parse(dt)) for dt in right_df[right_col].unique()]
        )
        left_keys = np.array(
            [np.datetime64(parser.parse(dt)) for dt in left_df[left_col].values]
        )
        time_tolerance = (1.0 - accuracy) * cls._compute_time_range(left_keys, choices)

        left_df.index = np.array(
            [cls._datetime_fuzzy_match(dt, choices, time_tolerance) for dt in left_keys]
        )

        # make the right col the right dataframe index
        right_df = right_df.set_index(right_col)

        # inner join on the left / right indices
        joined = container.DataFrame(
            left_df.join(right_df, lsuffix="_left", rsuffix="_right", how="inner")
        )

        # sort on the d3m index if there, otherwise use the joined column
        if "d3mIndex" in joined:
            joined = joined.sort_values(by=["d3mIndex"])
        else:
            joined = joined.sort_values(by=[left_col])
        joined = joined.reset_index(drop=True)

        return joined

    @classmethod
    def _datetime_fuzzy_match(
        cls,
        match: np.datetime64,
        choices: typing.Sequence[np.datetime64],
        tolerance: np.timedelta64,
    ) -> typing.Optional[np.datetime64]:
        min_distance = None
        min_date = None
        for i, date in enumerate(choices):
            distance = abs(match - date)
            if distance <= tolerance and (
                min_distance is None or distance < min_distance
            ):
                min_distance = distance
                min_date = date
        return min_date

    @classmethod
    def _compute_time_range(
        cls, left: typing.Sequence[np.datetime64], right: typing.Sequence[np.datetime64]
    ) -> float:
        left_min = np.amin(left)
        left_max = np.amax(left)
        left_delta = left_max - left_min

        right_min = np.amin(right)
        right_max = np.amax(right)
        right_delta = right_max - right_min

        return min(left_delta, right_delta)
