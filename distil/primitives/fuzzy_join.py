import typing
import os
import collections
from frozendict import FrozenOrderedDict

import pandas as pd  # type: ignore
import numpy as np

from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as d3m_base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from fuzzywuzzy import process
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
    accuracy = hyperparams.Union[typing.Union[float, typing.Sequence[float]]](
        configuration=collections.OrderedDict(
            set=hyperparams.List(
                elements=hyperparams.Hyperparameter[float](-1),
                default=(),
                semantic_types=[
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ],
                description="A list of accuracies, corresponding respectively to the columns to join on.",
            ),
            float=hyperparams.Hyperparameter[float](0),
        ),
        default="float",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Required accuracy of join ranging from 0.0 to 1.0, where 1.0 is an exact match.",
    )
    join_type = hyperparams.Enumeration[str](
        default="left",
        values=("left", "right", "outer", "inner", "cross"),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The type of join between two dataframes.",
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
        iterations: int = None,
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
        if type(accuracy) == float:
            if accuracy <= 0.0 or accuracy > 1.0:
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
            type(left_col) == list
            and len(left_col) != len(right_col)
            and type(accuracy) != list
            and len(accuracy) != len(left_col)
        ):
            raise exceptions.InvalidArgumentTypeError(
                "both left_col and right_col need to have same data type and if they are lists, the same list lengths"
            )
        if type(left_col) == str:
            left_col = [left_col]
            right_col = [right_col]
            accuracy = [accuracy]

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

        right_cols_to_drop = []
        new_left_cols = []
        new_right_cols = []
        for col_index in range(len(left_col)):
            # depending on the joining type, make a new dataframe that has columns we will want to merge on
            # keep track of which columns we will want to drop later on
            if join_types[col_index] in self._STRING_JOIN_TYPES:
                new_left_df = self._create_string_merge_cols(
                    left_df,
                    left_col[col_index],
                    right_df,
                    right_col[col_index],
                    accuracy[col_index],
                    col_index,
                )
                left_df[new_left_df.columns] = new_left_df
                right_name = "righty_string" + str(col_index)
                right_df.rename(
                    columns={right_col[col_index]: right_name}, inplace=True
                )
                new_left_cols += list(new_left_df.columns)
                new_right_cols.append(right_name)
            elif join_types[col_index] in self._NUMERIC_JOIN_TYPES:
                new_left_df = self._create_numeric_merge_cols(
                    left_df,
                    left_col[col_index],
                    right_df,
                    right_col[col_index],
                    accuracy[col_index],
                    col_index,
                )
                left_df[new_left_df.columns] = new_left_df
                right_name = "righty_numeric" + str(col_index)
                right_df.rename(
                    columns={right_col[col_index]: right_name}, inplace=True
                )
                new_left_cols += list(new_left_df.columns)
                new_right_cols.append(right_name)
            elif join_types[col_index] in self._VECTOR_JOIN_TYPES:
                new_left_df, new_right_df = self._create_vector_merging_cols(
                    left_df,
                    left_col[col_index],
                    right_df,
                    right_col[col_index],
                    accuracy[col_index],
                    col_index,
                )
                left_df[new_left_df.columns] = new_left_df
                right_df[new_right_df.columns] = new_right_df
                new_left_cols += list(new_left_df.columns)
                new_right_cols += list(new_right_df.columns)
                right_cols_to_drop.append(right_col[col_index])
            elif join_types[col_index] in self._DATETIME_JOIN_TYPES:
                new_left_df, new_right_df = self._create_datetime_merge_cols(
                    left_df,
                    left_col[col_index],
                    right_df,
                    right_col[col_index],
                    accuracy[col_index],
                    col_index,
                )
                left_df[new_left_df.columns] = new_left_df
                right_df[new_right_df.columns] = new_right_df
                new_left_cols += list(new_left_df.columns)
                new_right_cols += list(new_right_df.columns)
                right_cols_to_drop.append(right_col[col_index])
            else:
                raise exceptions.InvalidArgumentValueError(
                    "join not surpported on type " + str(join_types[col_index])
                )

        if "d3mIndex" in right_df.columns:
            right_cols_to_drop.append("d3mIndex")
        right_df.drop(columns=right_cols_to_drop, inplace=True)

        joined = pd.merge(
            left_df,
            right_df,
            how=self.hyperparams["join_type"],
            left_on=new_left_cols,
            right_on=new_right_cols,
            suffixes=["_left", "_right"],
        )

        # don't want to keep columns that were created specifically for merging
        # also, inner merge keeps the right column we merge on, we want to remove it
        joined.drop(columns=np.unique(new_left_cols + new_right_cols), inplace=True)

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

        # Generate metadata for the dataset using only the first row of the resource for speed -
        # metadata generation runs over each cell in the dataframe, but we only care about column
        # level generation.  Once that's done, set the actual dataframe value.
        result_dataset = container.Dataset(
            {k: v.head(1) for k, v in resource_map.items()}, generate_metadata=True
        )
        for k, v in resource_map.items():
            result_dataset[k] = v
            result_dataset.metadata = result_dataset.metadata.update(
                (k,), {"dimension": {"length": v.shape[0]}}
            )

        for key in float_vector_columns.keys():
            df = result_dataset[left_resource_id]
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
        iterations: int = None,
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
        iterations: int = None,
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
    def _create_string_merge_cols(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
        index: int,
    ) -> pd.DataFrame:

        left_keys = left_df[left_col].unique()
        right_keys = right_df[right_col].unique()
        matches: typing.Dict[str, typing.Optional[str]] = {}
        for left_key in left_keys:
            matches[left_key] = cls._string_fuzzy_match(
                left_key, right_keys, accuracy * 100
            )
        new_left_df = container.DataFrame(
            {
                "lefty_string"
                + str(index): left_df[left_col].map(lambda key: matches[key])
            }
        )
        return new_left_df

    def _numeric_fuzzy_match(match, choices, accuracy):
        # not sure if this is faster than applying a lambda against the sequence - probably is
        inv_accuracy = 1.0 - accuracy
        min_distance = float("inf")
        min_val = float("nan")
        tolerance = float(match) * inv_accuracy
        for i, num in enumerate(choices):
            distance = abs(match - num)
            if distance <= tolerance and distance <= min_distance:
                min_diff = distance
                min_val = num
        return min_val

    @classmethod
    def _create_numeric_merge_cols(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
        index: int,
    ) -> pd.DataFrame:
        choices = right_df[right_col].unique()
        new_left_df = container.DataFrame(
            {
                "lefty_numeric"
                + str(index): pd.to_numeric(left_df[left_col]).map(
                    lambda x: cls._numeric_fuzzy_match(x, choices, accuracy)
                )
            }
        )
        return new_left_df

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
    def _create_vector_merging_cols(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
        index: int,
    ) -> pd.DataFrame:
        def fromstring(x: str) -> np.ndarray:
            return np.fromstring(x, dtype=float, sep=",")

        if type(left_df[left_col][0]) == str:
            left_vector_length = np.fromstring(
                left_df[left_col][0], dtype=float, sep=","
            ).shape[0]
            new_left_cols = [
                "lefty_vector" + str(index) + "_" + str(i)
                for i in range(left_vector_length)
            ]
            new_left_df = container.DataFrame(
                left_df[left_col]
                .apply(fromstring, convert_dtype=False)
                .values.tolist(),
                columns=new_left_cols,
            )
        else:
            left_vector_length = left_df[left_col][0].shape[0]
            new_left_cols = [
                "lefty_vector" + str(index) + "_" + str(i)
                for i in range(left_vector_length)
            ]
            new_left_df = container.DataFrame(
                left_df[left_col].values.tolist(),
                columns=new_left_cols,
            )
        if type(right_df[right_col][0]) == str:
            right_vector_length = np.fromstring(
                right_df[right_col][0], dtype=float, sep=","
            ).shape[0]
            new_right_cols = [
                "righty_vector" + str(index) + "_" + str(i)
                for i in range(right_vector_length)
            ]
            new_right_df = container.DataFrame(
                right_df[right_col]
                .apply(fromstring, convert_dtype=False)
                .values.tolist(),
                columns=new_right_cols,
            )
        else:
            right_vector_length = right_df[right_col][0].shape[0]
            new_right_cols = [
                "righty_vector" + str(index) + "_" + str(i)
                for i in range(right_vector_length)
            ]
            new_right_df = container.DataFrame(
                right_df[right_col].values.tolist(),
                columns=new_right_cols,
            )

        for i in range(len(new_left_cols)):
            new_left_df[new_left_cols[i]] = new_left_df[new_left_cols[i]].map(
                lambda x: cls._numeric_fuzzy_match(
                    x, new_right_df[new_right_cols[i]], accuracy
                )
            )
        return (new_left_df, new_right_df)

    @classmethod
    def _create_datetime_merge_cols(
        cls,
        left_df: container.DataFrame,
        left_col: str,
        right_df: container.DataFrame,
        right_col: str,
        accuracy: float,
        index: int,
    ) -> pd.DataFrame:
        # use d3mIndex from left col if present
        # compute a tolerance delta for time matching based on a percentage of the minimum left/right time
        # range
        left_name = "lefty_datetime" + str(index)
        right_name = "righty_datetime" + str(index)
        new_right_df = container.DataFrame(
            {
                right_name: np.array(
                    [np.datetime64(parser.parse(dt)) for dt in right_df[right_col]]
                )
            }
        )
        choices = np.unique(new_right_df[right_name])
        left_keys = np.array(
            [np.datetime64(parser.parse(dt)) for dt in left_df[left_col].values]
        )
        time_tolerance = (1.0 - accuracy) * cls._compute_time_range(left_keys, choices)

        new_left_df = container.DataFrame(
            {
                left_name: np.array(
                    [
                        cls._datetime_fuzzy_match(dt, choices, time_tolerance)
                        for dt in left_keys
                    ]
                )
            }
        )
        return new_left_df, new_right_df

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
