import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from distil.primitives.column_parser import ColumnParserPrimitive
from distil.primitives.vector_filter import VectorBoundsFilterPrimitive
from d3m.metadata import base as metadata_base
import utils as test_utils


class VectorBoundsFilterPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "tabular_dataset_4"))

    def _load_data(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2),
            "https://metadata.datadrivendiscovery.org/types/FloatVector",
        )

        hyperparam_class = ColumnParserPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "parsing_semantics": (
                        "http://schema.org/Boolean",
                        "http://schema.org/Integer",
                        "http://schema.org/Float",
                        "https://metadata.datadrivendiscovery.org/types/FloatVector",
                    )
                }
            )
        )
        return cpp.produce(inputs=dataframe).value

    def test_basic(self) -> None:
        dataframe = self._load_data()

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {"mins": [[10, 20]], "maxs": [[50, 60]], "column": 2}
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 3)
        self.assertListEqual(result_df["bravo"][0].tolist(), [10.0, 20.0])
        self.assertListEqual(result_df["bravo"][1].tolist(), [30.0, 40.0])
        self.assertListEqual(result_df["bravo"][2].tolist(), [50.0, 60.0])

    def test_multiple_indices(self) -> None:
        dataframe = self._load_data()

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "mins": [[10, 20], [50, 60]],
                    "maxs": [[50, 60], [69, 80]],
                    "column": 2,
                    "row_indices_list": [[0, 1, 2], [3, 4]],
                }
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 3)
        self.assertListEqual(result_df["bravo"][0].tolist(), [10.0, 20.0])
        self.assertListEqual(result_df["bravo"][1].tolist(), [30.0, 40.0])
        self.assertListEqual(result_df["bravo"][2].tolist(), [50.0, 60.0])

    def test_exlusive(self) -> None:
        dataframe = self._load_data()

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "inclusive": False,
                    "mins": [[10, 20], [50, 60]],
                    "maxs": [[50, 60], [69, 80]],
                    "column": 2,
                    "row_indices_list": [[0, 1, 2], [3, 4]],
                }
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 4)
        self.assertListEqual(result_df["bravo"][0].tolist(), [1.0, 2.0])
        self.assertListEqual(result_df["bravo"][1].tolist(), [10.0, 20.0])
        self.assertListEqual(result_df["bravo"][2].tolist(), [50.0, 60.0])
        self.assertListEqual(result_df["bravo"][3].tolist(), [70.0, 80.0])

    def test_strict(self) -> None:
        dataframe = self._load_data()

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "inclusive": False,
                    "strict": True,
                    "mins": [[10, 20], [50, 60]],
                    "maxs": [[50, 60], [69, 80]],
                    "column": 2,
                    "row_indices_list": [[0, 1, 2], [3, 4]],
                }
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 1)
        self.assertListEqual(result_df["bravo"][0].tolist(), [1.0, 2.0])

    def test_uneven_vector_to_filters_length(self) -> None:
        dataframe = self._load_data()
        dataframe["bravo"][0] = np.append(dataframe["bravo"][0], [35])
        dataframe["bravo"][2] = np.array([30])
        dataframe["bravo"][3] = np.append(dataframe["bravo"][0], [10])

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {"mins": [[10, 20]], "maxs": [[50, 60]], "column": 2}
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 2)
        self.assertListEqual(result_df["bravo"][0].tolist(), [10.0, 20.0])
        self.assertListEqual(result_df["bravo"][1].tolist(), [30.0])

    def test_scalar(self) -> None:
        dataframe = self._load_data()

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {"mins": 15.0, "maxs": 40.0, "column": 2}
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 1)
        self.assertListEqual(result_df["bravo"][0].tolist(), [30.0, 40.0])

    def test_uneven_scalar(self) -> None:
        dataframe = self._load_data()
        dataframe["bravo"][0] = np.append(dataframe["bravo"][0], [35])
        dataframe["bravo"][2] = np.array([30])
        dataframe["bravo"][3] = np.append(dataframe["bravo"][0], [10])

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {"mins": 15.0, "maxs": 40.0, "column": 2}
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 1)
        self.assertListEqual(result_df["bravo"][0].tolist(), [30.0])

    def test_one_dim_filter(self) -> None:
        dataframe = self._load_data()

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "mins": [15.0, 20.0],
                    "maxs": [40.0, 50.0],
                    "column": 2,
                    "row_indices_list": [[0, 1, 2], [3, 4]],
                }
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 1)
        self.assertListEqual(result_df["bravo"][0].tolist(), [30.0, 40.0])

    def test_uneven_one_dim_filter(self) -> None:
        dataframe = self._load_data()
        dataframe["bravo"][0] = np.append(dataframe["bravo"][0], [35])
        dataframe["bravo"][2] = np.array([30])
        dataframe["bravo"][3] = np.append(dataframe["bravo"][0], [10])

        hyperparam_class = VectorBoundsFilterPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        vbf = VectorBoundsFilterPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "mins": [15.0, 20.0],
                    "maxs": [40.0, 50.0],
                    "column": 2,
                    "row_indices_list": [[0, 1, 2], [3, 4]],
                }
            )
        )

        result_df = vbf.produce(inputs=dataframe).value
        self.assertEqual(result_df.shape[0], 1)
        self.assertListEqual(result_df["bravo"][0].tolist(), [30.0])


if __name__ == "__main__":
    unittest.main()
