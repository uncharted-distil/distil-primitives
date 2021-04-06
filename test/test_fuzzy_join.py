import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from distil.primitives.column_parser import ColumnParserPrimitive
from distil.primitives.fuzzy_join import FuzzyJoinPrimitive as FuzzyJoin
from d3m.metadata import base as metadata_base


class FuzzyJoinPrimitiveTestCase(unittest.TestCase):

    _dataset_path_1 = path.abspath(path.join(path.dirname(__file__), "dataset_1"))
    _dataset_path_2 = path.abspath(path.join(path.dirname(__file__), "dataset_2"))

    def test_string_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = FuzzyJoin.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams_class.defaults().replace(
            {
                "left_col": "alpha",
                "right_col": "alpha",
                "accuracy": 0.9,
            }
        )
        fuzzy_join = FuzzyJoin(hyperparams=hyperparams)
        result_dataset = fuzzy_join.produce(left=dataframe_1, right=dataframe_2).value
        result_dataframe = result_dataset["0"]

        # verify the output
        self.assertListEqual(
            list(result_dataframe),
            [
                "d3mIndex",
                "alpha",
                "bravo",
                "whiskey",
                "sierra",
                "gamma_left",
                "charlie",
                "xray",
                "tango",
                "gamma_right",
            ],
        )
        self.assertListEqual(list(result_dataframe["d3mIndex"]), [1, 2, 3, 4, 5, 7, 8])
        self.assertListEqual(
            list(result_dataframe["alpha"]),
            ["yankee", "yankeee", "yank", "Hotel", "hotel", "foxtrot aa", "foxtrot"],
        )
        self.assertListEqual(
            list(result_dataframe["bravo"]), [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]
        )
        self.assertListEqual(
            list(result_dataframe["charlie"]),
            [100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 300.0],
        )

    def test_numeric_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = FuzzyJoin.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams_class.defaults().replace(
            {
                "left_col": "whiskey",
                "right_col": "xray",
                "accuracy": 0.9,
            }
        )
        fuzzy_join = FuzzyJoin(hyperparams=hyperparams)
        result_dataset = fuzzy_join.produce(left=dataframe_1, right=dataframe_2).value
        result_dataframe = result_dataset["0"]

        # verify the output
        self.assertListEqual(
            list(result_dataframe),
            [
                "d3mIndex",
                "alpha_left",
                "bravo",
                "whiskey",
                "sierra",
                "gamma_left",
                "alpha_right",
                "charlie",
                "tango",
                "gamma_right",
            ],
        )
        self.assertListEqual(list(result_dataframe["d3mIndex"]), [1, 2, 3, 4])
        self.assertListEqual(
            list(result_dataframe["alpha_left"]), ["yankee", "yankeee", "yank", "Hotel"]
        )
        self.assertListEqual(
            list(result_dataframe["alpha_right"]), ["hotel", "hotel", "hotel", "hotel"]
        )
        self.assertListEqual(
            list(result_dataframe["whiskey"]), [10.0, 10.0, 10.0, 10.0]
        )
        self.assertListEqual(
            list(result_dataframe["charlie"]), [200.0, 200.0, 200.0, 200.0]
        )

    def test_vector_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = FuzzyJoin.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace(
            {
                "left_col": "gamma",
                "right_col": "gamma",
                "accuracy": 0.95,
            }
        )
        fuzzy_join = FuzzyJoin(hyperparams=hyperparams)
        result_dataset = fuzzy_join.produce(left=dataframe_1, right=dataframe_2).value
        result_dataframe = result_dataset["0"]
        self.assertListEqual(
            list(result_dataframe.columns),
            [
                "d3mIndex",
                "alpha_left",
                "bravo",
                "whiskey",
                "sierra",
                "gamma",
                "alpha_right",
                "charlie",
                "xray",
                "tango",
            ],
        )
        self.assertListEqual(
            list(result_dataframe["d3mIndex"]),
            [
                1,
                5,
                6,
                7,
                8,
            ],
        )
        self.assertListEqual(
            list(result_dataframe["alpha_left"]),
            [
                "yankee",
                "hotel",
                "otel",
                "foxtrot aa",
                "foxtrot",
            ],
        )
        self.assertListEqual(
            list(result_dataframe["bravo"]),
            [
                1.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
        )
        self.assertListEqual(
            [row.tolist() for row in result_dataframe["gamma"]],
            [
                [10.0, 20.0],
                [10.0, 20.0],
                [13.0, 13.0],
                [13.0, 13.0],
                [3.0, 5.0],
            ],
        )

    def test_date_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = FuzzyJoin.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams_class.defaults().replace(
            {
                "left_col": "sierra",
                "right_col": "tango",
                "accuracy": 0.8,
            }
        )
        fuzzy_join = FuzzyJoin(hyperparams=hyperparams)
        result_dataset = fuzzy_join.produce(left=dataframe_1, right=dataframe_2).value
        result_dataframe = result_dataset["0"]

        # verify the output
        self.assertListEqual(
            list(result_dataframe),
            [
                "d3mIndex",
                "alpha_left",
                "bravo",
                "whiskey",
                "sierra",
                "gamma_left",
                "alpha_right",
                "charlie",
                "xray",
                "gamma_right",
            ],
        )
        self.assertListEqual(list(result_dataframe["d3mIndex"]), [1, 2, 3, 4, 5, 6])
        self.assertListEqual(
            list(result_dataframe["alpha_left"]),
            ["yankee", "yankeee", "yank", "Hotel", "hotel", "otel"],
        )
        self.assertListEqual(
            list(result_dataframe["alpha_right"]),
            ["yankee", "yankee", "yankee", "yankee", "foxtrot", "foxtrot"],
        )
        self.assertListEqual(
            list(result_dataframe["whiskey"]), [10.0, 10.0, 10.0, 10.0, 20.0, 20.0]
        )
        self.assertListEqual(
            list(result_dataframe["charlie"]),
            [100.0, 100.0, 100.0, 100.0, 300.0, 300.0],
        )

    def test_date_string_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = FuzzyJoin.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams_class.defaults().replace(
            {
                "left_col": ["alpha", "sierra"],
                "right_col": ["alpha", "tango"],
                "accuracy": [0.9, 0.8],
            }
        )
        fuzzy_join = FuzzyJoin(hyperparams=hyperparams)
        result_dataset = fuzzy_join.produce(left=dataframe_1, right=dataframe_2).value
        result_dataframe = result_dataset["0"]

        # verify the output
        self.assertListEqual(
            list(result_dataframe),
            [
                "d3mIndex",
                "alpha",
                "bravo",
                "whiskey",
                "sierra",
                "gamma_left",
                "charlie",
                "xray",
                "gamma_right",
            ],
        )
        self.assertListEqual(list(result_dataframe["d3mIndex"]), [1, 2, 3, 4, 5])
        self.assertListEqual(
            list(result_dataframe["alpha"]),
            ["yankee", "yankeee", "yank", "Hotel", "hotel"],
        )
        self.assertListEqual(
            list(result_dataframe["whiskey"]), [10.0, 10.0, 10.0, 10.0, 20.0]
        )
        self.assertListEqual(
            list(result_dataframe["charlie"]),
            [100.0, 100.0, 100.0, 100.0, 300.0],
        )

    def test_date_vector_join(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = FuzzyJoin.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults()
        hyperparams = hyperparams_class.defaults().replace(
            {
                "left_col": ["sierra", "gamma"],
                "right_col": ["tango", "gamma"],
                "accuracy": [0.8, 0.95],
            }
        )
        fuzzy_join = FuzzyJoin(hyperparams=hyperparams)
        result_dataset = fuzzy_join.produce(left=dataframe_1, right=dataframe_2).value
        result_dataframe = result_dataset["0"]

        # verify the output
        self.assertListEqual(
            list(result_dataframe),
            [
                "d3mIndex",
                "alpha_left",
                "bravo",
                "whiskey",
                "sierra",
                "gamma",
                "alpha_right",
                "charlie",
                "xray",
            ],
        )
        self.assertListEqual(list(result_dataframe["d3mIndex"]), [1, 5, 6])
        self.assertListEqual(
            list(result_dataframe["alpha_left"]),
            ["yankee", "hotel", "otel"],
        )
        self.assertListEqual(
            list(result_dataframe["alpha_right"]),
            ["yankee", "yankee", "foxtrot"],
        )
        self.assertListEqual(list(result_dataframe["whiskey"]), [10.0, 20.0, 20.0])
        self.assertListEqual(
            list(result_dataframe["charlie"]),
            [100.0, 100.0, 300.0],
        )
        self.assertListEqual(
            [row.tolist() for row in result_dataframe["gamma"]],
            [
                [10.0, 20.0],
                [10.0, 20.0],
                [13.0, 13.0],
            ],
        )

    def _load_data(cls, dataset_path: str) -> container.DataFrame:
        dataset_doc_path = path.join(dataset_path, "datasetDoc.json")

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load(
            "file://{dataset_doc_path}".format(dataset_doc_path=dataset_doc_path)
        )
        dataframe = dataset["0"]
        dataframe.metadata = dataframe.metadata.generate(dataframe)

        # set the struct type
        dataframe.metadata = dataframe.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), {"structural_type": int}
        )
        dataframe.metadata = dataframe.metadata.update(
            (metadata_base.ALL_ELEMENTS, 1), {"structural_type": str}
        )
        dataframe.metadata = dataframe.metadata.update(
            (metadata_base.ALL_ELEMENTS, 2), {"structural_type": float}
        )
        dataframe.metadata = dataframe.metadata.update(
            (metadata_base.ALL_ELEMENTS, 3), {"structural_type": float}
        )
        dataframe.metadata = dataframe.metadata.update(
            (metadata_base.ALL_ELEMENTS, 4), {"structural_type": str}
        )
        dataframe.metadata = dataframe.metadata.update(
            (metadata_base.ALL_ELEMENTS, 4), {"structural_type": float}
        )

        # set the semantic type
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/CategoricalData",
        )
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2), "http://schema.org/Float"
        )
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 3), "http://schema.org/Float"
        )
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 4), "http://schema.org/DateTime"
        )
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 5),
            "https://metadata.datadrivendiscovery.org/types/FloatVector",
        )

        # set the roles
        for i in range(1, 2):
            dataframe.metadata = dataframe.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, i),
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )

        # cast the dataframe to raw python types
        dataframe["d3mIndex"] = dataframe["d3mIndex"].astype(int)
        dataframe["alpha"] = dataframe["alpha"].astype(str)

        if "bravo" in dataframe:
            dataframe["bravo"] = dataframe["bravo"].astype(float)
            dataframe["whiskey"] = dataframe["whiskey"].astype(float)
            dataframe["sierra"] = dataframe["sierra"].astype(str)

        if "charlie" in dataframe:
            dataframe["charlie"] = dataframe["charlie"].astype(float)
            dataframe["xray"] = dataframe["xray"].astype(float)
            dataframe["tango"] = dataframe["tango"].astype(str)

        hyperparam_class = ColumnParserPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparam_class.defaults().replace(
                {
                    "parsing_semantics": (
                        "https://metadata.datadrivendiscovery.org/types/FloatVector",
                    )
                }
            )
        )
        dataset["0"] = cpp.produce(inputs=dataframe).value

        return dataset


if __name__ == "__main__":
    unittest.main()
