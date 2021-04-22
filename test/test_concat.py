import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container, exceptions
from distil.primitives.column_parser import ColumnParserPrimitive
from distil.primitives.concat import VerticalConcatenationPrimitive as VCPrimitive
from d3m.metadata import base as metadata_base


class VerticalConcatenationPrimitiveTestCase(unittest.TestCase):

    _dataset_path_1 = path.abspath(path.join(path.dirname(__file__), "dataset_1"))
    _dataset_path_2 = path.abspath(path.join(path.dirname(__file__), "dataset_2"))

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
        return cpp.produce(inputs=dataframe).value

    def test_union(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = VCPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace(
            {"remove_duplicate_rows": False}
        )
        concat_prim = VCPrimitive(hyperparams=hyperparams)
        result_dataframe = concat_prim.produce(
            inputs=container.List([dataframe_1, dataframe_2])
        ).value

        self.assertListEqual(
            list(result_dataframe.columns),
            [
                "d3mIndex",
                "alpha",
                "bravo",
                "whiskey",
                "sierra",
                "gamma",
                "charlie",
                "xray",
                "tango",
            ],
        )
        self.assertEqual(result_dataframe["alpha"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["bravo"].isnull().sum(), 4)
        self.assertEqual(result_dataframe["whiskey"].isnull().sum(), 4)
        self.assertEqual(result_dataframe["sierra"].isnull().sum(), 4)
        self.assertEqual(result_dataframe["gamma"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["charlie"].isnull().sum(), 8)
        self.assertEqual(result_dataframe["xray"].isnull().sum(), 8)
        self.assertEqual(result_dataframe["tango"].isnull().sum(), 8)

    def test_union_delete_duplicate_rows(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = VCPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults()
        concat_prim = VCPrimitive(hyperparams=hyperparams)
        result_dataframe = concat_prim.produce(
            inputs=container.List([dataframe_1, dataframe_2])
        ).value

        self.assertListEqual(list(result_dataframe.shape), [8, 9])
        self.assertListEqual(
            list(result_dataframe.columns),
            [
                "d3mIndex",
                "alpha",
                "bravo",
                "whiskey",
                "sierra",
                "gamma",
                "charlie",
                "xray",
                "tango",
            ],
        )
        self.assertEqual(result_dataframe["alpha"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["bravo"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["whiskey"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["sierra"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["gamma"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["charlie"].isnull().sum(), 8)
        self.assertEqual(result_dataframe["xray"].isnull().sum(), 8)
        self.assertEqual(result_dataframe["tango"].isnull().sum(), 8)

    def test_intersection(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = VCPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace(
            {"remove_duplicate_rows": False, "column_overlap": "intersection"}
        )
        concat_prim = VCPrimitive(hyperparams=hyperparams)
        result_dataframe = concat_prim.produce(
            inputs=container.List([dataframe_1, dataframe_2])
        ).value

        self.assertListEqual(
            list(result_dataframe.columns),
            [
                "d3mIndex",
                "alpha",
                "gamma",
            ],
        )
        self.assertEqual(result_dataframe["alpha"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["gamma"].isnull().sum(), 0)

    def test_exact_error(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = VCPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace(
            {"remove_duplicate_rows": False, "column_overlap": "exact"}
        )
        concat_prim = VCPrimitive(hyperparams=hyperparams)
        with self.assertRaises(exceptions.InvalidArgumentValueError):
            concat_prim.produce(inputs=container.List([dataframe_1, dataframe_2]))

    def test_exact(self) -> None:
        dataframe_1 = self._load_data(self._dataset_path_1)
        dataframe_2 = self._load_data(self._dataset_path_2)

        hyperparams_class = VCPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace(
            {"remove_duplicate_rows": False, "column_overlap": "exact"}
        )
        dataframe_1.drop(columns=["bravo", "whiskey", "sierra"], inplace=True)
        dataframe_2.drop(columns=["charlie", "xray", "tango"], inplace=True)
        concat_prim = VCPrimitive(hyperparams=hyperparams)
        result_dataframe = concat_prim.produce(
            inputs=container.List([dataframe_1, dataframe_2])
        ).value

        self.assertListEqual(
            list(result_dataframe.columns),
            [
                "d3mIndex",
                "alpha",
                "gamma",
            ],
        )
        self.assertEqual(result_dataframe["alpha"].isnull().sum(), 0)
        self.assertEqual(result_dataframe["gamma"].isnull().sum(), 0)


if __name__ == "__main__":
    unittest.main()
