import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from distil.primitives.isolation_forest import IsolationForestPrimitive
from d3m.metadata import base as metadata_base
import utils as test_utils

from sklearn.preprocessing import LabelEncoder


class IsolationForestPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "tabular_dataset_2"))

    def test_basic(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe.drop(columns=["delta", "echo"], inplace=True)

        hyperparams_class = IsolationForestPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace({"n_jobs": -1})

        isp = IsolationForestPrimitive(hyperparams=hyperparams)
        isp.set_training_data(
            inputs=dataframe[["alpha", "bravo"]],
        )
        isp.fit()
        results = isp.produce(inputs=dataframe[["alpha", "bravo"]]).value

        self.assertListEqual(
            list(results["outlier_label"]), [-1, -1, -1, -1, -1, -1, -1, -1, -1]
        )

    # def test_acled(self) -> None:
    #     dataset = test_utils.load_dataset(
    #         "/Users/vkorapaty/data/datasets/seed_datasets_current/LL0_acled_reduced_MIN_METADATA/LL0_acled_reduced_MIN_METADATA"
    #     )

    #     hyperparams_class = DatasetToDataFramePrimitive.metadata.query()[
    #         "primitive_code"
    #     ]["class_type_arguments"]["Hyperparams"]
    #     dataframe_primitive = DatasetToDataFramePrimitive(
    #         hyperparams=hyperparams_class.defaults()
    #     )
    #     dataframe = dataframe_primitive.produce(inputs=dataset).value

    #     dataframe.metadata = dataframe.metadata.update(
    #         (metadata_base.ALL_ELEMENTS, 6), {"structural_type": int}
    #     )
    #     le = LabelEncoder()
    #     dataframe["event_type"] = le.fit_transform(dataframe["event_type"])
    #     dataframe.metadata = dataframe.metadata.add_semantic_type(
    #         (metadata_base.ALL_ELEMENTS, 6),
    #         "httpse://metadata.datadrivendiscovery.org/types/Target",
    #     )
    #     dataframe.metadata = dataframe.metadata.add_semantic_type(
    #         (metadata_base.ALL_ELEMENTS, 6),
    #         "https://metadata.datadrivendiscovery.org/types/CategoricalData",
    #     )
    #     dataframe.metadata = dataframe.metadata.remove_semantic_type(
    #         (metadata_base.ALL_ELEMENTS, 6),
    #         "https://metadata.datadrivendiscovery.org/types/Attribute",
    #     )
    #     hyperparams_class = ColumnParserPrimitive.metadata.query()["primitive_code"][
    #         "class_type_arguments"
    #     ]["Hyperparams"]
    #     column_parser = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
    #     dataframe = column_parser.produce(inputs=dataframe).value

    #     hyperparams_class = IsolationForestPrimitive.metadata.query()["primitive_code"][
    #         "class_type_arguments"
    #     ]["Hyperparams"]
    #     hyperparams = hyperparams_class.defaults().replace({"n_jobs": -1})

    #     isp = IsolationForestPrimitive(hyperparams=hyperparams)
    #     isp.set_training_data(
    #         inputs=dataframe[
    #             [
    #                 "iso",
    #                 "year",
    #                 "inter1",
    #                 "inter2",
    #                 "interaction",
    #                 "latitude",
    #                 "longitude",
    #                 "geo_precision",
    #                 "fatalities",
    #             ]
    #         ],
    #     )
    #     isp.fit()
    #     results = isp.produce(
    #         inputs=dataframe[
    #             [
    #                 "iso",
    #                 "year",
    #                 "inter1",
    #                 "inter2",
    #                 "interaction",
    #                 "latitude",
    #                 "longitude",
    #                 "geo_precision",
    #                 "fatalities",
    #             ]
    #         ]
    #     ).value

    #     self.assertListEqual(
    #         list(results["label"]), [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    #     )


if __name__ == "__main__":
    unittest.main()
