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


if __name__ == "__main__":
    unittest.main()
