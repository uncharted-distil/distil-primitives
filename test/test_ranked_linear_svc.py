
import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from common_primitives.dataset_to_dataframe import DatasetToDataFramePrimitive
from common_primitives.column_parser import ColumnParserPrimitive
from distil.primitives.ranked_linear_svc import RankedLinearSVCPrimitive
from d3m.metadata import base as metadata_base
import utils as test_utils


class RankedLinearSVCPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'tabular_dataset_2'))

    def test_basic(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, 'learningData')
        dataframe.drop(columns=['delta', 'echo'], inplace=True)

        hyperparams_class = \
            RankedLinearSVCPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults()

        ranked_lsvc = RankedLinearSVCPrimitive(hyperparams=hyperparams)
        ranked_lsvc.set_training_data(inputs=dataframe[['alpha', 'bravo']], outputs=pd.DataFrame({'charlie': dataframe['charlie'].astype(int)}))
        ranked_lsvc.fit()
        results = ranked_lsvc.produce(inputs=dataframe[['alpha', 'bravo']]).value
        expected_labels = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        expected_confidence_rank = [8, 8, 8, 5, 5, 5, 2, 2, 2]
        self.assertListEqual(list(results['charlie']), expected_labels)
        self.assertListEqual(list(results['confidence']), expected_confidence_rank)

    def test_produce_no_fit(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, 'learningData')
        dataframe.drop(columns=['delta', 'echo'], inplace=True)

        hyperparams_class = \
            RankedLinearSVCPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults()

        ranked_lsvc = RankedLinearSVCPrimitive(hyperparams=hyperparams)
        ranked_lsvc.set_training_data(inputs=dataframe[['alpha', 'bravo']], outputs=pd.DataFrame({'charlie': dataframe['charlie'].astype(int)}))
        results = ranked_lsvc.produce(inputs=dataframe[['alpha', 'bravo']]).value
        expected_labels = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        expected_confidence_rank = [8, 8, 8, 5, 5, 5, 2, 2, 2]
        self.assertListEqual(list(results['charlie']), expected_labels)
        self.assertListEqual(list(results['confidence']), expected_confidence_rank)

        self.assertListEqual(results.metadata.list_columns_with_semantic_types(("https://metadata.datadrivendiscovery.org/types/PredictedTarget",)), [0, 1])
        self.assertListEqual(results.metadata.list_columns_with_semantic_types(("http://schema.org/Integer",)), [0])

if __name__ == '__main__':
    unittest.main()
