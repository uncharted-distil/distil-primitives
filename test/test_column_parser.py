import unittest
from os import path
import csv
import sys
import math
import pandas as pd
import numpy as np

# from common_primitives.column_parser import ColumnParserPrimitive
from d3m import container, exceptions
from d3m.metadata import base as metadata_base

from common_primitives import dataset_to_dataframe, dataframe_image_reader

from distil.primitives.column_parser import ColumnParserPrimitive
from distil.primitives import utils
import utils as test_utils

class ColumnParserPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'tabular_dataset_2'))
    _image_dataset_path = path.abspath(path.join(path.dirname(__file__), 'satellite_image_dataset'))

    def test_basic(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        df = test_utils.get_dataframe(dataset, 'learningData')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'http://schema.org/Integer')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df['d3mIndex'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['alpha'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['bravo'].dtype, np.dtype('float64'))
        self.assertEqual(result_df['charlie'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['delta'].dtype, np.dtype('object'))
        self.assertEqual(result_df['echo'].dtype, np.dtype('float64'))

    def test_hyperparams(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        df = test_utils.get_dataframe(dataset, 'learningData')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'http://schema.org/Integer')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace({'use_columns': [1, 2],
                                                                                    'exclude_columns': [0, 3]}))
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df['d3mIndex'].dtype, np.dtype('object'))
        self.assertEqual(result_df['alpha'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['bravo'].dtype, np.dtype('float64'))
        self.assertEqual(result_df['charlie'].dtype, np.dtype('object'))
        self.assertEqual(result_df['delta'].dtype, np.dtype('object'))
        self.assertEqual(result_df['echo'].dtype, np.dtype('object'))


    def test_hyperparams_exclude(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        df = test_utils.get_dataframe(dataset, 'learningData')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'http://schema.org/Integer')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults().replace({'exclude_columns': [0, 3]}))
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df['d3mIndex'].dtype, np.dtype('object'))
        self.assertEqual(result_df['alpha'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['bravo'].dtype, np.dtype('float64'))
        self.assertEqual(result_df['charlie'].dtype, np.dtype('object'))
        self.assertEqual(result_df['delta'].dtype, np.dtype('object'))
        self.assertEqual(result_df['echo'].dtype, np.dtype('float64'))


    def test_hyperparams_structural_type(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        df = test_utils.get_dataframe(dataset, 'learningData')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'http://schema.org/Integer')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 3), 'http://schema.org/Integer')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 4), 'http://schema.org/Boolean')
        df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 5), 'https://metadata.datadrivendiscovery.org/types/FloatVector')
        dataset = test_utils.load_dataset(self._image_dataset_path)
        # dataframe_hyperparams_class = dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        # dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=dataframe_hyperparams_class.defaults().replace({'dataframe_resource': '0'}))
        # dataframe = dataframe_primitive.produce(inputs=dataset).value
        # image_hyperparams_class = dataframe_image_reader.DataFrameImageReaderPrimitive.metadata.get_hyperparams()
        # image_primitive = dataframe_image_reader.DataFrameImageReaderPrimitive(hyperparams=image_hyperparams_class.defaults().replace({'return_result': 'replace'}))
        # images = image_primitive.produce(inputs=dataframe).value
        # images.loc[5] = images.iloc[0, :]
        # images.loc[6] = images.iloc[1, :]
        # images.loc[7] = images.iloc[2, :]
        images = test_utils.get_dataframe(dataset, 'learningData')
        df['echo'] = images['coordinates'][0:9]
        # df.metadata = df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 6), 'https://metadata.datadrivendiscovery.org/types/FloatVector')

        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults() \
            .replace({'parsing_semantics': ['http://schema.org/Float', 'http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/FloatVector']}))
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df['d3mIndex'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['alpha'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['bravo'].dtype, np.dtype('float64'))
        self.assertEqual(result_df['charlie'].dtype, np.dtype('int64'))
        self.assertEqual(result_df['delta'].dtype, np.dtype('object'))
        self.assertEqual(result_df['echo'].dtype, np.dtype('object'))
        for i in range(9):
            self.assertTrue((result_df['echo'][i] == np.fromstring(images['coordinates'][i], dtype=float, sep=',')).all())
        self.assertEqual(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 1))['structural_type'], int)
        self.assertEqual(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 2))['structural_type'], float)
        self.assertEqual(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 3))['structural_type'], int)
        self.assertEqual(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 4))['structural_type'], str)
        self.assertEqual(result_df.metadata.query((metadata_base.ALL_ELEMENTS, 5))['structural_type'], np.ndarray)


if __name__ == '__main__':
    unittest.main()
