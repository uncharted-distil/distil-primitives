
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
        dataset = test_utils.load_dataset(_dataset_path)
        dataframe = test_utils.get_dataframe(dataset, 'learningData')
