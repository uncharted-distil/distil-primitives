#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import unittest
from os import path
import numpy as np

# from common_primitives.column_parser import ColumnParserPrimitive
from d3m import container, exceptions
from d3m.metadata import base as metadata_base

from distil.primitives.column_parser import ColumnParserPrimitive
from distil.primitives import utils
import utils as test_utils
from common_primitives import utils as common_utils


class ColumnParserPrimitiveTestCase(unittest.TestCase):

    _tabular_dataset_path = path.abspath(
        path.join(path.dirname(__file__), "tabular_dataset_2")
    )
    _image_dataset_path = path.abspath(
        path.join(path.dirname(__file__), "satellite_image_dataset")
    )
    _dataset_path = path.abspath(path.join(path.dirname(__file__), "dataset_1"))

    def test_basic(self) -> None:
        dataset = test_utils.load_dataset(self._tabular_dataset_path)
        df = test_utils.get_dataframe(dataset, "learningData")
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1), "http://schema.org/Integer"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2), "http://schema.org/Float"
        )
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(hyperparams=hyperparams_class.defaults())
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df["d3mIndex"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["alpha"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["bravo"].dtype, np.dtype("float64"))
        self.assertEqual(result_df["charlie"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["delta"].dtype, np.dtype("object"))
        self.assertEqual(result_df["echo"].dtype, np.dtype("float64"))

    def test_hyperparams(self) -> None:
        dataset = test_utils.load_dataset(self._tabular_dataset_path)
        df = test_utils.get_dataframe(dataset, "learningData")
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1), "http://schema.org/Integer"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2), "http://schema.org/Float"
        )
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"use_columns": [1, 2], "exclude_columns": [0, 3]}
            )
        )
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df["d3mIndex"].dtype, np.dtype("object"))
        self.assertEqual(result_df["alpha"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["bravo"].dtype, np.dtype("float64"))
        self.assertEqual(result_df["charlie"].dtype, np.dtype("object"))
        self.assertEqual(result_df["delta"].dtype, np.dtype("object"))
        self.assertEqual(result_df["echo"].dtype, np.dtype("object"))

    def test_hyperparams_exclude(self) -> None:
        dataset = test_utils.load_dataset(self._tabular_dataset_path)
        df = test_utils.get_dataframe(dataset, "learningData")
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1), "http://schema.org/Integer"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2), "http://schema.org/Float"
        )
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {"exclude_columns": [0, 3]}
            )
        )
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df["d3mIndex"].dtype, np.dtype("object"))
        self.assertEqual(result_df["alpha"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["bravo"].dtype, np.dtype("float64"))
        self.assertEqual(result_df["charlie"].dtype, np.dtype("object"))
        self.assertEqual(result_df["delta"].dtype, np.dtype("object"))
        self.assertEqual(result_df["echo"].dtype, np.dtype("float64"))

    def test_hyperparams_structural_type(self) -> None:
        dataset = test_utils.load_dataset(self._tabular_dataset_path)
        df = test_utils.get_dataframe(dataset, "learningData")
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1), "http://schema.org/Integer"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 2), "http://schema.org/Float"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 3), "http://schema.org/Integer"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 4), "http://schema.org/Boolean"
        )
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 5),
            "https://metadata.datadrivendiscovery.org/types/FloatVector",
        )
        dataset = test_utils.load_dataset(self._image_dataset_path)
        images = test_utils.get_dataframe(dataset, "learningData")
        df["echo"] = images["coordinates"][0:9]

        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {
                    "parsing_semantics": [
                        "http://schema.org/Float",
                        "http://schema.org/Integer",
                        "https://metadata.datadrivendiscovery.org/types/FloatVector",
                    ]
                }
            )
        )
        result_df = cpp.produce(inputs=df).value
        self.assertEqual(result_df["d3mIndex"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["alpha"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["bravo"].dtype, np.dtype("float64"))
        self.assertEqual(result_df["charlie"].dtype, np.dtype("int64"))
        self.assertEqual(result_df["delta"].dtype, np.dtype("object"))
        self.assertEqual(result_df["echo"].dtype, np.dtype("object"))
        for i in range(9):
            self.assertTrue(
                (
                    result_df["echo"][i]
                    == np.fromstring(images["coordinates"][i], dtype=float, sep=",")
                ).all()
            )
        self.assertEqual(
            result_df.metadata.query((metadata_base.ALL_ELEMENTS, 1))[
                "structural_type"
            ],
            int,
        )
        self.assertEqual(
            result_df.metadata.query((metadata_base.ALL_ELEMENTS, 2))[
                "structural_type"
            ],
            float,
        )
        self.assertEqual(
            result_df.metadata.query((metadata_base.ALL_ELEMENTS, 3))[
                "structural_type"
            ],
            int,
        )
        self.assertEqual(
            result_df.metadata.query((metadata_base.ALL_ELEMENTS, 4))[
                "structural_type"
            ],
            str,
        )
        self.assertEqual(
            result_df.metadata.query((metadata_base.ALL_ELEMENTS, 5))[
                "structural_type"
            ],
            np.ndarray,
        )

    def test_vector_parse_twice(self) -> None:
        dataset = test_utils.load_dataset(self._image_dataset_path)
        df = test_utils.get_dataframe(dataset, "learningData")

        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {
                    "parsing_semantics": [
                        "https://metadata.datadrivendiscovery.org/types/FloatVector",
                    ]
                }
            )
        )
        target_coords = [
            20.999598,
            63.488694,
            20.999598,
            63.499462,
            21.023702,
            63.499462,
            21.023702,
            63.488694,
        ]
        result_df = cpp.produce(inputs=df).value
        result_coords = result_df["coordinates"][0]
        self.assertEquals(len(result_coords), len(target_coords))
        for a, b in zip(target_coords, result_coords):
            self.assertAlmostEqual(a, b, 5)

        result_2_df = cpp.produce(inputs=result_df).value
        result_2_coords = result_2_df["coordinates"][0]
        self.assertEquals(len(result_2_coords), len(target_coords))
        for a, b in zip(target_coords, result_2_coords):
            self.assertAlmostEqual(a, b, 5)

    def test_datetime(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        df = test_utils.get_dataframe(dataset, "0")
        df.metadata = df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 4), "http://schema.org/DateTime"
        )
        hyperparams_class = ColumnParserPrimitive.metadata.get_hyperparams()
        cpp = ColumnParserPrimitive(
            hyperparams=hyperparams_class.defaults().replace(
                {
                    "parsing_semantics": [
                        "http://schema.org/DateTime",
                    ]
                }
            )
        )
        result_df = cpp.produce(inputs=df).value
        self.assertListEqual(
            list(result_df["sierra"]),
            [
                common_utils.parse_datetime_to_float(date, fuzzy=True)
                for date in df["sierra"]
            ],
        )


if __name__ == "__main__":
    unittest.main()
