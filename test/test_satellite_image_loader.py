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


import struct
import unittest
from os import path
import numpy as np

from d3m import container
from d3m.metadata import base as metadata_base

from distil.primitives.satellite_image_loader import (
    DataFrameSatelliteImageLoaderPrimitive,
)
import utils as test_utils
import pathlib
import os
import lz4
import imageio


class DataFrameSatelliteImageLoaderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(
        path.join(path.dirname(__file__), "satellite_image_dataset")
    )
    _media_path = (
        pathlib.Path(
            path.abspath(
                path.join(
                    path.dirname(__file__), "satellite_image_dataset", "media", ""
                )
            )
        ).as_uri()
        + os.sep
    )  # pathlib strips trailing slash

    def test_band_mapping_append(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 2),
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        )
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/FileName",
        )
        dataset.metadata = dataset.metadata.update(
            ("0",), {"location_base_uris": self._media_path}
        )
        dataset.metadata = dataset.metadata.update(
            ("learningData", metadata_base.ALL_ELEMENTS, 1),
            {"location_base_uris": [self._media_path]},
        )
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        hyperparams_class = DataFrameSatelliteImageLoaderPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace({"n_jobs": -1})
        loader = DataFrameSatelliteImageLoaderPrimitive(hyperparams=hyperparams)
        result_dataframe = loader.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe.shape), [2, 8])
        self.assertListEqual(list(result_dataframe.iloc[0, 7].shape), [12, 120, 120])
        self.assertEqual(result_dataframe["d3mIndex"][0], "1")
        self.assertEqual(
            result_dataframe["group_id"][0], "S2A_MSIL2A_20170613T101031_0_49"
        )
        self.assertEqual(result_dataframe["d3mIndex"][1], "2")
        self.assertEqual(result_dataframe["group_id"][1], "2")

    def test_band_mapping_replace(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 2),
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        )
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/FileName",
        )
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 5),
            "https://metadata.datadrivendiscovery.org/types/FloatVector",
        )
        dataset.metadata = dataset.metadata.update(
            ("0",), {"location_base_uris": self._media_path}
        )
        dataset.metadata = dataset.metadata.update(
            ("learningData", metadata_base.ALL_ELEMENTS, 1),
            {"location_base_uris": [self._media_path]},
        )
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        hyperparams_class = DataFrameSatelliteImageLoaderPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        hyperparams = hyperparams_class.defaults().replace(
            {"return_result": "replace", "n_jobs": -1}
        )
        loader = DataFrameSatelliteImageLoaderPrimitive(hyperparams=hyperparams)
        result_dataframe = loader.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe.shape), [2, 7])
        self.assertListEqual(
            list(result_dataframe["image_file"][0].shape), [12, 120, 120]
        )
        self.assertEqual(result_dataframe["d3mIndex"][0], "1")
        self.assertEqual(
            result_dataframe["group_id"][0], "S2A_MSIL2A_20170613T101031_0_49"
        )
        self.assertEqual(result_dataframe["d3mIndex"][1], "2")
        self.assertEqual(result_dataframe["group_id"][1], "2")
        self.assertEqual(
            result_dataframe.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/LocationPolygon",)
            ),
            [5],
        )

    def test_band_mapping_compressed(self) -> None:
        dataset = test_utils.load_dataset(self._dataset_path)
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 2),
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        )
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/FileName",
        )
        # test band column
        dataset.metadata = dataset.metadata.add_semantic_type(
            ("learningData", metadata_base.ALL_ELEMENTS, 3),
            "https://metadata.datadrivendiscovery.org/types/Band",
        )
        dataset.metadata = dataset.metadata.update(
            ("0",), {"location_base_uris": self._media_path}
        )
        dataset.metadata = dataset.metadata.update(
            ("learningData", metadata_base.ALL_ELEMENTS, 1),
            {"location_base_uris": [self._media_path]},
        )
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        hyperparams_class = DataFrameSatelliteImageLoaderPrimitive.metadata.query()[
            "primitive_code"
        ]["class_type_arguments"]["Hyperparams"]
        # included bad column name for testing band column purposes
        hyperparams = hyperparams_class.defaults().replace(
            {"compress_data": True, "n_jobs": -1, "band_column": "bleh"}
        )
        loader = DataFrameSatelliteImageLoaderPrimitive(hyperparams=hyperparams)
        result_dataframe = loader.produce(inputs=dataframe).value

        # decompress
        compressed_bytes = result_dataframe.iloc[0, 7].tobytes()
        decompressed_bytes = lz4.frame.decompress(compressed_bytes)
        storage_type, shape_0, shape_1, shape_2 = struct.unpack(
            "cIII", decompressed_bytes[:16]
        )
        result_array = np.frombuffer(
            decompressed_bytes[16:], dtype=storage_type
        ).reshape(shape_0, shape_1, shape_2)

        # load a test image
        original_image = image_array = imageio.imread(
            "test/satellite_image_dataset/media/S2A_MSIL2A_20170613T101031_0_49_B02.tif"
        )
        loaded_image = result_array[1]
        self.assertEqual(original_image.tobytes(), loaded_image.tobytes())

        compressed_bytes = result_dataframe.iloc[1, 7].tobytes()
        decompressed_bytes = lz4.frame.decompress(compressed_bytes)
        storage_type, shape_0, shape_1, shape_2 = struct.unpack(
            "cIII", decompressed_bytes[:16]
        )
        result_array = np.frombuffer(
            decompressed_bytes[16:], dtype=storage_type
        ).reshape(shape_0, shape_1, shape_2)

        # load a test image
        # original_image = image_array = imageio.imread("test/satellite_image_dataset/media/S2A_MSIL2A_20170613T101031_0_49_B8A.tif")
        loaded_image = result_array[1]
        self.assertEqual(original_image.tobytes(), loaded_image.tobytes())


if __name__ == "__main__":
    unittest.main()
