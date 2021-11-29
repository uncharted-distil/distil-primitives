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
import os
from os import path
import sys

from d3m import container
from d3m.metadata import base as metadata_base
from common_primitives import dataset_to_dataframe, dataframe_image_reader
from distil.primitives.image_transfer import ImageTransferPrimitive
import utils as test_utils


class ImageTransferPrimitveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), "image_dataset_1"))

    def test_basic(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        dataframe_hyperparams_class = (
            dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        )
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataframe_hyperparams_class.defaults().replace(
                {"dataframe_resource": "0"}
            )
        )
        dataframe = dataframe_primitive.produce(inputs=dataset).value
        image_hyperparams_class = (
            dataframe_image_reader.DataFrameImageReaderPrimitive.metadata.get_hyperparams()
        )
        image_primitive = dataframe_image_reader.DataFrameImageReaderPrimitive(
            hyperparams=image_hyperparams_class.defaults().replace(
                {"return_result": "replace"}
            )
        )
        images = image_primitive.produce(inputs=dataframe).value

        image_transfer_hyperparams = ImageTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = ImageTransferPrimitive.metadata.get_volumes()
        volumes = {
            primitive_volumes[0]["key"]: os.getenv("D3MSTATICDIR")
            + "/"
            + primitive_volumes[0]["file_digest"]
        }
        image_transfer_primitive = ImageTransferPrimitive(
            hyperparams=image_transfer_hyperparams.defaults().replace(
                {"filename_col": 0}
            ),
            volumes=volumes,
        )
        result = image_transfer_primitive.produce(inputs=images).value
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 512)

    def test_no_hyperparam(self):
        dataset = test_utils.load_dataset(self._dataset_path)

        dataframe_hyperparams_class = (
            dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.get_hyperparams()
        )
        dataframe_primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(
            hyperparams=dataframe_hyperparams_class.defaults().replace(
                {"dataframe_resource": "0"}
            )
        )
        dataframe = dataframe_primitive.produce(inputs=dataset).value
        image_hyperparams_class = (
            dataframe_image_reader.DataFrameImageReaderPrimitive.metadata.get_hyperparams()
        )
        image_primitive = dataframe_image_reader.DataFrameImageReaderPrimitive(
            hyperparams=image_hyperparams_class.defaults().replace(
                {"return_result": "replace"}
            )
        )
        images = image_primitive.produce(inputs=dataframe).value
        images.metadata = images.metadata.add_semantic_type(
            (
                metadata_base.ALL_ELEMENTS,
                images.metadata.get_column_index_from_column_name("filename"),
            ),
            "http://schema.org/ImageObject",
        )

        image_transfer_hyperparams = ImageTransferPrimitive.metadata.get_hyperparams()
        primitive_volumes = ImageTransferPrimitive.metadata.get_volumes()
        volumes = {
            primitive_volumes[0]["key"]: os.getenv("D3MSTATICDIR")
            + "/"
            + primitive_volumes[0]["file_digest"]
        }
        image_transfer_primitive = ImageTransferPrimitive(
            hyperparams=image_transfer_hyperparams.defaults(), volumes=volumes
        )
        result = image_transfer_primitive.produce(inputs=images).value
        self.assertEqual(result.shape[0], 5)
        self.assertEqual(result.shape[1], 512)


if __name__ == "__main__":
    unittest.main()
