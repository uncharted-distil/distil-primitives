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
import sys

from d3m import container

from d3m.metadata import base as metadata_base
from distil.primitives.text_encoder import TextEncoderPrimitive
from distil.primitives import utils
import utils as test_utils


class TextEncoderPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(
        path.join(path.dirname(__file__), "text_encoder_dataset")
    )

    def test_defaults(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # create the encoder
        hyperparams_class = TextEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = TextEncoderPrimitive(hyperparams=hyperparams_class.defaults())
        encoder.set_training_data(
            inputs=dataframe.iloc[:, [0, 1]], outputs=dataframe[['bravo']]
        )
        encoder.fit()
        result = encoder.produce(inputs=dataframe).value

        # don't assert on invidual values - just check basic sanity of result
        self.assertEqual(len(result.index), 9)
        self.assertEqual(
            result.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/Attribute",)
            ),
            [1, 2, 3, 4],
        )
        self.assertSequenceEqual(
            list(result.columns),
            ["d3mIndex", "bravo", "__text_0", "__text_1", "__text_2"],
        )
        self.assertSequenceEqual(
            result.dtypes.tolist(), [object, object, float, float, float]
        )

    # def test_empty_col(self) -> None:
    #     dataset = test_utils.load_dataset('/Users/vkorapaty/data/datasets/seed_datasets_current/JIDO_SOHR_Tab_Articles_8569/TRAIN/dataset_TRAIN/')
    #     dataframe = test_utils.get_dataframe(dataset, 'learningData')

    #     hyperparams_class = \
    #         TextEncoderPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    #     encoder = TextEncoderPrimitive(hyperparams=hyperparams_class.defaults().replace({
    #         'use_columns': [0, 3, 4, 5, 6, 7, 8]
    #     }))
    #     encoder.set_training_data(inputs=dataframe, outputs=dataframe.iloc[:, 1])
    #     encoder.fit()

    def test_classification_binary_label(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe = dataframe.iloc[0:5]

        # create the encoder
        hyperparams_class = TextEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = TextEncoderPrimitive(hyperparams=hyperparams_class.defaults())
        encoder.set_training_data(
            inputs=dataframe.iloc[:, [0, 1]], outputs=dataframe[['bravo']]
        )
        encoder.fit()
        result = encoder.produce(inputs=dataframe).value

        # don't assert on invidual values - just check basic sanity of result
        self.assertEqual(len(result.index), 5)
        self.assertEqual(
            result.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/Attribute",)
            ),
            [1, 2],
        )
        self.assertSequenceEqual(
            list(result.columns), ["d3mIndex", "bravo", "__text_0"]
        )
        self.assertSequenceEqual(result.dtypes.tolist(), [object, object, float])

    def test_classification_singleton_label(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")
        dataframe = dataframe.iloc[0:6]

        # create the encoder
        hyperparams_class = TextEncoderPrimitive.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        encoder = TextEncoderPrimitive(hyperparams=hyperparams_class.defaults())
        encoder.set_training_data(
            inputs=dataframe.iloc[:, [0, 1]], outputs=dataframe[['bravo']]
        )

        # should fail in this case because we have a label with a cardinality of 1
        self.assertRaises(ValueError, encoder.fit)


if __name__ == "__main__":
    unittest.main()
