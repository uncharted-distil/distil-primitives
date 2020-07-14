"""
   Copyright © 2018 Uncharted Software Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import unittest
from os import path
import csv
import typing
import pandas as pd
import numpy as np

from d3m import container
from distil.primitives.mi_ranking import MIRankingPrimitive as MIRanking
from d3m.metadata import base as metadata_base


class MIRankingPrimitiveTestCase(unittest.TestCase):

    _dataset_path = path.abspath(path.join(path.dirname(__file__), 'tabular_dataset_2'))

    def test_discrete_target(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 1
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [2, 5, 3])
        self.assertListEqual(list(result_dataframe['name']), ['bravo', 'echo', 'charlie'])
        expected_ranks = [1.0, 1.0, 0.0]
        for i, r in enumerate(result_dataframe['rank']):
            self.assertAlmostEqual(r, expected_ranks[i], places=6)

    def test_continuous_target(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 2
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [1, 5, 3])
        self.assertListEqual(list(result_dataframe['name']), ['alpha', 'echo', 'charlie'])
        expected_ranks = [1.0, 0.930536, 0.0]
        for i, r in enumerate(result_dataframe['rank']):
            self.assertAlmostEqual(r, expected_ranks[i], places=6)

    def test_full_mutual_info(self) -> None:
        dataframe = self._load_data(alpha_class=float, charlie_class=float)

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 2
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        # dataframe.drop(columns=['charlie', 'delta', 'echo'], inplace=True)
        # dataframe['bravo'][dataframe['bravo'] == 4] = 3
        dataframe['bravo'] = [1, 0.5, 10, 6.7, 6.9, 2.3, 5.5, 7.3, 9]
        dataframe['alpha'] = [1, 0.5, 10, 6.7, 6.9, 2.3, 5.5, 7.3, 9]
        dataframe['charlie'] = [2, 2.5, 9, 5.7, 7.9, 1.3, 6.5, 8.3, 8]
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [1, 3, 5])
        self.assertListEqual(list(result_dataframe['name']), ['alpha', 'charlie', 'echo'])
        expected_ranks = [1.0, 0.665361, 0.0]
        for i, r in enumerate(result_dataframe['rank']):
            self.assertAlmostEqual(r, expected_ranks[i], places=6)

    def test_incompatible_target(self) -> None:
        dataframe = self._load_data(bad_features=[4])

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 4
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [])
        self.assertListEqual(list(result_dataframe['name']), [])
        self.assertListEqual(list(result_dataframe['rank']), [])

    def test_incompatible_features(self) -> None:
        dataframe = self._load_data(bad_features=[2, 3, 5])

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 1
            }
        )
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [])
        self.assertListEqual(list(result_dataframe['name']), [])
        self.assertListEqual(list(result_dataframe['rank']), [])


    def test_unique_categorical_removed(self) -> None:
        dataframe = self._load_data()

        hyperparams_class = \
            MIRanking.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        hyperparams = hyperparams_class.defaults().replace(
            {
                'target_col_index': 1
            }
        )
        dataframe.insert(len(dataframe.columns), 'removed', [1, 2, 3, 4, 5, 6, 7, 8, 9])
        dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        mi_ranking = MIRanking(hyperparams=hyperparams)
        result_dataframe = mi_ranking.produce(inputs=dataframe).value

        # verify the output
        self.assertListEqual(list(result_dataframe['idx']), [2, 5, 3])
        self.assertListEqual(list(result_dataframe['name']), ['bravo', 'echo', 'charlie'])

    def _load_data(cls, bad_features: typing.Sequence[int]=[], alpha_class=int, charlie_class=int) -> container.DataFrame:
        dataset_doc_path = path.join(cls._dataset_path, 'datasetDoc.json')

        # load the dataset and convert resource 0 to a dataframe
        dataset = container.Dataset.load('file://{dataset_doc_path}'.format(dataset_doc_path=dataset_doc_path))
        dataframe = dataset['learningData']
        dataframe.metadata = dataframe.metadata.generate(dataframe)

        # set the struct type
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 0),
                                                        {'structural_type': int})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 1),
                                                        {'structural_type': alpha_class})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 2),
                                                        {'structural_type': float})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 3),
                                                        {'structural_type': charlie_class})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 4),
                                                        {'structural_type': str})
        dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, 5),
                                                        {'structural_type': int})

        # set the semantic type
        if alpha_class == int:
            dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, 1),
                                'https://metadata.datadrivendiscovery.org/types/CategoricalData')
        elif alpha_class == float:
            dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'http://schema.org/Float')

        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 2), 'http://schema.org/Float')
        if charlie_class == int:
            dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, 3), 'http://schema.org/Boolean')
        elif charlie_class == float:
            dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, 3), 'http://schema.org/Float')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 4), 'http://schema.org/Text')
        dataframe.metadata = dataframe.metadata.\
            add_semantic_type((metadata_base.ALL_ELEMENTS, 5), 'http://schema.org/Integer')

        # override with incompatible features
        for i in bad_features:
            dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, i),
                                                            {'structural_type': str})
        for i in bad_features:
            dataframe.metadata = dataframe.metadata.update((metadata_base.ALL_ELEMENTS, i),
                                                            {'semantic_types': ['http://schema.org/Text']})

        # set the roles
        for i in range(1, 6):
            dataframe.metadata = dataframe.metadata.\
                add_semantic_type((metadata_base.ALL_ELEMENTS, i),
                                  'https://metadata.datadrivendiscovery.org/types/Attribute')

        # handle the missing data as a NaN
        dataframe = dataframe.replace(r'^\s*$', np.nan, regex=True)

        # cast the dataframe to raw python types
        dataframe['d3mIndex'] = dataframe['d3mIndex'].astype(int)
        dataframe['alpha'] = dataframe['alpha'].astype(alpha_class)
        dataframe['bravo'] = dataframe['bravo'].astype(float)
        dataframe['charlie'] = dataframe['charlie'].astype(alpha_class)
        dataframe['delta'] = dataframe['delta'].astype(str)
        dataframe['echo'] = dataframe['echo'].astype(float)

        return dataframe


if __name__ == '__main__':
    unittest.main()
