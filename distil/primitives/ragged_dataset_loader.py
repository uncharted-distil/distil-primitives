import os
import logging
import copy

from common_primitives import utils as common_utils
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.primitives import utils

from typing import List, Sequence, Optional
import numpy as np
import pandas as pd

__all__ = ('RaggedDatasetLoader',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    collection_type = hyperparams.Hyperparameter[str](
        default='timeseries',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='the type of collection to load')
    sample = hyperparams.Hyperparameter[float](
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='a value ranging from 0.0 to 1.0 indicating how much of the source data to load'
    )


class RaggedDatasetLoaderPrimitive(transformer.TransformerPrimitiveBase[container.Dataset, container.List, Hyperparams]):
    """
    A primitive that loads ragged datasets.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'c3f9e9c5-7d16-4608-a875-d59147d12f39',
            'version': '0.1.0',
            'name': "Load ragged collection from dataset into a single dataframe",
            'python_path': 'd3m.primitives.data_transformation.data_cleaning.DistilRaggedDatasetLoader',
            'source': {
                'name': 'distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/ragged_dataset_loader.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )


    def produce(self, *, inputs: container.Dataset, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Running {__name__}')

        # get the learning data (the dataset entry point)
        learning_id, learning_df = common_utils.get_tabular_resource(inputs, None, pick_entry_point=True)
        learning_df = learning_df.head(int(learning_df.shape[0]*self.hyperparams['sample']))
        learning_df.metadata = self._update_metadata(inputs.metadata, learning_id, learning_df)

        logger.debug(f'\n{learning_df}')

        return base.CallResult(learning_df)


    def produce_collection(self, *, inputs: container.Dataset, timeout: float = None, iterations: int = None) -> base.CallResult[container.List]:
        logger.debug(f'Running {__name__}')

        # get the learning data (the dataset entry point)
        learning_id, learning_df = common_utils.get_tabular_resource(inputs, None, pick_entry_point=True)
        learning_df = learning_df.head(int(learning_df.shape[0]*self.hyperparams['sample']))
        learning_df.metadata = self._update_metadata(inputs.metadata, learning_id, learning_df)

        # find the column that is acting as the foreign key and extract the resource + column it references
        for i in range(learning_df.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            column_metadata = learning_df.metadata.query_column(i)
            if 'foreign_key' in column_metadata and column_metadata['foreign_key']['type'] is 'COLUMN':
                resource_id = column_metadata['foreign_key']['resource_id']
                file_column_idx = column_metadata['foreign_key']['column_index']

        # get the learning data (the dataset entry point)
        collection_id, collection_df = common_utils.get_tabular_resource(inputs, resource_id)
        collection_df = collection_df.head(learning_df.shape[0])
        collection_df.metadata = self._update_metadata(inputs.metadata, collection_id, collection_df)

        # get the base path
        base_path = collection_df.metadata.query((metadata_base.ALL_ELEMENTS, file_column_idx))['location_base_uris'][0]

        # create fully resolved paths and load
        paths = collection_df.iloc[:, file_column_idx]
        file_paths = [ os.path.join(base_path, p) for p in paths ]
        outputs = self._timeseries_load(file_paths)

        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    @classmethod
    def _update_metadata(cls, metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment,
                         for_value: Optional[container.DataFrame]) -> metadata_base.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update({'schema': metadata_base.CONTAINER_SCHEMA_VERSION,})
        new_metadata = metadata.clear(resource_metadata, for_value=for_value, generate_metadata=False)
        new_metadata = common_utils.copy_metadata(metadata, new_metadata, (resource_id,))
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata

    @classmethod
    def _timeseries_load(cls, paths: Sequence[str]) -> List:
        tmp = [pd.read_csv(p) for p in paths]
        tmp = [t.values[:,1] for t in tmp]
        return tmp