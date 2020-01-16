import os
import typing

from typing import List, Sequence


from d3m import container, utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP

import pandas as pd

import common_primitives
from common_primitives.utils import list_columns_with_semantic_types
from common_primitives import dataset_to_dataframe

__all__ = ('DistilEdgeListLoaderPrimitive',)

Inputs = container.Dataset
Outputs = container.List

import logging
import networkx as nx

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    dataframe_resource = hyperparams.Hyperparameter[typing.Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=".",
    )


class DistilEdgeListLoaderPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A primitive which passes both the networkX loaded graph object and
    th associated dataframe to the next primitive.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '4acc81e5-4b9c-443e-a72a-18dd9a7dcc3b',
            'version': '0.1.2',
            'name': "Load edgelist into a parseable object",
            'python_path': 'd3m.primitives.data_transformation.load_edgelist.DistilEdgeListLoader',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/load_edge_list.py',
                    'https://github.com/uncharted-distil/distil-primitives/',
                ],
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:

        dataframe_resource_id, dataframe = base_utils.get_tabular_resource(inputs,
                                                                           self.hyperparams['dataframe_resource'])        #get attribute columns

        hyperparams_class = \
            dataset_to_dataframe.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments'][
                'Hyperparams']
        primitive = dataset_to_dataframe.DatasetToDataFramePrimitive(hyperparams=hyperparams_class.defaults())

        dataframe_meta = primitive.produce(inputs=inputs).value

        attributes = list_columns_with_semantic_types(metadata=dataframe_meta.metadata, semantic_types=[
            'https://metadata.datadrivendiscovery.org/types/Attribute'])


        base_file_path ='/'.join(inputs.metadata._current_metadata.metadata['location_uris'][0].split('/')[:-1])
        edge_list = pd.read_csv(os.path.join(base_file_path, 'graphs', 'edgeList.csv'), index_col=0)
        if len(edge_list.columns) > 2:
            graph = nx.from_pandas_edgelist(edge_list, source=edge_list.columns[0], target=edge_list.columns[1], edge_attr=edge_list.columns[2])
        else:
            graph = nx.from_pandas_edgelist(edge_list, source=edge_list.columns[0], target=edge_list.columns[1])

        if len(attributes) > 1:
            # add attributers to nodes.
            attribute_node_map = dataframe_meta[dataframe_meta.columns[attributes]]
            attribute_node_map['nodeID'] = attribute_node_map['nodeID'].astype(int)
            attribute_node_map.index = attribute_node_map['nodeID']
            attribute_cols = attribute_node_map.columns
            attribute_node_map.drop(['nodeID'], axis=1)
            attribute_node_map = attribute_node_map.to_dict(orient='index')

            for i in graph.nodes:
                default = {attribute: 0 for attribute in attribute_cols}
                default['nodeID'] = i
                graph.nodes[i].update(attribute_node_map.get(i, default))

        else:
            # featurizer expects at a minimum nodeids to be present
            for i in graph.nodes:
                default = {}
                default['nodeID'] = i
                graph.nodes[i].update(default)
        # int2str_map = dict(zip(graph.nodes, [str(n) for n in graph.nodes]))
        # graph = nx.relabel_nodes(graph, mapping=int2str_map)


        dataframe.metadata = self._update_metadata(inputs.metadata, dataframe_resource_id)

        assert isinstance(dataframe, container.DataFrame), type(dataframe)

        U_train = {'graph': graph}
        y_train = self.produce_target(inputs=inputs).value
        X_train = dataframe # TODO use attribute in vertex classification

        X_train = self._typify_dataframe(X_train)
        X_train.value = pd.DataFrame(X_train.value['nodeID'])
        return base.CallResult([X_train, y_train, U_train])

    def _typify_dataframe(self, df):
        outputs = df.copy()

        num_cols = outputs.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        remove_indices = []
        target_idx = -1
        suggested_target_idx = -1
        for i in range(num_cols):
            semantic_types = outputs.metadata.query((metadata_base.ALL_ELEMENTS,i))['semantic_types']
            # mark target + index for removal
            if 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types or \
                'https://metadata.datadrivendiscovery.org/types/TrueTarget' in semantic_types or \
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in semantic_types:
                target_idx = i
                remove_indices.append(i)
            elif 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types:
                suggested_target_idx = i

            # update the structural / df type from the semantic type
            outputs = self._update_type_info(semantic_types, outputs, i)

        # fallback on suggested target if no true target / target marked
        if target_idx == -1:
            target_idx = suggested_target_idx
            remove_indices.append(target_idx)

        # flip the d3mIndex to be the df index as well
        outputs = outputs.set_index('d3mIndex', drop=False)

        # remove target and primary key
        outputs = outputs.remove_columns(remove_indices)

        logger.debug(f'\n{outputs.dtypes}')
        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    def produce_target(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Running {__name__} produce_target')

        _, dataframe = base_utils.get_tabular_resource(inputs, self.hyperparams['dataframe_resource'])
        outputs = dataframe.copy()

        # find the target column and remove all others
        num_cols = outputs.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']
        target_idx = -1
        suggested_target_idx = -1
        for i in range(num_cols):
            semantic_types = outputs.metadata.query((metadata_base.ALL_ELEMENTS,i))['semantic_types']
            if 'https://metadata.datadrivendiscovery.org/types/Target' in semantic_types or \
               'https://metadata.datadrivendiscovery.org/types/TrueTarget' in semantic_types:
                target_idx = i
                outputs = self._update_type_info(semantic_types, outputs, i)
            elif 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types:
                suggested_target_idx = i
            elif 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in semantic_types:
                outputs = self._update_type_info(semantic_types, outputs, i)
        # fall back on suggested target
        if target_idx == -1:
            target_idx = suggested_target_idx

        # flip the d3mIndex to be the df index as well
        outputs = outputs.set_index('d3mIndex', drop=False)

        remove_indices = set(range(num_cols))
        remove_indices.remove(target_idx)
        outputs = outputs.remove_columns(remove_indices)

        logger.debug(f'\n{outputs.dtypes}')
        logger.debug(f'\n{outputs}')

        return base.CallResult(outputs)

    @classmethod
    def _update_metadata(cls, metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment) -> metadata_base.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update(
            {
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            },
        )

        new_metadata = metadata_base.DataMetadata(resource_metadata)

        new_metadata = metadata.copy_to(new_metadata, (resource_id,))

        # Resource is not anymore an entry point.
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata

    @classmethod
    def _update_type_info(self, semantic_types: Sequence[str], outputs: container.DataFrame, i: int) -> container.DataFrame:
        # update the structural / df type from the semantic type
        if 'http://schema.org/Integer' in semantic_types:
            outputs.metadata = outputs.metadata.update_column(i, {'structural_type': int})
            outputs.iloc[:,i] = pd.to_numeric(outputs.iloc[:,i])
        elif 'http://schema.org/Float' in semantic_types:
            outputs.metadata = outputs.metadata.update_column(i, {'structural_type': float})
            outputs.iloc[:,i] = pd.to_numeric(outputs.iloc[:,i])
        elif 'http://schema.org/Boolean' in semantic_types:
            outputs.metadata = outputs.metadata.update_column(i, {'structural_type': bool})
            outputs.iloc[:,i] = outputs.iloc[:,i].astype('bool')

        return outputs
