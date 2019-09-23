import os
import sys
import logging
from typing import Set, List, Dict, Any, Optional

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from distil.modeling.forest import ForestCV
from distil.modeling.metrics import classification_metrics, regression_metrics

from ShapExplainers import tree

import pandas as pd
import numpy as np

from common_primitives import denormalize, dataset_to_dataframe as DatasetToDataFrame

__all__ = ('EnsembleForest',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Hyperparameter[str](
        default='f1Macro',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The D3M scoring metric to use during the fit phase.  This can be any of the regression, classification or " +
                    "clustering metrics."
    )
    small_dataset_threshold = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls the application of the 'small_dataset_fits' and 'large_dataset_fits' parameters - if the input dataset has " +
                    "fewer rows than the threshold value, 'small_dateset_fits' will be used when fitting.  Otherwise, 'num_large_fits' is used."
    )
    small_dataset_fits = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The number of random forests to fit when using small datasets."
    )
    large_dataset_fits = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="The number of random forests to fit when using large datasets."
    )
    samples = hyperparams.Set(
        elements = hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A list of indices for specific rows to return shap explainer info for."
    )
    pass

class Params(params.Params):
    pass

class EnsembleForestPrimitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    Generates an ensemble of random forests, with the number of internal models created controlled by the size of the
    input dataframe.  It accepts a dataframe as input, and returns a dataframe consisting of prediction values only as output.
    Columns with string structural types are ignored.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'e0ad06ce-b484-46b0-a478-c567e1ea7e02',
            'version': '0.1.0',
            'name': "EnsembleForest",
            'python_path': 'd3m.primitives.learner.random_forest.DistilEnsembleForest',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/ensemble_forest.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.LEARNER,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        PrimitiveBase.__init__(self, hyperparams=hyperparams, random_seed=random_seed)
        self._model = ForestCV(self.hyperparams['metric'])
        self._needs_fit = True

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._model
        state['needs_fit'] = self._needs_fit
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._model = state['models']
        self._needs_fit = True

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._model.num_fits = self.hyperparams['large_dataset_fits'] \
            if self._inputs.shape[0] > self.hyperparams['small_dataset_threshold'] else self.hyperparams['small_dataset_fits']
        self._needs_fit = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        if self._needs_fit:
            self._model.fit(self._inputs.values, self._outputs.values)
            self._needs_fit = False
        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # force a fit it hasn't yet been done
        if self._needs_fit:
            self.fit()

        # create dataframe to hold the result
        result = self._model.predict(inputs.values)
        result_df = container.DataFrame({self._outputs.columns[0]: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')
        return base.CallResult(result_df)

    def produce_feature_importances(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__} feature weights')

        # force a fit it hasn't yet been done
        if self._needs_fit:
            self.fit()

        # extract the feature weights
        column_names = inputs.columns
        output = container.DataFrame(self._model.feature_importances().reshape((1, len(inputs.columns))), generate_metadata=True)
        output.columns = inputs.columns
        for i in range(len(inputs.columns)):
            output.metadata = output.metadata.update_column(i, {"name": output.columns[i]})

        print(output)

        return CallResult(output)

    def produce_shap_values(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:

        index = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        index_names = [list(inputs)[i] for i in index]
                
        if self._needs_fit:
            self.fit()

        #get the task type from the model instance
        task_type = self._model.mode
                        
        #shap needs a pandas type dataframe, not d3 container type dataframe
        inputs[index_names[0]] = inputs[index_names[0]].astype(int)
        shap_df = pd.DataFrame(inputs.set_index([index_names[0]]))
        
        exp = tree.Tree(self._model._models[0].model, X = shap_df, model_type = 'Random_Forest', task_type = task_type)
        
        if len(self.hyperparams['samples']) == 0:
            output_df = container.DataFrame(exp.produce_global(), generate_metadata = True)
            
        else:
            samples = inputs[inputs[index_names[0]].isin(self.hyperparams['samples'])].index
            output_df = container.DataFrame(exp.produce_sample(samples), generate_metadata=True)
        
        output_df.reset_index(level=0, inplace = True)
        
        #build and add metadata for index
        col_dict = dict(output_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type(1)
        col_dict['name'] = index_names[0]
        col_dict['semantic_types'] = ('http://schema.org/Int', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        output_df.metadata = output_df.metadata.update((metadata_base.ALL_ELEMENTS,0),col_dict)

        #metadata for columns
        for c in range(1, len(output_df.columns)):
            col_dict = dict(output_df.metadata.query((metadata_base.ALL_ELEMENTS, c)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name'] = output_df.columns[c]
            col_dict['semantic_type'] = ('https://metadata.datadrivendiscovery.org/types/Attribute',)
            output_df.metadata = output_df.metadata.update((metadata_base.ALL_ELEMENTS,c),col_dict)
        
        df_dict = dict(output_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
        df_dict_1 = dict(output_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
        df_dict['dimension'] = df_dict_1
        df_dict_1['name'] = 'columns'
        df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
        df_dict_1['length'] =len(inputs.columns)     
        output_df.metadata = output_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
     
        return CallResult(output_df)
        
        
    def get_params(self) -> Params:
        return Params()

    def set_params(self, *, params: Params) -> None:
        return

if __name__ == '__main__':

    #Load data and preprocessing
    
    filepath = 'file:///home/alexmably/datasets/seed_datasets_current/196_autoMpg/TRAIN/dataset_TRAIN/datasetDoc.json'
    test_filepath = 'file:///home/alexmably/datasets/seed_datasets_current/196_autoMpg/TEST/dataset_TEST/datasetDoc.json'

    input_dataset = container.Dataset.load(filepath)
    test_dataset = container.Dataset.load(test_filepath)

    hyperparams_class = denormalize.DenormalizePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    denorm = denormalize.DenormalizePrimitive(hyperparams = hyperparams_class.defaults())
    input_dataset = denorm.produce(inputs = input_dataset).value
    test_dataset = denorm.produce(inputs = test_dataset).value

    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
    input_dataframe = container.DataFrame(ds2df_client.produce(inputs = input_dataset).value)
    test_dataframe = container.DataFrame(ds2df_client.produce(inputs = test_dataset).value)
    input_dataframe = input_dataframe.replace({'':np.nan}).dropna()
    test_dataframe = test_dataframe.drop(columns = ['class'])
    test_dataframe = test_dataframe.replace({'':np.nan}).dropna()
    
    input_dataframe = input_dataframe.iloc[:150] 
        
    hyperparams_class = EnsembleForestPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    tree_client = EnsembleForestPrimitive(hyperparams=hyperparams_class.defaults().replace({'metric':'meanSquaredError', 'samples':[3,5]}))
    tree_client.set_training_data(inputs = input_dataframe.drop(columns = ['class']), outputs = input_dataframe['class'].to_frame())
    tree_client.fit()
      
    results = tree_client.produce_shap_values(inputs = test_dataframe)
    print(results.value)
    #print(input_dataframe)