import logging
import os
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.modeling.metrics import classification_metrics, regression_metrics
from distil.modeling.text_classification import TextClassifierCV
from distil.utils import CYTHON_DEP

__all__ = ('TextClassifierPrimitive',)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    metric = hyperparams.Hyperparameter[str](
        default='f1',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )
    fast = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )


class Params(params.Params):
    _models: Optional[TextClassifierCV]
    _grid: Optional[Dict]


class TextClassifierPrimitive(base.PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    This primitive takes a dataframe containing input texts, performs TFIDF on this text, and then builds a classifier using
    these features.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '24f51246-7487-454e-8d69-7cdf289994d1',
            'version': '0.1.1',
            'name': "Text Classifier",
            'python_path': 'd3m.primitives.classification.text_classifier.DistilTextClassifier',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/text_classifier.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [

                metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        },
    )

    _FAST_GRIDS = {
        "classification": {
            "vect__ngram_range": [(1, 1)],
            "vect__max_features": [1000, ],
            "cls__C": [float(xx) for xx in np.logspace(-3, 1, 100)],
            "cls__class_weight": ['balanced', None],
        }
    }

    _FAST_FIT_ROWS = 15

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._grid = self._get_grid_for_metric() if self.hyperparams['fast'] else None

        self._model = TextClassifierCV(self.hyperparams['metric'], param_grid=self._grid)

        self.label_map = None

    def __getstate__(self) -> dict:
        state = PrimitiveBase.__getstate__(self)
        state['models'] = self._model
        state['grid'] = self._grid
        return state

    def __setstate__(self, state: dict) -> None:
        PrimitiveBase.__setstate__(self, state)
        self._model = state['models']
        self._grid = state['grid']

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        """ TODO: `TextReaderPrimivite` has a weird output format from `read_file_uri`
        to remain consistent with common primitives base `FileReaderPrimitive` """

        self._inputs = inputs

        # map labels instead of trying to force to int.
        col = outputs.columns[0]
        if len(pd.factorize(outputs[col])[1]) <= 2:
            factor = pd.factorize(outputs[col])
            outputs = pd.DataFrame(factor[0], columns=[col])
            self.label_map = {k: v for k, v in enumerate(factor[1])}

        self._outputs = outputs

    def _format_text(self, inputs):
        return inputs['filename'].values

    def _format_output(self, outputs):
        return outputs

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        if self.hyperparams['fast']:
            rows = self._inputs.shape[0]  # len(self._inputs.index)
            if rows > self._FAST_FIT_ROWS:
                sampled_inputs = self._inputs.sample(n=self._FAST_FIT_ROWS, random_state=1)
                sampled_outputs = self._outputs.loc[self._outputs.index.intersection(sampled_inputs.index),]
                self._model.fit(self._format_text(sampled_inputs), self._format_output(sampled_outputs))
        else:
            self._model.fit(self._format_text(self._inputs), self._format_output(self._outputs))

        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[
        container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # create dataframe to hold d3mIndex and result

        result = self._model.predict(self._format_text(inputs))
        result_df = container.DataFrame({self._outputs.columns[0]: result}, generate_metadata=True)
        # if we mapped values earlier map them back.
        if self.label_map:
            result_df[self._outputs.columns[0]] = result_df[self._outputs.columns[0]].map(self.label_map)

        # mark the semantic types on the dataframe
        # result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params(_models = self._model,
                      _grid = self._grid)

    def set_params(self, *, params: Params) -> None:
        self._model = params['_models']
        self._grid = params['_grid']

    def _get_grid_for_metric(self) -> Dict[str, Any]:
        if self.hyperparams['metric'] in classification_metrics:
            return self._FAST_GRIDS['classification']
        elif self.hyperparams['metric'] in regression_metrics:
            raise NotImplementedError
        else:
            raise Exception('ForestCV: unknown metric')
