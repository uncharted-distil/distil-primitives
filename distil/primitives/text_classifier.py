import logging
import os
from typing import Dict, Any, List

import pandas as pd
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from distil.modeling.metrics import classification_metrics
from distil.modeling.text_classification import TextClassifierCV
from distil.utils import CYTHON_DEP

__all__ = ('TextClassifierPrimitive',)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Enumeration[str](
        values=classification_metrics,
        default='f1',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Params(params.Params):
    model: TextClassifierCV
    label_map: Dict[int, str]
    target_col_names: List[str]



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

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._model = TextClassifierCV(self.hyperparams['metric'], random_seed=random_seed)
        self._label_map: Dict[int, str] = {}

    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        """ TODO: `TextReaderPrimitive` has a weird output format from `read_file_uri`
        to remain consistent with common primitives base `FileReaderPrimitive` """

        self._inputs = inputs
        self._target_col_names = list(outputs.columns)

        # map labels instead of trying to force to int.
        col = outputs.columns[0]
        if len(pd.factorize(outputs[col])[1]) <= 2:
            factor = pd.factorize(outputs[col])
            outputs = pd.DataFrame(factor[0], columns=[col])
            self._label_map = {k: v for k, v in enumerate(factor[1])}

        self._outputs = outputs

    def _format_text(self, inputs):
        return inputs['filename'].values

    def _format_output(self, outputs):
        return outputs.values.ravel(order='C')

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')
        self._model.fit(self._format_text(self._inputs), self._format_output(self._outputs))
        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[
        container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        # create dataframe to hold d3mIndex and result
        result = self._model.predict(self._format_text(inputs))
        df = pd.DataFrame(result)

        # pipline run saving is now getting fussy about the prediction names matching the original target column
        # name
        df.columns = self._target_col_names

        #if we mapped values earlier map them back.
        if self._label_map:
            df.replace(self._label_map, inplace=True)
        result_df = container.DataFrame(df, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0),
                                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params(
            model=self._model,
            label_map=self._label_map,
            target_col_names=self._target_col_names
        )

    def set_params(self, *, params: Params) -> None:
        self._model = params['model']
        self._label_map = params['label_map']
        self._target_col_names = params['target_col_names']
        return
