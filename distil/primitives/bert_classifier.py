import logging
import os
from typing import Dict, Optional

import torch
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.modeling.bert_models import BERTPairClassification
from distil.utils import CYTHON_DEP

_all__ = ('BertPairClassification',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    doc_col_0 = hyperparams.Hyperparameter[int](
        default=0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The index of the column containing the first documents in the classification pairs.'
    )
    doc_col_1 = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='The index of the column containing the second documents in the classification pairs.'
    )
    force_cpu = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Force CPU execution regardless of GPU availability.'
    )
    batch_size = hyperparams.Hyperparameter[int](
        default=32,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='Number of samples to load in each training batch.'
    )
    epochs = hyperparams.Hyperparameter[int](
        default=3,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='The number of passes to make over the training set.'
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default=5e-5,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description='The change in the model in reponse to estimated error.'
    )

class Params(params.Params):
    model: BERTPairClassification
    target_col: str


class BertPairClassificationPrimitive(PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    Uses a pre-trained pytorch BERT model to predict a label of 0 or 1 for a pair of documents, given training samples
    of document pairs labelled 0/1.  Takes a datrame of documents and a dataframe of labels as inputs, and returns
    a dataframe containing the predictions as a result.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '7c305f3a-442a-41ad-b9db-8c437753b119',
            'version': '0.1.1',
            'name': "BERT pair classification",
            'python_path': 'd3m.primitives.classification.bert_classifier.DistilBertPairClassification',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/bert_classifier.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            },
            {
                    "type": "FILE",
                    "key": "bert-base-uncased-model",
                    "file_uri": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
                    "file_digest": "57f8763c92909d8ab1b0d2a059d27c9259cf3f2ca50f7683edfa11aee1992a59",
            },
            {
                    "type": "FILE",
                    "key": "bert-base-uncased-vocab",
                    "file_uri": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
                    "file_digest": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.BERT,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.CLASSIFICATION,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0,
                 volumes: Dict[str, str] = None) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        self._volumes = volumes
        self._model: Optional[BERTPairClassification] = None
        self._target_col: str = ""


    def set_training_data(self, *, inputs: container.DataFrame, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._target_col = self._outputs.columns[0]

    def fit(self, *, timeout: float = None, iterations: int = None) -> base.CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        # lazy init because we needed data to be set
        if not self._model:
            columns = (self._inputs.columns[self.hyperparams['doc_col_0']], self._inputs.columns[self.hyperparams['doc_col_1']])
            if torch.cuda.is_available():
                if self.hyperparams['force_cpu']:
                    logger.info("Detected CUDA support - forcing use of CPU")
                    device = "cpu"
                else:
                    logger.info("Detected CUDA support - using GPU")
                    device = "cuda"
            else:
                logger.info("CUDA does not appear to be supported - using CPU.")
                device = "cpu"

            if self._volumes:
                model_path = self._volumes['bert-base-uncased-model']
                vocab_path = self._volumes['bert-base-uncased-vocab']
            else:
                raise ValueError("No volumes supplied for primitive - static models cannot be loaded.")

            self._model = BERTPairClassification(model_path=model_path, vocab_path=vocab_path, device=device, columns=columns,
                epochs=self.hyperparams['epochs'], batch_size=self.hyperparams['batch_size'], learning_rate=self.hyperparams['learning_rate'])

        self._model.fit(self._inputs, self._outputs)
        return base.CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        inputs = inputs

        # create dataframe to hold result
        if self._model is None:
            raise ValueError("No model available for primitive")
        result = self._model.predict(inputs)

        # use the original saved target column name
        result_df = container.DataFrame({self._target_col: result}, generate_metadata=True)

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        logger.debug(f'\n{result_df}')

        print(result_df)

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params(
            model=self._model,
            target_col=self._target_col
        )

    def set_params(self, *, params: Params) -> None:
        self._model = params['model']
        self._target_col = params['target_col']
        return
