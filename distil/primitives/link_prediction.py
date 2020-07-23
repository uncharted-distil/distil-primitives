import logging
import os

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.modeling.link_prediction import RescalLinkPrediction
from distil.modeling.metrics import classification_metrics
from distil.utils import CYTHON_DEP
import version

__all__ = ('LinkPrediction',)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Enumeration[str](
        values=classification_metrics + ['rootMeanSquaredError'],
        default='accuracy',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter']
    )

class Params(params.Params):
    model: RescalLinkPrediction
    target_col: str


class DistilLinkPredictionPrimitive(PrimitiveBase[container.List, container.DataFrame, Params, Hyperparams]):
    """
    A primitive that uses RESCAL to predict links in graphs.
    """
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'fc138210-c317-4528-81ae-5eed3a1a0267',
            'version': version.__version__,
            'name': "LinkPrediction",
            'python_path': 'd3m.primitives.link_prediction.link_prediction.DistilLinkPrediction',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/link_prediction.py',
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
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.LINK_PREDICTION,
        },
    )

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._model = RescalLinkPrediction(target_metric=self.hyperparams['metric'], random_seed=random_seed)
        self._target_col = ""

    def set_training_data(self, *, inputs: container.List, outputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._outputs = outputs
        self._target_col = outputs.columns[0]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f'Fitting {__name__}')

        X_train, y_train, U_train = self._inputs
        X_train = X_train.value
        y_train = y_train.squeeze()
        self._model.fit(X_train, y_train, U_train)

        return CallResult(None)

    def produce(self, *, inputs: container.List, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        logger.debug(f'Producing {__name__}')

        X_train, _, _ = inputs
        X_train = X_train.value
        result = self._model.predict(X_train).astype(int)

        # create dataframe to hold d3mIndex and result
        result_df = container.DataFrame({X_train.index.name: X_train.index, self._target_col: result})

        # mark the semantic types on the dataframe
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 1), 'https://metadata.datadrivendiscovery.org/types/PredictedTarget')

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params(
            model=self._model,
            target_col=self._target_col
        )

    def set_params(self, *, params: Params) -> None:
        self._model=params['model']
        self._target_col=params['target_col']

