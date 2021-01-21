import logging
import os
import math
from typing import List, Optional
import typing

import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, unsupervised_learning

from sklearn.ensemble import IsolationForest

from distil.utils import CYTHON_DEP
import version

__all__ = ("PrefeaturisedPoolingPrimitive",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    n_jobs = hyperparams.Hyperparameter[int](
        default=64,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The value of the n_jobs parameter for the joblib library",
    )
    n_estimators = hyperparams.Hyperparameter[int](
        default=100,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="The amount of ensembles used in the primitive.",
    )


class Params(params.Params):
    model: IsolationForest
    needs_fit: bool


class IsolationForestPrimitive(
    unsupervised_learning.UnsupervisedLearnerPrimitiveBase[
        container.DataFrame, container.DataFrame, Params, Hyperparams
    ]
):
    """
    Uses scikit learn's Isolated Forest primitive to detect and label anomalies.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "793f0b17-7413-4962-9f1d-0b285540b21f",
            "version": version.__version__,
            "name": "Isolation Forest",
            "python_path": "d3m.primitives.classification.isolation_forest.IsolationForestPrimitive",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/primitives/isolation_forest.py",
                    "https://github.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                CYTHON_DEP,
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.BINARY_CLASSIFICATION,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._model = IsolationForest(
            n_estimators=self.hyperparams["n_estimators"],
            random_state=np.random.RandomState(random_seed),
        )

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._inputs = inputs
        self._needs_fit = True

    def fit(
        self, *, timeout: float = None, iterations: int = None
    ) -> base.CallResult[None]:
        logger.debug(f"Fitting {__name__}")

        if self._needs_fit:
            self._model.fit(self._inputs)
            self._needs_fit = False

        return base.CallResult(None)

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[container.DataFrame]:

        if self._needs_fit:
            self.fit()

        index = inputs.index
        result = self._model.predict(inputs)

        result_df = container.DataFrame(
            {
                "d3mIndex": index,
                "outlier_label": result,
            },
            generate_metadata=True,
        )
        result_df.set_index(index, inplace=True)

        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )

        return base.CallResult(result_df)

    def get_params(self) -> Params:
        return Params(
            model=self._model,
            needs_fit=self._needs_fit,
        )

    def set_params(self, *, params: Params) -> None:
        self._model = params["model"]
        self._needs_fit = params["needs_fit"]
        return
