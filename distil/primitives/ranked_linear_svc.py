import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.core.computation.pytables import ConditionBinOp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import rankdata
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.utils import CYTHON_DEP
import version

__all__ = ("RankedLinearSVC",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    penalty = hyperparams.Enumeration[str](
        default="l2",
        values=("l1", "l2"),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The type of regularization for loss.",
    )
    loss = hyperparams.Enumeration[str](
        default="squared_hinge",
        values=("squared_hinge", "hinge"),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The type of loss function.",
    )
    rank_confidences = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Returns confidences as ranks.",
    )
    tolerance = hyperparams.Hyperparameter[float](
        default=1e-4,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="Tolerance for error. Aims to stop within th is tolerance",
    )
    scaling = hyperparams.Enumeration[Optional[str]](
        default=None,
        values=("standardize", "unit_norm", None),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="How to scale the data before running SVC.",
    )
    calibrate = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Calibrates probabilities for confidence.",
    )
    confidences = hyperparams.Hyperparameter[bool](
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Whether or not to calculate confidences.",
    )


class Params(params.Params):
    model: LinearSVC
    target_cols: List[str]
    needs_fit: bool
    binary: bool
    standard_scaler: Optional[StandardScaler]


class RankedLinearSVCPrimitive(
    PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]
):
    """
    Classifies data using an SVM with a linear kernel. It also provides a confidence rank for each classification,
    and provides a dataframe with those two columns.
    Parameters
    ----------
    inputs : A container.Dataframe with columns containing numeric data.
    Returns
    -------
    output : A DataFrame containing (target value, confidence ranking) tuples for each input row.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "10d21bbe-9c58-4dc1-8f71-2b3834b71a5e",
            "version": version.__version__,
            "name": "DistilRankedLinearSVC",
            "python_path": "d3m.primitives.classification.linear_svc.DistilRankedLinearSVC",
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/distil/primitives/ranked_linear_svc.py",
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
                metadata_base.PrimitiveAlgorithmType.SUPPORT_VECTOR_MACHINE,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._model = LinearSVC(
            penalty=self.hyperparams["penalty"],
            loss=self.hyperparams["loss"],
            tol=self.hyperparams["tolerance"],
            random_state=random_seed,
        )
        self._needs_fit = True
        self._binary = False
        self._standard_scaler: StandardScaler = None

    def set_training_data(
        self, *, inputs: container.DataFrame, outputs: container.DataFrame
    ) -> None:
        if self.hyperparams["scaling"] == "standardize":
            self._standard_scaler = StandardScaler()
            self._inputs = self._standard_scaler.fit_transform(inputs.values)
        elif self.hyperparams["scaling"] == "unit_norm":
            self._inputs = inputs.values
            self._inputs = normalize(self._inputs)
        else:
            self._inputs = inputs.values
        self._outputs = outputs
        self._needs_fit = True
        self._target_cols: List[str] = []
        self._binary = self._outputs.iloc[:, 0].nunique(dropna=True) <= 2

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f"Fitting {__name__}")

        if self._needs_fit:
            self._model.fit(self._inputs, self._outputs.values.ravel())
            self._needs_fit = False
        return CallResult(None)

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> CallResult[container.DataFrame]:
        logger.debug(f"Producing {__name__}")

        # force a fit it hasn't yet been done
        if self._needs_fit:
            self.fit()
        if not self._target_cols:
            self._target_cols = [self._outputs.columns[0]]

        result: pd.DataFrame = None
        index = inputs.index
        inputs = inputs.values
        # create dataframe to hold the result
        if self.hyperparams["scaling"] == "standarize":
            inputs = self._standard_scaler.transform(inputs)
        elif self.hyperparams["scaling"] == "unit_norm":
            inputs = normalize(inputs)
        result = self._model.predict(inputs)
        result_df: container.DataFrame = None
        if self._binary:
            if self.hyperparams["confidences"]:
                if self.hyperparams["calibrate"]:
                    cccv = CalibratedClassifierCV(self._model, cv="prefit")
                    cccv.fit(inputs, result)
                    confidences = cccv.predict_proba(inputs)[:, 1]
                else:
                    confidences = self._model.decision_function(inputs)
                if self.hyperparams["rank_confidences"]:
                    confidences = rankdata(confidences)
                result_df = container.DataFrame(
                    {self._target_cols[0]: result, "confidence": confidences},
                    generate_metadata=True,
                )
            else:
                result_df = container.DataFrame(
                    {self._target_cols[0]: result},
                    generate_metadata=True,
                )
        else:
            if self.hyperparams["confidences"]:
                if self.hyperparams["calibrate"]:
                    cccv = CalibratedClassifierCV(self._model, cv="prefit")
                    cccv.fit(inputs, result)
                    confidences = cccv.predict_proba(inputs)
                else:
                    confidences = self._get_confidence(inputs)
                result_df = container.DataFrame(
                    {
                        "temp_index": np.repeat(index, confidences.shape[1]),
                        self._target_cols[0]: np.tile(
                            self._model.classes_, index.shape[0]
                        ),
                        "confidence": np.concatenate(confidences),
                    },
                    generate_metadata=True,
                )
                result_df.set_index("temp_index", inplace=True)
            else:
                result_df = container.DataFrame(
                    {
                        self._target_cols[0]: result,
                    },
                    generate_metadata=True,
                )

        # mark the semantic types on the dataframe
        # result_df.metadata = result_df.metadata.add_semantic_type(
        #     (metadata_base.ALL_ELEMENTS, 0),
        #     "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        # )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "http://schema.org/Integer",
        )

        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/Score",
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "http://schema.org/Float",
        )
        return base.CallResult(result_df)

    def _get_confidence(self, X):
        decisions = self._model.decision_function(X)
        exp_decisions = np.exp(decisions - np.max(decisions, axis=1).reshape(-1, 1))
        exp_sum = np.sum(exp_decisions, axis=1)
        return exp_decisions / exp_sum.reshape((-1, 1))

    def get_params(self) -> Params:
        return Params(
            model=self._model,
            needs_fit=self._needs_fit,
            target_cols=self._target_cols,
            binary=self._binary,
            standard_scaler=self._standard_scaler,
        )

    def set_params(self, *, params: Params) -> None:
        self._model = params["model"]
        self._needs_fit = params["needs_fit"]
        self._target_cols = params["target_cols"]
        self._binary = params["binary"]
        self._standard_scaler = params["standard_scaler"]
        return
