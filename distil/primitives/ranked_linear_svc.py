import logging
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from pandas.core.computation.pytables import ConditionBinOp
import sklearn
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
    pos_label = hyperparams.Hyperparameter[Optional[str]](
        default=None,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Name of the positive label in the binary case. If none is provided, second column is assumed to be positive",
    )


class Params(params.Params):
    model: LinearSVC
    target_cols: List[str]
    needs_fit: bool
    binary: bool
    standard_scaler: Optional[StandardScaler]
    label_map: Dict[int, str]


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
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/ranked_linear_svc.py",
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
        self._label_map: Dict[int, str] = {}

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
        if self._binary:
            pos_label = self.hyperparams["pos_label"]
            labels = self._outputs.values.ravel()
            unique_labels = np.unique(labels)
            # needed to get decision values or confidences of correct column, in binary case
            if pos_label == unique_labels[0]:
                self._label_map[1] = unique_labels[0]
                self._label_map[0] = unique_labels[1]
                self._outputs[self._outputs.columns[0]] = np.array(
                    [1 if l == pos_label else 0 for l in labels]
                )

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f"Fitting {__name__}")

        if self._needs_fit:
            labels = self._outputs.values.ravel()
            self._model.fit(self._inputs, labels)
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
                # If we're generating confidences, check to see where or not we calibrate
                # to probabilities, or just return the raw decision function values.
                if self.hyperparams["calibrate"]:
                    cccv = CalibratedClassifierCV(self._model, cv="prefit")
                    try:
                        cccv.fit(inputs, result)
                        confidences = cccv.predict_proba(inputs)[:, 1]
                    except:
                        # calibration can fail for a variety of reasons - we'll just fall back on
                        # simpler methods when it does
                        confidences = self._get_confidence(inputs)
                else:
                    confidences = self._model.decision_function(inputs)

                # Generate ranks if required, otherwise we just include the confidence / decision
                # function values.
                if self.hyperparams["rank_confidences"]:
                    ranks = rankdata(self._model.decision_function(inputs))
                    result_df = container.DataFrame(
                        {
                            self._target_cols[0]: result,
                            "confidence": confidences,
                            "rank": ranks,
                        },
                        generate_metadata=True,
                    )
                else:
                    result_df = container.DataFrame(
                        {
                            self._target_cols[0]: result,
                            "confidence": confidences,
                        },
                        generate_metadata=True,
                    )
            else:
                result_df = container.DataFrame(
                    {self._target_cols[0]: result},
                    generate_metadata=True,
                )
            pos_label = self.hyperparams["pos_label"]
            # need to map labels back if we mapped to a different label set
            if len(self._label_map) > 0:
                result_df[self._target_cols[0]] = np.array(
                    [
                        self._label_map[1] if l == 1 else self._label_map[0]
                        for l in result
                    ]
                )
        else:
            if self.hyperparams["confidences"]:
                # If generating confidences, generate calibrated or uncalibrated probabilities.
                if self.hyperparams["calibrate"]:
                    cccv = CalibratedClassifierCV(self._model, cv="prefit")
                    try:
                        cccv.fit(inputs, result)
                        confidences = cccv.predict_proba(inputs)
                    except:
                        confidences = self._get_confidence(inputs)
                else:
                    confidences = self._get_confidence(inputs)

                result_df = container.DataFrame(
                    {
                        self._target_cols[0]: np.tile(
                            self._model.classes_, index.shape[0]
                        ),
                        "confidence": np.concatenate(confidences),
                    },
                    generate_metadata=True,
                )
                temp_index = np.repeat(index, confidences.shape[1])
                result_df.set_index(temp_index, inplace=True)
            else:
                result_df = container.DataFrame(
                    {
                        self._target_cols[0]: result,
                    },
                    generate_metadata=True,
                )

        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )

        if "confidence" in result_df.columns:
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

        if "rank" in result_df.columns:
            result_df.metadata = result_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, 2),
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            )
            result_df.metadata = result_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, 2),
                "https://metadata.datadrivendiscovery.org/types/Rank",
            )
            result_df.metadata = result_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, 2),
                "http://schema.org/Float",
            )

        return base.CallResult(result_df)

    def _get_confidence(self, X):
        decisions = self._model.decision_function(X)
        if self._binary:
            # in the binary case we'll just apply a sigmoid function to get everything into a [0,1]
            # interval
            return 1 / (1 + np.exp(-decisions))

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
            label_map=self._label_map,
        )

    def set_params(self, *, params: Params) -> None:
        self._model = params["model"]
        self._needs_fit = params["needs_fit"]
        self._target_cols = params["target_cols"]
        self._binary = params["binary"]
        self._standard_scaler = params["standard_scaler"]
        self._label_map = params["label_map"]
        return
