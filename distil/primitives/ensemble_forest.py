import logging
import os
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import PrimitiveBase
from distil.modeling.forest import ForestCV
from distil.modeling.metrics import classification_metrics, regression_metrics
from distil.utils import CYTHON_DEP
import version

__all__ = ("EnsembleForest",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Enumeration[str](
        values=classification_metrics + regression_metrics,
        default="f1Macro",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The D3M scoring metric to use during the fit phase.  This can be any of the regression, classification or "
        + "clustering metrics.",
    )
    shap_max_dataset_size = hyperparams.Hyperparameter[int](
        default=1500,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The maximum dataset size on which to apply SHAP interpretation to each sample individually. Otherwise, this number of samples will be"
        + "drawn from the data distribution after clustering (to approximate the distribution) and interpretation will only be applied to these"
        + "samples",
    )

    n_estimators = hyperparams.UniformInt(
        lower=1,
        upper=2048,
        default=32,
        description="The number of trees in the forest.",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter",
            "https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter",
        ],
    )

    min_samples_leaf = hyperparams.UniformInt(
        lower=1,
        upper=31,
        default=2,
        description="Minimum number of samples to split leaf",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter",
            "https://metadata.datadrivendiscovery.org/types/ResourcesUseParameter",
        ],
    )

    class_weight = hyperparams.Enumeration[str](
        values=["None", "balanced", "balanced_subsample"],
        default="None",
        description="todo",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
    )

    estimator = hyperparams.Enumeration[str](
        values=["ExtraTrees", "RandomForest"],
        default="ExtraTrees",
        description="todo",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
    )

    grid_search = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Runs an internal grid search to fit the primitive, ignoring caller supplied values for "
        + "n_estimators, min_samples_leaf, class_weight, estimator",
    )

    small_dataset_threshold = hyperparams.Hyperparameter[int](
        default=2000,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="If grid_search  is true, controls the application of the 'small_dataset_fits' and 'large_dataset_fits' "
        + "parameters - if the input dataset has fewer rows than the threshold value, 'small_dateset_fits' will be used when fitting.  "
        + "Otherwise, 'num_large_fits' is used.",
    )
    small_dataset_fits = hyperparams.Hyperparameter[int](
        default=5,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="If grid_search  is true, the number of random forests to fit when using small datasets.",
    )
    large_dataset_fits = hyperparams.Hyperparameter[int](
        default=1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="If grid_search  is true, the number of random forests to fit when using large datasets.",
    )
    compute_confidences = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Compute confidence values.  Only valid when the task is classification.",
    )
    n_jobs = hyperparams.Hyperparameter[int](
        default=64,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The value of the n_jobs parameter for the joblib library",
    )


class Params(params.Params):
    model: ForestCV
    target_cols: List[str]
    label_map: Dict[int, str]
    needs_fit: bool
    binary: bool
    input_hash: pd.Series


class EnsembleForestPrimitive(
    PrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]
):
    """
    Generates an ensemble of random forests, with the number of internal models created controlled by the size of the
    input dataframe.  It accepts a dataframe as input, and returns a dataframe consisting of prediction values only as output.
    Columns with string structural types are ignored.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "e0ad06ce-b484-46b0-a478-c567e1ea7e02",
            "version": version.__version__,
            "name": "EnsembleForest",
            "python_path": "d3m.primitives.learner.random_forest.DistilEnsembleForest",
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/ensemble_forest.py",
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
                metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.LEARNER,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        # hack to get around typing constraints.
        if self.hyperparams["class_weight"] == "None":
            class_weight = None
        else:
            class_weight = self.hyperparams["class_weight"]

        grid_search = self.hyperparams["grid_search"]
        if grid_search is True:
            current_hyperparams = None
        else:
            current_hyperparams = {
                "estimator": self.hyperparams["estimator"],
                "n_estimators": self.hyperparams[
                    "n_estimators"
                ],  # [32, 64, 128, 256, 512, 1024, 2048],
                "min_samples_leaf": self.hyperparams[
                    "min_samples_leaf"
                ],  # '[1, 2, 4, 8, 16, 32],
            }
            if self.hyperparams["metric"] in classification_metrics:
                current_hyperparams.update({"class_weight": class_weight})
            else:  # regression
                current_hyperparams.update({"bootstrap": True})

        self._model = ForestCV(
            self.hyperparams["metric"],
            random_seed=self.random_seed,
            hyperparams=current_hyperparams,
            grid_search=grid_search,
            n_jobs=self.hyperparams["n_jobs"],
        )
        self._needs_fit = True
        self._label_map: Dict[int, str] = {}
        self._target_cols: List[str] = []
        self._binary = False

    def _get_component_columns(
        self, output_df: container.DataFrame, source_col_index: int
    ) -> List[int]:
        # Component columns are all column which have as source the referenced
        # column index. This includes the aforementioned column index.
        component_cols = [source_col_index]

        # get the column name
        col_name = output_df.metadata.query(
            (metadata_base.ALL_ELEMENTS, source_col_index)
        )["name"]

        # get all columns which have this column as source
        for c in range(0, len(output_df.columns)):
            src = output_df.metadata.query((metadata_base.ALL_ELEMENTS, c))
            if "source_column" in src and src["source_column"] == col_name:
                component_cols.append(c)

        return component_cols

    def set_training_data(
        self, *, inputs: container.DataFrame, outputs: container.DataFrame
    ) -> None:
        # At this point anything that needed to be imputed should have been, so we'll
        # clear out any remaining NaN values as a last measure.

        # if we are doing classification the outputs need to be integer classes.
        # label map is used to covert these back on produce.
        col = outputs.columns[0]
        if self._model.mode == "classification":
            factor = pd.factorize(outputs[col])
            outputs = pd.DataFrame(factor[0], columns=[col])
            self._label_map = {k: v for k, v in enumerate(factor[1])}

        self._target_cols = list(outputs.columns)

        # remove nans from outputs, apply changes to inputs as well to ensure alignment
        self._input_hash = pd.util.hash_pandas_object(inputs)
        self._outputs = outputs[
            outputs[col] != ""
        ].dropna()  # not in place because we don't want to modify passed input
        self._binary = self._outputs.iloc[:, 0].nunique(dropna=True) <= 2
        row_diff = outputs.shape[0] - self._outputs.shape[0]
        if row_diff != 0:
            logger.warn(f"Removed {row_diff} rows due to NaN values in target data.")
            self._inputs = inputs.loc[self._outputs.index, :]
        else:
            self._inputs = inputs

        # same in other direction
        inputs_rows = self._inputs.shape[0]
        inputs_cols = self._inputs.shape[1]
        self._inputs = self._inputs.select_dtypes(include="number")
        col_diff = inputs_cols - self._inputs.shape[1]
        if col_diff != 0:
            logger.warn(f"Removed {col_diff} unencoded columns from training data.")

        self._inputs = (
            self._inputs.dropna()
        )  # not in place because because selection above doesn't create a copy
        row_diff = inputs_rows - self._inputs.shape[0]
        if row_diff != 0:
            logger.warn(f"Removed {row_diff} rows due to NaN values in training data.")
            self._outputs = self._outputs.loc[self._inputs.index, :]

        self._model.num_fits = (
            self.hyperparams["large_dataset_fits"]
            if self._inputs.shape[0] > self.hyperparams["small_dataset_threshold"]
            else self.hyperparams["small_dataset_fits"]
        )

        self._needs_fit = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        logger.debug(f"Fitting {__name__}")

        if self._needs_fit:
            self._model.fit(self._inputs.values, self._outputs.values)
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

        # drop any non-numeric columns
        # drop all non-numeric columns
        num_cols = inputs.shape[1]
        inputs = inputs.select_dtypes(include="number")
        col_diff = num_cols - inputs.shape[1]
        if col_diff > 0:
            logger.warn(f"Removed {col_diff} unencoded columns from produce data.")

        # create dataframe to hold the result
        result = self._model.predict(inputs.values)
        if len(self._target_cols) > 1:
            result_df = container.DataFrame()
            for i, c in enumerate(self._target_cols):
                col = container.DataFrame({c: result[:, i]})
                result_df = pd.concat([result_df, col], axis=1)
            for c in range(result_df.shape[1]):
                result_df.metadata = result_df.metadata.add_semantic_type(
                    (metadata_base.ALL_ELEMENTS, c), "http://schema.org/Float"
                )
        else:
            result_df = container.DataFrame(
                {self._target_cols[0]: result}, generate_metadata=True
            )
        # if we mapped values earlier map them back.
        if len(self._label_map) > 0:
            # TODO label map will not work if there are multiple output columns.
            result_df[self._target_cols[0]] = result_df[self._target_cols[0]].map(
                self._label_map
            )
        # mark the semantic types on the dataframe
        for i, _ in enumerate(result_df.columns):
            result_df.metadata = result_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, i),
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            )
        if (
            self._model.mode == "classification"
            and self.hyperparams["compute_confidences"]
        ):
            confidence = self._model.predict_proba(inputs.values)
            if self._binary:
                # result_df = pd.concat([result_df, confidence[:, 1]], axis=1)
                result_df.insert(result_df.shape[1], "confidence", confidence[:, 1])
            else:
                # add confidence scores as some metrics require them.
                confidence = pd.Series(confidence.tolist(), name="confidence")
                result_df = pd.concat([result_df, confidence], axis=1)

                confidences = [
                    item
                    for sublist in result_df["confidence"].values.tolist()
                    for item in sublist
                ]
                labels = np.array(list(self._label_map.values()) * len(result_df))

                index = [
                    item
                    for sublist in [
                        [i] * len(np.unique(labels)) for i in result_df.index
                    ]
                    for item in sublist
                ]
                result_df_temp = container.DataFrame()
                result_df_temp["Class"] = labels
                result_df_temp["confidence"] = confidences
                result_df_temp.metadata = result_df.metadata
                result_df_temp["index_temp"] = index
                result_df_temp = result_df_temp.set_index("index_temp")
                result_df = result_df_temp
                result_df.metadata = result_df.metadata.add_semantic_type(
                    (metadata_base.ALL_ELEMENTS, len(result_df.columns) - 1),
                    "https://metadata.datadrivendiscovery.org/types/FloatVector",
                )

            result_df.metadata = result_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, len(result_df.columns) - 1),
                "https://metadata.datadrivendiscovery.org/types/Score",
            )
            result_df.metadata = result_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, len(result_df.columns) - 1),
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            )

        logger.debug(f"\n{result_df}")
        return base.CallResult(result_df)

    def produce_feature_importances(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> CallResult[container.DataFrame]:
        logger.debug(f"Producing {__name__} feature weights")

        # force a fit it hasn't yet been done
        if self._needs_fit:
            self.fit()

        # extract the feature weights
        output = container.DataFrame(
            self._model.feature_importances().reshape((1, len(inputs.columns))),
            generate_metadata=True,
        )
        output.columns = inputs.columns
        for i in range(len(inputs.columns)):
            output.metadata = output.metadata.update_column(
                i, {"name": output.columns[i]}
            )

        # map component columns back to their source - this would cover things like
        # a one hot encoding column, that is derived from some original source column
        source_col_importances: Dict[str, float] = {}
        for col_idx in range(0, len(output.columns)):
            col_dict = dict(
                inputs.metadata.query((metadata_base.ALL_ELEMENTS, col_idx))
            )
            # if a column points back to a source column, add that columns importance to the
            # total for that source column
            if "source_column" in col_dict:
                source_col = col_dict["source_column"]
                if source_col not in source_col_importances:
                    source_col_importances[source_col] = 0.0
                source_col_importances[source_col] += output.iloc[:, col_idx]

        for source_col, importance in source_col_importances.items():
            # add the source columns and their importances to the returned data
            output_col_length = len(output.columns)
            output.insert(output_col_length, source_col, importance, True)
            output.metadata = output.metadata.update_column(
                output_col_length, {"name": source_col}
            )

        return CallResult(output)

    def produce_shap_values(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> CallResult[container.DataFrame]:

        if self._needs_fit:
            self.fit()

        # don't want to produce SHAP predictions on train set because too computationally intensive
        check_rows = min(self._input_hash.shape[0], inputs.shape[0])
        if (
            pd.util.hash_pandas_object(inputs.head(check_rows))
            == self._input_hash.head(check_rows)
        ).all():
            logger.info(
                "Not producing SHAP interpretations on train set because of computational considerations"
            )
            return CallResult(container.DataFrame([]))

        # drop any non-numeric columns
        num_cols = inputs.shape[1]
        inputs = inputs.select_dtypes(include="number")
        col_diff = num_cols - inputs.shape[1]
        if col_diff > 0:
            logger.warn(f"Removed {col_diff} unencoded columns.")

        explainer = shap.TreeExplainer(self._model._models[0].model)
        max_size = self.hyperparams["shap_max_dataset_size"]
        if inputs.shape[0] > max_size:
            logger.warning(
                f"There are more than {max_size} rows in dataset, sub-sampling ~{max_size} approximately representative rows "
                + "on which to produce interpretations"
            )
            df = self._shap_sub_sample(inputs)
            shap_values = explainer.shap_values(df)
        else:
            shap_values = explainer.shap_values(pd.DataFrame(inputs))

        if self._model.mode == "classification":
            logger.info(
                f"Returning interpretability values offset from most frequent class in dataset"
            )
            shap_values = shap_values[np.argmax(explainer.expected_value)]

        output_df = container.DataFrame(shap_values, generate_metadata=True)
        for i, col in enumerate(inputs.columns):
            output_df.metadata = output_df.metadata.update_column(i, {"name": col})

        component_cols: Dict[str, List[int]] = {}
        for c in range(0, len(output_df.columns)):
            col_dict = dict(inputs.metadata.query((metadata_base.ALL_ELEMENTS, c)))
            if "source_column" in col_dict:
                src = col_dict["source_column"]
                if src not in component_cols:
                    component_cols[src] = []
                component_cols[src].append(c)

        # build the source column values and add them to the output
        for s, cc in component_cols.items():
            src_col = output_df.iloc[:, cc].apply(lambda x: sum(x), axis=1)
            src_col_index = len(output_df.columns)
            output_df.insert(src_col_index, s, src_col)
            output_df.metadata = output_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, src_col_index),
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )

        df_dict = dict(output_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict_1 = dict(output_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict["dimension"] = df_dict_1
        df_dict_1["name"] = "columns"
        df_dict_1["semantic_types"] = (
            "https://metadata.datadrivendiscovery.org/types/TabularColumn",
        )
        df_dict_1["length"] = len(output_df.columns)
        output_df.metadata = output_df.metadata.update(
            (metadata_base.ALL_ELEMENTS,), df_dict
        )

        return CallResult(output_df)

    def _shap_sub_sample(self, inputs: container.DataFrame):

        df = pd.DataFrame(inputs)
        df["cluster_assignment"] = (
            KMeans(random_state=self.random_seed).fit_predict(df).astype(int)
        )
        n_classes = df["cluster_assignment"].unique()

        # deal with cases in which the predictions are all one class
        if len(n_classes) == 1:
            return df.sample(self.hyperparams["shap_max_dataset_size"]).drop(
                columns=["cluster_assignment"]
            )

        else:
            proportion = round(
                self.hyperparams["shap_max_dataset_size"] / len(n_classes)
            )
            dfs = []
            for i in n_classes:
                # dealing with classes that have less than or equal to their proportional representation
                if df[df["cluster_assignment"] == i].shape[0] <= proportion:
                    dfs.append(df[df["cluster_assignment"] == i])
                else:
                    dfs.append(
                        df[df["cluster_assignment"] == i].sample(
                            proportion, random_state=self.random_seed
                        )
                    )

            sub_sample_df = pd.concat(dfs)
            return sub_sample_df.drop(columns=["cluster_assignment"])

    def get_params(self) -> Params:
        return Params(
            model=self._model,
            target_cols=self._target_cols,
            label_map=self._label_map,
            needs_fit=self._needs_fit,
            input_hash=self._input_hash,
            binary=self._binary,
        )

    def set_params(self, *, params: Params) -> None:
        self._model = params["model"]
        self._target_cols = params["target_cols"]
        self._label_map = params["label_map"]
        self._needs_fit = params["needs_fit"]
        self._input_hash = params["input_hash"]
        self._binary = params["binary"]
        return
