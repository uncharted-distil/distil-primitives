import logging
import os
from typing import Optional, List, Any
import pandas as pd

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base
from distil.preprocessing.transformers import SVMTextEncoder, TfidifEncoder
from distil.primitives import utils as distil_utils
from distil.modeling.metrics import (
    classification_metrics,
    clustering_metrics,
    regression_metrics,
)
import version

__all__ = ("TextEncoderPrimitive",)

logger = logging.getLogger(__name__)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    metric = hyperparams.Enumeration[str](
        values=classification_metrics + regression_metrics + clustering_metrics,
        default="f1Macro",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The D3M scoring metric to use during the fit phase.  This can be any of the regression, classification or "
        + "clustering metrics.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )

    encoder_type = hyperparams.Enumeration(
        default="svm",
        values=["svm", "tfidf"],
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Vectorization Strategy.",
    )


class Params(params.Params):
    _cols: Optional[List[int]]
    _encoders: Any
    # _encoders: Optional[List[Union[SVMTextEncoder, TfidifEncoder]]]


class TextEncoderPrimitive(base.PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Encodes string fields using TFIDF scoring combined with a linear SVC classifier.  The original string field is removed
    and replaced with encoding columns.
    """

    _attribute_semantic = "https://metadata.datadrivendiscovery.org/types/Attribute"
    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "09f252eb-215d-4e0b-9a60-fcd967f5e708",
            "version": version.__version__,
            "name": "Text encoder",
            "python_path": "d3m.primitives.data_transformation.encoder.DistilTextEncoder",
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/text_encoder.py",
                    "https://github.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.ENCODE_BINARY,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self._encoders: List[SVMTextEncoder] = []
        self._cols: List[int] = []

    def __getstate__(self) -> dict:
        state = base.PrimitiveBase.__getstate__(self)
        state["models"] = self._encoders
        state["columns"] = self._cols
        return state

    def __setstate__(self, state: dict) -> None:
        base.PrimitiveBase.__setstate__(self, state)
        self._encoders = state["models"]
        self._cols = state["columns"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        self._inputs = inputs

        # https://github.com/scikit-learn/scikit-learn/issues/14429#issuecomment-513887163
        if type(outputs) == container.pandas.DataFrame and outputs.shape[1] == 1:
            outputs = outputs.values.reshape(
                outputs.shape[0],
            )
        else:
            outputs = outputs.iloc[:, 0].values
        self._outputs = pd.Series(outputs)

    def fit(
        self, *, timeout: float = None, iterations: int = None
    ) -> base.CallResult[None]:
        logger.debug(f"Fitting {__name__}")

        # determine columns to operate on
        cols = distil_utils.get_operating_columns(
            self._inputs, self.hyperparams["use_columns"], ("http://schema.org/Text",)
        )

        logger.debug(f"Found {len(cols)} columns to encode")

        self._cols = list(cols)
        self._encoders: List[SVMTextEncoder] = []
        if len(cols) is 0:
            return base.CallResult(None)

        for i, c in enumerate(self._cols):
            if self.hyperparams["encoder_type"] == "svm":
                self._encoders.append(
                    SVMTextEncoder(self.hyperparams["metric"], self.random_seed)
                )
            elif self.hyperparams["encoder_type"] == "tfidf":
                self._encoders.append(TfidifEncoder())
            else:
                raise Exception(
                    f"{self.hyperparams['encoder_type']} is not a valid encoder type"
                )
            text_inputs = self._inputs.iloc[:, c]
            try:
                self._encoders[i].fit_transform(
                    text_inputs, self._outputs
                )  # requires fit transform to fit SVM on vectorizer results
            except:
                text_inputs[:] = "avoiding a bug"
                self._encoders[i].fit_transform(text_inputs, self._outputs)

        return base.CallResult(None)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> base.CallResult[Outputs]:
        logger.debug(f"Producing {__name__}")

        if len(self._cols) == 0:
            return base.CallResult(inputs)

        outputs = inputs.copy()
        encoded_cols = container.DataFrame()
        encoded_cols_source = []
        # encode columns into a new dataframe
        for i, c in enumerate(self._cols):
            text_inputs = outputs.iloc[:, c]
            result = self._encoders[i].transform(text_inputs)
            for j in range(result.shape[1]):
                encoded_idx = i * result.shape[1] + j
                encoded_cols[(f"__text_{encoded_idx}")] = result[:, j]
                encoded_cols_source.append(c)
        # generate metadata for encoded columns
        encoded_cols.metadata = encoded_cols.metadata.generate(encoded_cols)
        for c in range(encoded_cols.shape[1]):
            encoded_cols.metadata = encoded_cols.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, c), "http://schema.org/Float"
            )
            encoded_cols.metadata = encoded_cols.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, c), self._attribute_semantic
            )
            col_dict = dict(
                encoded_cols.metadata.query((metadata_base.ALL_ELEMENTS, c))
            )
            col_dict["source_column"] = outputs.metadata.query(
                (metadata_base.ALL_ELEMENTS, encoded_cols_source[c])
            )["name"]
            encoded_cols.metadata = encoded_cols.metadata.update(
                (metadata_base.ALL_ELEMENTS, c), col_dict
            )

        # append the encoded columns and remove the source columns
        outputs = outputs.append_columns(encoded_cols)
        outputs = outputs.remove_columns(self._cols)

        logger.debug(f"\n{outputs}")

        return base.CallResult(outputs)

    def get_params(self) -> Params:
        return Params(_encoders=self._encoders, _cols=self._cols)

    def set_params(self, *, params: Params) -> None:
        self._encoders = params["_encoders"]
        self._cols = params["_cols"]
