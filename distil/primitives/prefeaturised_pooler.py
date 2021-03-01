import logging
import os
import math
from typing import List, Optional
import typing

import numpy as np
import pandas as pd

from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces import base, transformer

from distil.utils import CYTHON_DEP
import version

__all__ = ("PrefeaturisedPoolingPrimitive",)

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=256,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="inference batch size",
    )
    height = hyperparams.Hyperparameter[typing.Optional[int]](
        default=4,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Height of pooled images",
    )
    width = hyperparams.Hyperparameter[typing.Optional[int]](
        default=4,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Width of pooled images",
    )


class PrefeaturisedPoolingPrimitive(
    transformer.TransformerPrimitiveBase[
        container.DataFrame, container.DataFrame, Hyperparams
    ]
):
    """
    Made specifically to take unpooled outputs from RemoteSensingPretrainedPrimitive
    and pool them.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "825ea1fb-90b2-442c-9905-efba48872102",
            "version": version.__version__,
            "name": "Prefeaturised Pooler",
            "python_path": "d3m.primitives.remote_sensing.remote_sensing_pretrained.PrefeaturisedPooler",
            "source": {
                "name": "Distil",
                "contact": "mailto:vkorapaty@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/prefeaturised_pooler.py",
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
                metadata_base.PrimitiveAlgorithmType.MOMENTUM_CONTRAST,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.REMOTE_SENSING,
        },
    )

    def produce(
        self,
        *,
        inputs: container.DataFrame,
        timeout: float = None,
        iterations: int = None,
    ) -> base.CallResult[container.DataFrame]:

        df = inputs.select_columns(
            inputs.metadata.list_columns_with_semantic_types(
                ("http://schema.org/Float",)
            )
        )
        df = df.to_numpy().reshape(
            df.shape[0], 2048, self.hyperparams["height"], self.hyperparams["width"]
        )
        all_img_features = []
        batch_size = self.hyperparams["batch_size"]
        spatial_a = 2.0
        spatial_b = 2.0
        for i in range(math.ceil(df.shape[0] / batch_size)):
            features = df[i * batch_size : (i + 1) * batch_size]
            spatial_weight = features.sum(axis=1, keepdims=True)
            z = (spatial_weight ** spatial_a).sum(axis=(2, 3), keepdims=True)
            z = z ** (1.0 / spatial_a)
            spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

            _, c, w, h = features.shape
            nonzeros = (features != 0).astype(float).sum(axis=(2, 3)) / 1.0 / (
                w * h
            ) + 1e-6
            channel_weight = np.log(nonzeros.sum(axis=1, keepdims=True) / nonzeros)

            features = features * spatial_weight
            features = features.sum(axis=(2, 3))
            features = features * channel_weight
            all_img_features.append(features)
        all_img_features = np.vstack(all_img_features)
        col_names = [f"feat_{i}" for i in range(0, all_img_features.shape[1])]
        feature_df = pd.DataFrame(all_img_features, columns=col_names)
        feature_df = container.DataFrame(feature_df, generate_metadata=True)

        for idx in range(feature_df.shape[1]):
            feature_df.metadata = feature_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, idx), "http://schema.org/Float"
            )

        # if inputs.shape[1] > 1:
        #     input_df = inputs.remove_columns(image_cols)
        #     feature_df = input_df.append_columns(feature_df)
        return base.CallResult(feature_df)
