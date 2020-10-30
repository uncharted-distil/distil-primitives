import logging
import os

import pandas as pd
from d3m import container, utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.primitives import utils as distil_utils
from distil.utils import CYTHON_DEP
import version

__all__ = ("EnrichDatesPrimitive",)

logger = logging.getLogger(__name__)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    replace = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Whether or not to replace enriched DateTime columns or append them",
    )


class EnrichDatesPrimitive(
    transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]
):
    """
    Enriches dates by converting to seconds from a base time and computing Z scores.  The results
    are appended to the existing dataset, and the original column is left in place for additional
    downstream processing.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "b1367f5b-bab1-4dfc-a1a9-6a56430e516a",
            "version": version.__version__,
            "name": "Enrich dates",
            "python_path": "d3m.primitives.data_transformation.data_cleaning.DistilEnrichDates",
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/distil/primitives/enrich_dates.py",
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
                metadata_base.PrimitiveAlgorithmType.ENCODE_BINARY,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> base.CallResult[Outputs]:
        logger.debug(f"Running {__name__}")

        outputs = inputs.copy()
        outputs = self._enrich_dates(outputs)

        logger.debug(f"\n{outputs}")

        return base.CallResult(outputs)

    def _enrich_dates(self, inputs: Inputs) -> Outputs:

        # determine columns we need to operate on
        cols = distil_utils.get_operating_columns(
            inputs, self.hyperparams["use_columns"], ("http://schema.org/DateTime",)
        )

        date_num = 0
        for c in cols:
            try:
                # compute z scores for column members
                inputs_seconds = (
                    (pd.to_datetime(inputs.iloc[:, c]) - pd.to_datetime("2000-01-01"))
                    .dt.total_seconds()
                    .values
                )
                sec_mean = inputs_seconds.mean()
                sec_std = inputs_seconds.std()
                sec_val = 0.0
                if sec_std != 0.0:
                    sec_val = (inputs_seconds - sec_mean) / sec_std

                if self.hyperparams["replace"]:
                    inputs.metadata = inputs.metadata.add_semantic_type(
                        (metadata_base.ALL_ELEMENTS, c), "http://schema.org/Float"
                    )
                    inputs.metadata = inputs.metadata.remove_semantic_type(
                        (metadata_base.ALL_ELEMENTS, c), "http://schema.org/DateTime"
                    )
                    inputs.metadata = inputs.metadata.update(
                        (metadata_base.ALL_ELEMENTS, c), {"structural_type": float}
                    )
                    inputs[inputs.columns[c]] = sec_val
                else:
                    # append the results and update semantic types
                    result = container.DataFrame(
                        {f"__date_{date_num}": sec_val}, generate_metadata=True
                    )
                    result.metadata = result.metadata.add_semantic_type(
                        (metadata_base.ALL_ELEMENTS, 0), "http://schema.org/Float"
                    )
                    inputs = inputs.append_columns(result)

                date_num += 1
            except:
                continue

        return inputs
