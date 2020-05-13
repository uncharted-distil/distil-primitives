import os
import logging
import typing

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from d3m.contrib.primitives import compute_scores

import common_primitives
from common_primitives import construct_predictions

__all__ = ('PredictionExpansionPrimitive',)
logger = logging.getLogger(__name__)

Inputs = container.DataFrame
Outputs = container.DataFrame

class PredictionExpansionPrimitive(construct_predictions.ConstructPredictionsPrimitive):
    """
    A primitive which takes as input a DataFrame and outputs a DataFrame in Lincoln Labs predictions
    format: first column is a d3mIndex column (and other primary index columns, e.g., for object detection
    problem), and then predicted targets, each in its column, followed by optional confidence column(s).

    If the reference dataset has a multi index, the predictions will be expanded to match the reference
    dataset shape.

    It supports both input columns annotated with semantic types (``https://metadata.datadrivendiscovery.org/types/PrimaryKey``,
    ``https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey``, ``https://metadata.datadrivendiscovery.org/types/PredictedTarget``,
    ``https://metadata.datadrivendiscovery.org/types/Confidence``), or trying to reconstruct metadata.
    This is why the primitive takes also additional input of a reference DataFrame which should
    have metadata to help reconstruct missing metadata. If metadata is missing, the primitive
    assumes that all ``inputs`` columns are predicted targets, without confidence column(s).
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '36bedff2-1c99-47f3-92a8-a581f35b1924',
            'version': '0.1.0',
            'name': "Expand group predictions",
            'python_path': 'd3m.primitives.data_transformation.prediction_expansion.DistilPredictionExpansion',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/prediction_expansion.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
               'type': metadata_base.PrimitiveInstallationType.PIP,
               'package_uri': 'git+https://gitlab.com/datadrivendiscovery/common-primitives.git@{git_commit}#egg=common_primitives'.format(
                   git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
               ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.DATA_CONVERSION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(self, *, inputs: Inputs, reference: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:  # type: ignore
        index_columns = inputs.metadata.get_index_columns()
        target_columns = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/PredictedTarget',))

        # Target columns cannot be also index columns. This should not really happen,
        # but it could happen with buggy primitives.
        target_columns = [target_column for target_column in target_columns if target_column not in index_columns]

        # only expand data if there are fewer prediction rows than reference rows
        if index_columns and target_columns and len(inputs.index) < len(reference.index):
            outputs = self._expand_predictions(inputs, reference, index_columns, target_columns)
        else:
            return super().produce(inputs=inputs, reference=reference, timeout=timeout, iterations=iterations)

        outputs = compute_scores.ComputeScoresPrimitive._encode_columns(outputs)

        # Generally we do not care about column names in DataFrame itself (but use names of columns from metadata),
        # but in this case setting column names makes it easier to assure that "to_csv" call produces correct output.
        # See: https://gitlab.com/datadrivendiscovery/d3m/issues/147
        column_names = []
        for column_index in range(len(outputs.columns)):
            column_names.append(outputs.metadata.query_column(column_index).get('name', outputs.columns[column_index]))
        outputs.columns = column_names

        return base.CallResult(outputs)

    def _expand_predictions(self, inputs: Inputs, reference: Inputs, index_columns: typing.Sequence[int], target_columns: typing.Sequence[int]) -> Outputs:
        # join the inputs to the reference dataset using the index columns
        if len(index_columns) != 1:
            return inputs

        reference_index = reference.metadata.get_index_columns()
        if len(reference_index) != 1:
            return inputs

        # only the index column is needed from the reference dataset since the
        # rest of the data will come from the inputs
        reference_base = reference.select_columns(reference_index)
        output = reference_base.join(inputs.set_index(inputs.columns[0]), on=reference_base.columns[0], rsuffix='predicted')
        output.metadata = output.metadata.append_columns(inputs.metadata.remove_columns(index_columns))

        return output
