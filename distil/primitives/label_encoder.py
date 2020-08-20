from typing import Any, Callable, List, Dict, Union, Optional, Sequence, Tuple
from numpy import ndarray
from collections import OrderedDict, defaultdict
from scipy import sparse
import os
import sklearn
import numpy
import typing

# Custom import commands if any
from sklearn.preprocessing.label import LabelEncoder

from d3m.container.numpy import ndarray as d3m_ndarray
from d3m.container import DataFrame as d3m_dataframe
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m import utils, container
from d3m.primitive_interfaces.base import CallResult
import common_primitives.utils as common_utils
from d3m.primitive_interfaces.unsupervised_learning import UnsupervisedLearnerPrimitiveBase
from distil.utils import CYTHON_DEP

class Params(params.Params):
    classes_: Optional[Dict]
    training_indices_: Optional[Sequence[int]]


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should parsed columns be appended, should they replace original columns, or should only parsed columns be returned?",
    )
    use_semantic_types = hyperparams.UniformBool(
        default=False,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Controls whether semantic_types metadata will be used for filtering columns in input dataframe."
    )


class SKLabelEncoder(UnsupervisedLearnerPrimitiveBase[container.DataFrame, container.DataFrame, Params, Hyperparams]):
    """
    Primitive wrapping for sklearn LabelEncoder
    """

    __author__ = "Venkat Korapaty"
    metadata = metadata_base.PrimitiveMetadata({
        "id": "",
        "version": "1.0.0",
        "name": "Label Encoder",
        "python_path": "d3m.primitives.data_transformation.encoder.SKLabelEncoder",
        "source": {
            'name': 'Distil',
            'contact': 'mailto:vkorapaty@uncharted.software',
            'uris': [
                'https://github.com/uncharted-distil/distil-primitives/distil/primitives/label_encoder.py',
                'https://github.com/uncharted-distil/distil-primitives',
            ],
        },
        'installation': [CYTHON_DEP, {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
        }],
        "algorithm_types": [metadata_base.PrimitiveAlgorithmType.CATEGORY_ENCODER, ],
        "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
    })

    def __init__(self, *,
                 hyperparams: Hyperparams,
                 random_seed: int = 0) -> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._clf = defaultdict(LabelEncoder)
        self._training_inputs = None
        self._target_names = None
        self._training_indices = None
        self._fitted = False

    def set_training_data(self, *, inputs: container.DataFrame) -> None:
        self._training_inputs, self._training_indices = self._get_columns_to_fit(inputs, self.hyperparams)
        self._fitted = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._fitted:
            return CallResult(None)

        if self._training_inputs is None:
            raise ValueError("Missing training data.")

        # self._clf.fit(self._training_inputs)
        output = self._training_inputs.apply(lambda x: self._clf[x.name].fit(x))
        self._fitted = True

        return CallResult(None)

    def produce(self, *, inputs: container.DataFrame, timeout: float = None, iterations: int = None) -> CallResult[container.DataFrame]:
        sk_inputs = inputs
        if self.hyperparams['use_semantic_types']:
            sk_inputs = inputs.iloc[:, self._training_indices]
        sk_output = sk_inputs.apply(lambda x: self._clf[x.name].transform(x))
        if sparse.issparse(sk_output):
            sk_output = sk_output.toarray()
        output = d3m_dataframe(sk_output, generate_metadata=False, source=self)
        output.metadata = inputs.metadata.clear(source=self, for_value=output, generate_metadata=True)

        return CallResult(output)

    def get_params(self) -> Params:
        if not self._fitted:
            raise ValueError("Fit not performed.")
        classes = defaultdict()
        for key, value in self._clf.items():
            classes[key] = getattr(value, 'classes_', None)
        return Params(
            classes_= classes,
            training_indices_=self._training_indices
        )

    def set_params(self, *, params: Params) -> None:
        for key, value in params['classes_'].items():
            setattr(self._clf[key], 'classes_', value)
        self._training_indices = params['training_indices_']
        self._fitted = True

    @classmethod
    def _get_columns_to_fit(cls, inputs: container.DataFrame, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return inputs, [len(inputs.columns)]

        inputs_metadata = inputs.metadata

        def can_produce_column(column_index: int) -> bool:
            return cls._can_produce_column(inputs_metadata, column_index, hyperparams)

        columns_to_produce, columns_not_to_produce = common_utils.get_columns_to_use(inputs_metadata,
                                                                                     use_columns=hyperparams[
                                                                                         'use_columns'],
                                                                                     exclude_columns=hyperparams[
                                                                                         'exclude_columns'],
                                                                                     can_use_column=can_produce_column)
        return inputs.iloc[:, columns_to_produce], columns_to_produce
        # return columns_to_produce

    @classmethod
    def _can_produce_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: int,
                            hyperparams: Hyperparams) -> bool:
        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        accetped_structural_types = [int, float, numpy.int64, numpy.float64]
        if column_metadata['structural_type'] not in accetped_structural_types:
            return False

        semantic_types = column_metadata.get('semantic_types', [])
        if len(semantic_types) == 0:
            return True
        if "https://metadata.datadrivendiscovery.org/types/Attribute" in semantic_types:
            return True

        return False

    @classmethod
    def _get_targets(cls, data: container.DataFrame, hyperparams: Hyperparams):
        if not hyperparams['use_semantic_types']:
            return data, []
        target_names = []
        target_column_indices = []
        metadata = data.metadata
        target_column_indices.extend(
            metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget'))
        target_column_indices.extend(
            metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/RedactedTarget'))
        target_column_indices.extend(
            metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget'))
        target_column_indices = list(set(target_column_indices))
        for column_index in target_column_indices:
            if column_index is metadata_base.ALL_ELEMENTS:
                continue
            column_index = typing.cast(metadata_base.SimpleSelectorSegment, column_index)
            column_metadata = metadata.query((metadata_base.ALL_ELEMENTS, column_index))
            target_names.append(column_metadata.get('name', str(column_index)))

        targets = data.iloc[:, target_column_indices]
        return targets, target_names

    @classmethod
    def _add_target_semantic_types(cls, metadata: metadata_base.DataMetadata,
                                   source: typing.Any) -> metadata_base.DataMetadata:
        for column_index in range(metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/Target',
                                                  source=source)
            metadata = metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, column_index),
                                                  'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                                  source=source)

        return metadata


SKLabelEncoder.__doc__ = LabelEncoder.__doc__
