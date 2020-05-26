import os
import typing

import numpy as np
import pandas as pd  # type: ignore
from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
from distil.utils import CYTHON_DEP
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

__all__ = ('MIRankingPrimitive',)


class Hyperparams(hyperparams.Hyperparams):
    target_col_index = hyperparams.Hyperparameter[typing.Optional[int]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of target feature to rank against.'
    )

class MIRankingPrimitive(transformer.TransformerPrimitiveBase[container.DataFrame,
                                                              container.DataFrame,
                                                              Hyperparams]):
    """
    Feature ranking based on a mutual information between features and a selected
    target.  Will rank any feature column with a semantic type of Float, Boolean,
    Integer or Categorical, and a corresponding structural type of int, float or str.
    Features that could not be ranked are excluded from the returned set.
    Parameters
    ----------
    inputs : A container.Dataframe with columns containing numeric or string data.
    Returns
    -------
    output : A DataFrame containing (col_idx, col_name, score) tuples for each ranked feature.
    """

    # allowable target column types
    _discrete_types = (
        'http://schema.org/Boolean',
        'http://schema.org/Integer',
        'https://metadata.datadrivendiscovery.org/types/CategoricalData'
    )

    _continous_types = (
        'http://schema.org/Float',
    )

    _roles = (
        'https://metadata.datadrivendiscovery.org/types/Attribute',
        'https://metadata.datadrivendiscovery.org/types/Target',
        'https://metadata.datadrivendiscovery.org/types/TrueTarget',
        'https://metadata.datadrivendiscovery.org/types/SuggestedTarget',
    )

    _structural_types = set((
        int,
        float
    ))

    _semantic_types = set(_discrete_types).union(_continous_types)

    _random_seed = 100

    __author__ = 'Uncharted Software',
    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'a31b0c26-cca8-4d54-95b9-886e23df8886',
            'version': '0.2.1',
            'name': 'Mutual Information Feature Ranking',
            'python_path': 'd3m.primitives.feature_selection.mutual_info_classif.DistilMIRanking',
            'keywords': ['vector', 'columns', 'dataframe'],
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/mi_ranking.py',
                    'https://github.com/uncharted-distil/distil-primitives/',
                ]
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        }
    )

    @classmethod
    def _can_use_column(cls, inputs_metadata: metadata_base.DataMetadata, column_index: typing.Optional[int]) -> bool:

        column_metadata = inputs_metadata.query((metadata_base.ALL_ELEMENTS, column_index))

        valid_struct_type = column_metadata.get('structural_type', None) in cls._structural_types
        semantic_types = column_metadata.get('semantic_types', [])
        valid_semantic_type = len(set(cls._semantic_types).intersection(semantic_types)) > 0
        valid_role_type = len(set(cls._roles).intersection(semantic_types)) > 0

        return valid_struct_type and valid_semantic_type

    @classmethod
    def _append_rank_info(cls,
                          inputs: container.DataFrame,
                          result: typing.List[typing.Tuple[int, str, float]],
                          rank_np: np.array,
                          rank_df: pd.DataFrame) -> typing.List[typing.Tuple[int, str, float]]:
        for i, rank in enumerate(rank_np):
            col_name = rank_df.columns.values[i]
            result.append((inputs.columns.get_loc(col_name), col_name, rank))
        return result

    def produce(self, *,
                inputs: container.DataFrame,
                timeout: float = None,
                iterations: int = None) -> base.CallResult[container.DataFrame]:

        cols = ['idx', 'name', 'rank']

        # Make sure the target column is of a valid type and return no ranked features if it isn't.
        target_idx = self.hyperparams['target_col_index']
        if not self._can_use_column(inputs.metadata, target_idx):
            return base.CallResult(container.DataFrame(data={}, columns=cols))

        # check if target is discrete or continuous
        semantic_types = inputs.metadata.query_column(target_idx)['semantic_types']
        discrete = len(set(semantic_types).intersection(self._discrete_types)) > 0

        # make a copy of the inputs and clean out any missing data
        feature_df = inputs.copy()
        feature_df.dropna(inplace=True)

        # split out the target feature
        target_df = feature_df.iloc[:, target_idx]

        # drop features that are not compatible with ranking
        feature_indices = set(inputs.metadata.list_columns_with_semantic_types(self._semantic_types))
        role_indices = set(inputs.metadata.list_columns_with_semantic_types(self._roles))
        feature_indices = feature_indices.intersection(role_indices)
        feature_indices.remove(target_idx)

        # return an empty result if all features were incompatible
        if len(feature_indices) is 0:
            return base.CallResult(container.DataFrame(data={}, columns=cols))

        all_indices = set(range(0, inputs.shape[1]))
        skipped_indices = all_indices.difference(feature_indices)
        for i, v in enumerate(skipped_indices):
            feature_df.drop(inputs.columns[v], axis=1, inplace=True)

        # figure out the discrete and continuous feature indices and create an array
        # that flags them
        discrete_indices = inputs.metadata.list_columns_with_semantic_types(self._discrete_types)
        discrete_flags = [False] * feature_df.shape[1]
        for v in discrete_indices:
            col_name = inputs.columns[v]
            if col_name in feature_df:
                # only mark columns with a least 1 duplicate value as discrete when predicting
                # a continuous target - there's a check in the bowels of MI code that will throw
                # an exception otherwise
                if feature_df[col_name].duplicated().any() and not discrete:
                    col_idx = feature_df.columns.get_loc(col_name)
                    discrete_flags[col_idx] = True

        target_np = target_df.values
        feature_np = feature_df.values

        # compute mutual information for discrete or continuous target
        ranked_features_np = None
        if discrete:
            ranked_features_np = mutual_info_classif(feature_np,
                                                     target_np,
                                                     discrete_features=discrete_flags,
                                                     random_state=self._random_seed)
        else:
            ranked_features_np = mutual_info_regression(feature_np,
                                                        target_np,
                                                        discrete_features=discrete_flags,
                                                        random_state=self._random_seed)

        # merge back into a single list of col idx / rank value tuples
        data: typing.List[typing.Tuple[int, str, float]] = []
        data = self._append_rank_info(inputs, data, ranked_features_np, feature_df)

        # wrap as a D3M container - metadata should be auto generated
        results = container.DataFrame(data=data, columns=cols, generate_metadata=True)
        results = results.sort_values(by=['rank'], ascending=False).reset_index(drop=True)

        return base.CallResult(results)
