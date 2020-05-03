import os

import frozendict  # type: ignore
import imageio  # type: ignore
import numpy  # type: ignore

from d3m import container, utils as d3m_utils
from d3m.metadata import base as metadata_base

import common_primitives
from common_primitives import dataframe_image_reader
from common_primitives import base


class DataFrameSatelliteImageReaderHyperparams(base.FileReaderHyperparams):
    grouping_key_column = hyperparams.Hyperparameter[int](
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="grouping key identifying a unique set of satellite images connecting all bands"
    )


class DataFrameSatelliteImageReaderPrimitive(dataframe_image_reader.DataFrameImageReaderPrimitive):
    """
    A primitive which reads columns referencing image files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``image/jpeg``, ``image/png``) has every filename read into an image
    represented as a numpy array. By default the resulting column with read arrays is appended
    to existing columns.

    The shape of numpy arrays is H x W x C. C is the number of channels in an image
    (e.g., C = 1 for greyscale, C = 3 for RGB), H is the height, and W is the width.
    dtype is uint8.
    """

    _supported_media_types = (
        'image/tiff',
    )
    _file_structural_type = container.ndarray
    _file_semantic_types = ('http://schema.org/ImageObject',)

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8f2e51e8-da59-456d-ae29-53912b2b9f3d',
            'version': '0.1.0',
            'name': 'Columns satellite image reader',
            'python_path': 'd3m.primitives.data_preprocessing.satellite_image_reader.DistilSatelliteImageReader',
            'keywords': ['satellite', 'image', 'reader', 'tiff'],
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/satellite_image_loader.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [CYTHON_DEP, {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
            ],
            'supported_media_types': _supported_media_types,
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        }
    )

    def _produce_column(self, inputs: base.FileReaderInputs, column_index: int) -> base.FileReaderOutputs:
        loaded_images = super()._produce_column(inputs, column_index)
        loaded_images_clone = loaded_images.copy()

        # need to flatten the dataframe, having one column per band
        # start by creating all the flat rows, using the grouping key to connect the bands together
        grouping_column = self._get_grouping_key_column(loaded_images_clone)
        if grouping_column < 0:
            self.logger.warning('no columns to use for grouping key so returning loaded images as output')
            return base.CallResult(loaded_images_clone)

        flat_rows = {}
        band_index = {}
        for df_row in loaded_images_clone:
            flat_row = flat_rows[df_row[grouping_column]]
            if flat_row == none:
                flat_row = df_row
                flat_rows[df_row[grouping_column]] = flat_row
            band_column = band_index[band]
            flat_row[band_column] = df_row[column_index]


        # set the metadata properly

    def _get_grouping_key_column(self, inputs: base.FileReaderInputs) -> int:
        # use the hyperparam if provided
        column = self.hyperparams['grouping_key_column']
        if column >= 0:
            return column

        # use the column typed as grouping key
        cols = distil_utils.get_operating_columns(inputs, none, ('https://metadata.datadrivendiscovery.org/types/GroupingKey',))
        if len(cols) == 1:
            return cols[0]

        # no suitable column found
        return -1
