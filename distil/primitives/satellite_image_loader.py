import os

import frozendict  # type: ignore
import imageio  # type: ignore
import numpy as np  # type: ignore

from PIL import Image

from d3m import container, utils as utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base as base_prim

import common_primitives
from common_primitives import base

from distil.utils import CYTHON_DEP
from distil.primitives import utils as distil_utils

class DataFrameSatelliteImageLoaderPrimitive(base.FileReaderPrimitiveBase):
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

    _band_order = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7, '8A': 8, '09': 9, '11': 10, '12': 11}

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '77d20419-aeb6-44f9-8e63-349ea5b654f7',
            'version': '0.1.0',
            'name': 'Columns satellite image loader',
            'python_path': 'd3m.primitives.data_preprocessing.satellite_image_loader.DistilSatelliteImageLoader',
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
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
            ],
            'supported_media_types': _supported_media_types,
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        }
    )

    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> container.ndarray:
        return None

    def produce(self, *, inputs: base.FileReaderInputs, timeout: float = None, iterations: int = None) -> base_prim.CallResult[base.FileReaderOutputs]:
        columns_to_use = self._get_columns(inputs.metadata)
        inputs_clone = inputs.copy()
        if len(columns_to_use) == 0:
            return inputs_clone
        column_index = columns_to_use[0]

        # need to flatten the dataframe, creating a list of files per tile
        grouping_column = self._get_grouping_key_column(inputs_clone)
        if grouping_column < 0:
            self.logger.warning('no columns to use for grouping key so returning loaded images as output')
            return inputs_clone

        base_uri = inputs_clone.metadata.query((metadata_base.ALL_ELEMENTS, column_index))['location_base_uris'][0]
        grouping_name = inputs_clone.columns[grouping_column]
        file_column_name = inputs_clone.columns[column_index]
        band_column_name = 'band'

        # group by grouping key to get all the images loaded in one row
        grouped_images = inputs_clone.groupby([grouping_name]) \
            .apply(lambda x: self._load_image_group(x[file_column_name], x[band_column_name], base_uri)) \
            .rename(file_column_name + '_loaded')

        # only keep one row / group from the input
        first_band = list(self._band_order.keys())[0]
        first_groups = inputs_clone.loc[inputs_clone[band_column_name] == first_band]
        joined_df = first_groups.join(grouped_images, on=grouping_name)

        # update the metadata
        joined_df.metadata = joined_df.metadata.generate(joined_df)

        return base_prim.CallResult(joined_df)

    def _load_image_group(self, uris, bands, base_uri: str) -> container.ndarray:

        images = list(map(lambda uri: self._load_image(uri, base_uri), uris))

        # reshape images (upsample) to have it all fit within an array
        max_dimension = max(i.shape[0] for i in images)
        images_result = [None] * len(self._band_order)
        for i in range(len(images)):
            images_result[self._band_order[bands[i]]] = self._bilinear_upsample(images[i], max_dimension)

        output = np.array(images_result)
        output = container.ndarray(output, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        }, generate_metadata=False)

        return output

    def _load_image(self, uri: str, base_uri: str):
        image_array = imageio.imread(base_uri + uri)
        image_reader_metadata = image_array.meta

        # make sure the image is of the expected size
        assert image_array.dtype == np.uint16, image_array.dtype

        return image_array

    def _bilinear_upsample(self, x, n=120):
        dtype = x.dtype
        assert len(x.shape) == 2
        if (x.shape[0] == n) and (x.shape[1] == n):
            return x
        else:
            x = x.astype(np.float)
            x = Image.fromarray(x)
            x = x.resize((n, n), Image.BILINEAR)
            x = np.array(x)
            x = x.astype(dtype)

        return x

    def _get_grouping_key_column(self, inputs: base.FileReaderInputs) -> int:
        # use the column typed as grouping key
        cols = distil_utils.get_operating_columns(inputs, self.hyperparams['use_columns'],
            ('https://metadata.datadrivendiscovery.org/types/GroupingKey',))

        if len(cols) == 1:
            return cols[0]

        # no suitable column found
        return -1
