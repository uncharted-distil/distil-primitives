import logging
import os
from d3m.primitive_interfaces.base import Hyperparams

import frozendict  # type: ignore
import imageio  # type: ignore
import numpy as np  # type: ignore
from PIL import Image
from common_primitives import base
from d3m import container, utils as utils
from d3m.base import utils as base_utils, primitives
from d3m.metadata import base as metadata_base
from d3m.primitive_interfaces import base as base_prim
from distil.primitives import utils as distil_utils
from distil.utils import CYTHON_DEP
import version

logger = logging.getLogger(__name__)

class DataFrameSatelliteImageLoaderPrimitive(primitives.FileReaderPrimitiveBase):
    """
    A primitive which reads columns referencing satellite image files, where each file is for a single band.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``image/tiff``) has every filename read into an image
    represented as a numpy array. The input dataframe is then reduced to 1 row / tile
    using the grouping column and the images are loaded into a single numpy array (1 dimension / band).
    By default the resulting column with read arrays is appended
    to existing columns.

    The shape of numpy arrays is H x W x C. C is the number of bands captured,
    H is the height, and W is the width. Images are upsampled to the biggest size.
    dtype is uint16.
    """

    _supported_media_types = (
        'image/tiff',
    )
    _file_structural_type = container.ndarray
    _file_semantic_types = ('http://schema.org/ImageObject',)

    _band_order = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7, '8a': 8, '09': 9, '11': 10, '12': 11}

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '77d20419-aeb6-44f9-8e63-349ea5b654f7',
            'version': version.__version__,
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

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self._max_dimension = 0

    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> container.ndarray:
        return None

    def produce(self, *, inputs: base.FileReaderInputs, timeout: float = None, iterations: int = None) -> base_prim.CallResult[base.FileReaderOutputs]:
        columns_to_use = self._get_columns(inputs.metadata)
        inputs_clone = inputs.copy()
        if len(columns_to_use) == 0:
            return base_prim.CallResult(inputs_clone)
        column_index = columns_to_use[0]

        # need to flatten the dataframe, creating a list of files per tile
        grouping_column = self._get_grouping_key_column(inputs_clone)
        if grouping_column < 0:
            self.logger.warning('no columns to use for grouping key so returning loaded images as output')
            return base_prim.CallResult(inputs_clone)

        base_uri = inputs_clone.metadata.query((metadata_base.ALL_ELEMENTS, column_index))['location_base_uris'][0]
        grouping_name = inputs_clone.columns[grouping_column]
        file_column_name = inputs_clone.columns[column_index]
        band_column_name = 'band'

        # group by grouping key to get all the images loaded in one row
        grouped_images = inputs_clone.groupby([grouping_name], sort=False) \
            .apply(lambda x: self._load_image_group(x[file_column_name], x[band_column_name], base_uri)) \
            .rename(file_column_name).reset_index(drop=True)
        grouped_df = container.DataFrame({file_column_name: grouped_images}, generate_metadata=False)
        grouped_df.metadata = grouped_df.metadata.generate(grouped_df, compact=True)
        grouped_df.metadata = grouped_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), 'http://schema.org/ImageObject')

        # only keep one row / group from the input
        first_band = list(self._band_order.keys())[0]
        first_groups = inputs_clone.loc[inputs_clone[band_column_name] == first_band].reset_index(drop=True)

        outputs = base_utils.combine_columns(first_groups, [column_index], [grouped_df], return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])
        if self.hyperparams['return_result'] == 'append':
            outputs.metadata = self._reassign_boundaries(outputs.metadata, columns_to_use)
        outputs.metadata = outputs.metadata.update((), {'dimension': {'length': outputs.shape[0]}})

        return base_prim.CallResult(outputs)

    def _load_image_group(self, uris, bands, base_uri: str) -> container.ndarray:

        zipped = zip(bands, uris)

        images = list(map(lambda image: self._load_image(image[0], image[1], base_uri), zipped))

        # reshape images (upsample) to have it all fit within an array
        if self._max_dimension == 0:
            self._max_dimension = max(i[1].shape[0] for i in images)

        images_result = [None] * len(self._band_order)
        for band, image in images:
            images_result[self._band_order[band.lower()]] = self._bilinear_resample(image, self._max_dimension)

        output = np.array(images_result)
        output = container.ndarray(output, {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        }, generate_metadata=True)

        return output

    def _load_image(self, band: str, uri: str, base_uri: str):
        image_array = imageio.imread(base_uri + uri)
        image_reader_metadata = image_array.meta

        # make sure the image is of the expected size
        assert image_array.dtype == np.uint16, image_array.dtype

        return (band, image_array)

    def _bilinear_resample(self, x, n=120):
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
