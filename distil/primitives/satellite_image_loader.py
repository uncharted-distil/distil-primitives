import time
import logging
import os
import typing
import re
from typing import Tuple
from d3m.metadata.hyperparams import List
from d3m.primitive_interfaces.base import Hyperparams

import frozendict  # type: ignore
import imageio  # type: ignore
import numpy as np  # type: ignore
import pandas as pd
from PIL import Image
from common_primitives import base
from d3m import container, utils as utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base as base_prim
from d3m.primitive_interfaces import transformer
from distil.primitives import utils as distil_utils
import version
import lz4
import struct

from joblib import Parallel, delayed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column does not contain filenames for supported media types, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description='A set of column indices to not operate on. Applicable only if "use_columns" is not provided.',
    )
    return_result = hyperparams.Enumeration(
        values=["append", "replace", "new"],
        default="append",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Should columns with read files be appended, should they replace original columns, or should only columns with read files be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description='Also include primary index columns if input data has them. Applicable only if "return_result" is set to "new".',
    )
    compress_data = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="If True, applies LZO compression algorithm to the data.\
                    Store a header consisting of the dtype character and the data shape as unsigned integers.\
                    Given c struct alignment, will occupy 16 bytes (1 + 4 + 4 + 4 + 3 ) padding",
    )
    n_jobs = hyperparams.Hyperparameter[int](
        default=64,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="The value of the n_jobs parameter for the joblib library",
    )
    band_column = hyperparams.Hyperparameter[str](
        default="band",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Name of the band column used if no band semantic type is present.",
    )


class DataFrameSatelliteImageLoaderPrimitive(
    transformer.TransformerPrimitiveBase[
        container.DataFrame, container.DataFrame, Hyperparams
    ]
):
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

    _supported_media_types = ("image/tiff",)
    _file_structural_type = container.ndarray
    _file_semantic_types = ("http://schema.org/ImageObject",)
    _band_semantic_types = ("https://metadata.datadrivendiscovery.org/types/Band",)

    _BAND_ORDER = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "7": 6,
        "8": 7,
        "8a": 8,
        "9": 9,
        "11": 10,
        "12": 11,
    }

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "77d20419-aeb6-44f9-8e63-349ea5b654f7",
            "version": version.__version__,
            "name": "Columns satellite image loader",
            "python_path": "d3m.primitives.data_transformation.satellite_image_loader.DistilSatelliteImageLoader",
            "keywords": ["satellite", "image", "reader", "tiff"],
            "source": {
                "name": "Distil",
                "contact": "mailto:cbethune@uncharted.software",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/blob/main/distil/primitives/satellite_image_loader.py",
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
                metadata_base.PrimitiveAlgorithmType.FILE_MANIPULATION,
            ],
            "supported_media_types": _supported_media_types,
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams) -> None:
        super().__init__(hyperparams=hyperparams)

    def _read_fileuri(
        self, metadata: frozendict.FrozenOrderedDict, fileuri: str
    ) -> container.ndarray:
        return None

    def _can_use_column(
        self, inputs_metadata: metadata_base.DataMetadata, column_index: int
    ) -> bool:
        column_metadata = inputs_metadata.query(
            (metadata_base.ALL_ELEMENTS, column_index)
        )

        if column_metadata["structural_type"] != str:
            return False

        semantic_types = column_metadata.get("semantic_types", [])
        media_types = set(column_metadata.get("media_types", []))

        if (
            "https://metadata.datadrivendiscovery.org/types/FileName" in semantic_types
            and media_types <= set(self._supported_media_types)
        ):
            return True

        return False

    def _get_columns(
        self, inputs_metadata: metadata_base.DataMetadata
    ) -> typing.List[int]:
        def can_use_column(column_index: int) -> bool:
            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            inputs_metadata,
            self.hyperparams["use_columns"],
            self.hyperparams["exclude_columns"],
            can_use_column,
        )

        # We are OK if no columns ended up being read.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams["use_columns"] and columns_not_to_use:
            self.logger.warning(
                "Not all specified columns contain filenames for supported media types. Skipping columns: %(columns)s",
                {
                    "columns": columns_not_to_use,
                },
            )

        return columns_to_use

    def _get_band_column(self, inputs_metadata: metadata_base.DataMetadata):
        return inputs_metadata.list_columns_with_semantic_types(
            self._band_semantic_types
        )

    def produce(
        self,
        *,
        inputs: base.FileReaderInputs,
        timeout: float = None,
        iterations: int = None,
    ) -> base_prim.CallResult[base.FileReaderOutputs]:
        logger.debug(f"Producing {__name__}")

        columns_to_use = self._get_columns(inputs.metadata)
        inputs_clone = inputs.copy()
        if len(columns_to_use) == 0:
            return base_prim.CallResult(inputs_clone)
        column_index = columns_to_use[0]

        band_column_indices = self._get_band_column(inputs.metadata)
        if len(band_column_indices) == 0:
            band_column_name = self.hyperparams["band_column"]
        else:
            band_column_name = inputs.columns[band_column_indices[0]]

        # need to flatten the dataframe, creating a list of files per tile
        grouping_column = self._get_grouping_key_column(inputs_clone)
        if grouping_column < 0:
            self.logger.warning(
                "no columns to use for grouping key so returning loaded images as output"
            )
            return base_prim.CallResult(inputs_clone)

        base_uri = inputs_clone.metadata.query(
            (metadata_base.ALL_ELEMENTS, column_index)
        )["location_base_uris"][0]
        grouping_name = inputs_clone.columns[grouping_column]
        file_column_name = inputs_clone.columns[column_index]

        start = time.time()
        logger.debug("Loading images")

        # group by grouping key to get all the images loaded in one row
        groups = inputs_clone.groupby([grouping_name], sort=False)

        # use the max dimension for the first group as the max dimension for all groups
        group_key = groups[grouping_name].first()[0]
        max_dimension = self._get_group_image_size(
            groups.get_group(group_key), file_column_name, band_column_name, base_uri
        )

        # load images for each group and store them in a matrix of [band, x, y]
        jobs = [
            delayed(self._load_image_group)(
                group[1][file_column_name],
                group[1][band_column_name],
                base_uri,
                max_dimension,
            )
            for group in tqdm(groups, total=len(groups))
        ]
        groups = Parallel(
            n_jobs=self.hyperparams["n_jobs"], backend="loky", verbose=10
        )(jobs)
        end = time.time()
        logger.debug(f"Loaded images in {end-start}s")

        logger.debug("Updating metadata")
        start = time.time()

        # auto-generate metdata for one row's worth of data - necessary to avoid having the generation step traverse all of the data
        # which is extremely slow
        first_df = container.DataFrame(
            {file_column_name: [groups[0]]}, generate_metadata=True
        ).reset_index(drop=True)
        rest_df = container.DataFrame({file_column_name: groups[1:]})
        grouped_df = first_df.append(rest_df, ignore_index=True)

        grouped_df.metadata = grouped_df.metadata.update(
            (), {"dimension": {"length": grouped_df.shape[0]}}
        )
        grouped_df.metadata = grouped_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0), "http://schema.org/ImageObject"
        )
        end = time.time()
        logger.debug(f"Updated metadata in {end-start}s")

        # only keep one row / group from the input - use the first band value to select against
        first_band = inputs_clone[band_column_name][0]
        first_groups = inputs_clone.loc[
            inputs_clone[band_column_name] == first_band
        ].reset_index(drop=True)

        outputs = base_utils.combine_columns(
            first_groups,
            [column_index],
            [grouped_df],
            return_result=self.hyperparams["return_result"],
            add_index_columns=self.hyperparams["add_index_columns"],
        )
        if self.hyperparams["return_result"] == "append":
            outputs.metadata = self._reassign_boundaries(
                outputs.metadata, columns_to_use
            )
        outputs.metadata = outputs.metadata.update(
            (), {"dimension": {"length": outputs.shape[0]}}
        )

        polygon_columns = outputs.metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/LocationPolygon",)
        )
        vector_columns = outputs.metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/FloatVector",)
        )
        if len(vector_columns) > 0 and len(polygon_columns) == 0:
            outputs.metadata = outputs.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, vector_columns[0]),
                "https://metadata.datadrivendiscovery.org/types/LocationPolygon",
            )

        return base_prim.CallResult(outputs)

    def _reassign_boundaries(
        self, inputs_metadata: metadata_base.DataMetadata, columns: typing.List[int]
    ) -> metadata_base.DataMetadata:
        """
        Moves metadata about boundaries from the filename column to image object column.
        """

        outputs_metadata = inputs_metadata
        columns_length = inputs_metadata.query((metadata_base.ALL_ELEMENTS,))[
            "dimension"
        ]["length"]

        for column_index in range(columns_length):
            column_metadata = outputs_metadata.query_column(column_index)

            if "boundary_for" not in column_metadata:
                continue

            # TODO: Support also "column_name" boundary metadata.
            if "column_index" not in column_metadata["boundary_for"]:
                continue

            try:
                i = columns.index(column_metadata["boundary_for"]["column_index"])
            except ValueError:
                continue

            outputs_metadata = outputs_metadata.update_column(
                column_index,
                {
                    "boundary_for": {
                        # We know that "columns" were appended at the end.
                        "column_index": columns_length
                        - len(columns)
                        + i,
                    }
                },
            )

        return outputs_metadata

    def _load_image_group(
        self, uris: List[str], bands: List[str], base_uri: str, max_dimension: int
    ) -> container.ndarray:

        zipped = zip(bands, uris)
        images = list(
            map(lambda image: self._load_image(image[0], image[1], base_uri), zipped)
        )

        # reshape images (upsample) to have it all fit within an array
        if self.hyperparams["compress_data"]:
            # Store a header consisting of the dtype character and the data shape as unsigned integers.
            # Given c struct alignment, will occupy 16 bytes (1 + 4 + 4 + 4 + 3 padding)
            output_bytes = bytearray(
                struct.pack(
                    "cIII",
                    bytes(images[0][1].dtype.char.encode()),
                    len(images),
                    max_dimension,
                    max_dimension,
                )
            )
            for band, image in images:
                output_bytes.extend(
                    self._bilinear_resample(image, max_dimension).tobytes()
                )
            output_compressed_bytes = lz4.frame.compress(bytes(output_bytes))
            output = np.frombuffer(
                output_compressed_bytes,
                dtype="uint8",
                count=len(output_compressed_bytes),
            )
        else:
            output = np.ndarray(
                (
                    len(DataFrameSatelliteImageLoaderPrimitive._BAND_ORDER),
                    max_dimension,
                    max_dimension,
                )
            )
            for band, image in images:
                band_idx = DataFrameSatelliteImageLoaderPrimitive._BAND_ORDER[
                    self._normalized_band_id(band)
                ]
                output[band_idx] = self._bilinear_resample(image, max_dimension)

        output = container.ndarray(
            output,
            {
                "schema": metadata_base.CONTAINER_SCHEMA_VERSION,
                "structural_type": container.ndarray,
            },
            generate_metadata=True,
        )

        return output

    def _load_image(self, band: str, uri: str, base_uri: str) -> Tuple[str, List[str]]:
        image_array = imageio.imread(base_uri + uri)
        # make sure the image is of the expected size
        assert image_array.dtype == np.uint16, image_array.dtype

        return (band, image_array)

    def _bilinear_resample(self, x: np.ndarray, n: int = 120) -> np.array:
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
        cols = distil_utils.get_operating_columns(
            inputs,
            self.hyperparams["use_columns"],
            ("https://metadata.datadrivendiscovery.org/types/GroupingKey",),
        )

        if len(cols) == 0:
            cols = distil_utils.get_operating_columns(
                inputs,
                self.hyperparams["use_columns"],
                (
                    "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey",
                ),
            )

        if len(cols) == 1:
            return cols[0]

        # no suitable column found
        return -1

    def _generate_metadata(self, inputs: container.DataFrame) -> container.DataFrame:
        return None

    def _get_group_image_size(
        self,
        group: pd.DataFrame,
        file_column_name: str,
        band_column_name: str,
        base_uri: str,
    ) -> int:
        # load group images and find the max dimension in the set
        max_dimension = 0
        zipped = zip(group[band_column_name], group[file_column_name])
        images = list(
            map(lambda image: self._load_image(image[0], image[1], base_uri), zipped)
        )
        if max_dimension == 0:
            max_dimension = max(i[1].shape[0] for i in images)
        return max_dimension

    def _normalized_band_id(self, band_id: str) -> str:
        # force to lower case and remove any prefixed zeroes
        return re.sub("^0+", "", band_id.lower())
