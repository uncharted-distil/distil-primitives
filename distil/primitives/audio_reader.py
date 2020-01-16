import os
import signal
import subprocess

import logging
import copy

from d3m import container, utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer


from typing import List, Sequence, Optional, Tuple, Union
import numpy as np
import pandas as pd
import soundfile as sf
import prctl
import tempfile
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.io import wavfile
from distil.utils import CYTHON_DEP

__all__ = ('AudioDatasetLoaderPrimitive',)

logger = logging.getLogger(__name__)

class Hyperparams(hyperparams.Hyperparams):
    sample = hyperparams.Hyperparameter[float](
        default=1.0,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='a value ranging from 0.0 to 1.0 indicating how much of the source data to load'
    )
    dataframe_resource = hyperparams.Hyperparameter[Union[str, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description=".",
    )

class WavInput:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


def convert_load_file(fileuri, start, end):

    with tempfile.NamedTemporaryFile(mode='rb') as output_file:
        # We use ffmpeg to convert all audio files to same format.
        args = [
            'ffmpeg',
            '-y',  # Always overwrite existing files.
            '-nostdin',  # No interaction.
            '-i', fileuri,  # Input file.
            '-vn',  # There is no video.
            #'-acodec', 'pcm_f32le',  # We want everything in float32 dtype.
            '-f', 'wav',  # This will give us sample rate available in metadata.
            output_file.name,  # Output file.
        ]

        try:
            result = subprocess.run(
                args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                # Setting "pdeathsig" will make the ffmpeg process be killed if our process dies for any reason.
                encoding='utf8', check=True, preexec_fn=lambda: prctl.set_pdeathsig(signal.SIGKILL),
            )
        except subprocess.CalledProcessError as error:
            logger.error("Error running ffmpeg: %(stderr)s", {'stderr': error.stderr})
            raise

        info = sf.info(output_file.name)

        if start is not None and end is not None and info.duration > 0:
            start = int(info.frames * (start / info.duration))
            end = int(info.frames * (end / info.duration))
            audio_array, sample_rate = sf.read(output_file.name, start=start, stop=end, dtype='int16')
        else:
            audio_array, sample_rate = sf.read(output_file.name, dtype='int16')

    if len(audio_array.shape) == 1:
        audio_array = audio_array.reshape(-1, 1)

    if audio_array.shape[0] < sample_rate:
        audio_array = np.vstack([audio_array, np.zeros((sample_rate - audio_array.shape[0], audio_array.shape[1]), dtype='int16')])

    return WavInput(audio_array, sample_rate)



class AudioDatasetLoaderPrimitive(transformer.TransformerPrimitiveBase[container.Dataset, container.List, Hyperparams]):
    """
    A primitive which reads columns referencing audio files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``audio/aiff``, ``audio/flac``, ``audio/ogg``, ``audio/wav``, ``audio/mpeg``)
    has every filename read into an audio represented as a numpy array. By default the resulting column
    with read arrays is appended to existing columns.

    The shape of numpy arrays is S x C. S is the number of samples, C is the number of
    channels in an audio (e.g., C = 1 for mono, C = 2 for stereo). dtype is float32.

    """


    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': 'f2a0cf71-0f61-41a7-a0ad-b907083ae56c',
            'version': '0.2.0',
            'name': "Load audio collection from dataset into a single dataframe",
            'python_path': 'd3m.primitives.data_preprocessing.audio_reader.DistilAudioDatasetLoader',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/audio_reader.py',
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
            'primitive_family': metadata_base.PrimitiveFamily.DATA_PREPROCESSING,
        },
    )


    def produce(self, *, inputs: container.Dataset, timeout: float = None, iterations: int = None) -> base.CallResult[container.DataFrame]:
        logger.debug(f'Running {__name__}')

        # get the learning data (the dataset entry point)
        learning_id, learning_df = base_utils.get_tabular_resource(inputs, None, pick_entry_point=True)
        learning_df = learning_df.head(int(learning_df.shape[0]*self.hyperparams['sample']))
        learning_df.metadata = self._update_metadata(inputs.metadata, learning_id, learning_df)

        logger.debug(f'\n{learning_df}')

        return base.CallResult(learning_df)


    def produce_collection(self, *, inputs: container.Dataset, timeout: float = None, iterations: int = None) -> base.CallResult[container.List]:
        logger.debug(f'Running {__name__}')

        # get the learning data (the dataset entry point)
        learning_id, learning_df = base_utils.get_tabular_resource(inputs, None, pick_entry_point=True)

        learning_df = learning_df.head(int(learning_df.shape[0]*self.hyperparams['sample']))
        learning_df.metadata = self._update_metadata(inputs.metadata, learning_id, learning_df)

        # find the column that is acting as the foreign key and extract the resource + column it references
        for i in range(learning_df.metadata.query((metadata_base.ALL_ELEMENTS,))['dimension']['length']):
            column_metadata = learning_df.metadata.query_column(i)
            if 'foreign_key' in column_metadata and column_metadata['foreign_key']['type'] is 'COLUMN':
                resource_id = column_metadata['foreign_key']['resource_id']
                file_column_idx = column_metadata['foreign_key']['column_index']

        # get the learning data (the dataset entry point)
        collection_id, collection_df = base_utils.get_tabular_resource(inputs, resource_id)

        collection_df = collection_df.head(learning_df.shape[0])
        collection_df.metadata = self._update_metadata(inputs.metadata, collection_id, collection_df)

        # get the base path
        base_path = collection_df.metadata.query((metadata_base.ALL_ELEMENTS, file_column_idx))['location_base_uris'][0]

        # create fully resolved paths and load
        paths = learning_df.iloc[:, file_column_idx] #TODO: remove, unused?

        file_paths = []
        for i, row in learning_df.iterrows():
            if i % 100 == 0:
                logger.debug(f"Loaded {i} / {len(learning_df.index)} files")
            try:
                file_paths.append((os.path.join(base_path, row['filename']), row.start, row.end))
            except AttributeError as e:
                logger.warning('no start/end ts for {}'.format(row))
                file_paths.append((os.path.join(base_path, row['filename']), None, None))

        outputs = self._audio_load(file_paths)

        logger.debug(f'\n{outputs}')

        result_df = pd.DataFrame({'audio': outputs}) # d3m container takes for_ever_
        return base.CallResult(container.DataFrame(result_df, generate_metadata=False))


    @classmethod
    def _update_metadata(cls, metadata: metadata_base.DataMetadata, resource_id: metadata_base.SelectorSegment,
                         for_value: Optional[container.DataFrame]) -> metadata_base.DataMetadata:
        resource_metadata = dict(metadata.query((resource_id,)))

        if 'structural_type' not in resource_metadata or not issubclass(resource_metadata['structural_type'], container.DataFrame):
            raise TypeError("The Dataset resource is not a DataFrame, but \"{type}\".".format(
                type=resource_metadata.get('structural_type', None),
            ))

        resource_metadata.update({'schema': metadata_base.CONTAINER_SCHEMA_VERSION,})
        new_metadata = metadata_base.DataMetadata()
        new_metadata = metadata.copy_to(new_metadata, (resource_id,))
        new_metadata = new_metadata.remove_semantic_type((), 'https://metadata.datadrivendiscovery.org/types/DatasetEntryPoint')

        return new_metadata



    @classmethod
    def _audio_load(cls, files_in: Sequence[Tuple]) -> List:
        jobs = [delayed(convert_load_file)(f[0], float(f[1]), float(f[2])) for f in tqdm(files_in, total=len(files_in))]
        files_out = Parallel(n_jobs=64, backend='loky', verbose=10)(jobs)
        return files_out