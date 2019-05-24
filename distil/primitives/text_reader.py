import os
from urllib import parse as url_parse

import frozendict  # type: ignore

from d3m import exceptions, utils as d3m_utils
from d3m.metadata import base as metadata_base

import common_primitives
from common_primitives import base

from d3m import container

class TextReaderPrimitive(base.FileReaderPrimitiveBase):
    """
    A primitive which reads columns referencing plain text files.

    Each column which has ``https://metadata.datadrivendiscovery.org/types/FileName`` semantic type
    and a valid media type (``text/plain``) has every filename read as a Python string. By default
    the resulting column with read strings is appended to existing columns.
    """

    _supported_media_types = (
        'text/plain',
    )
    _file_structural_type = str
    _file_semantic_types = ('http://schema.org/Text',)

    metadata = metadata_base.PrimitiveMetadata(
        {
            'id': '8837cc45-457e-4e9d-84c9-09050f6c2070',
            'version': '0.1.0',
            'name': 'Columns text reader',
            'python_path': 'd3m.primitives.data_transformation.encoder.DistilTextReader',
            'source': {
                'name': 'Distil',
                'contact': 'mailto:cbethune@uncharted.software',
                'uris': [
                    'https://github.com/uncharted-distil/distil-primitives/distil/primitives/text_reader.py',
                    'https://github.com/uncharted-distil/distil-primitives',
                ],
            },
            'installation': [{
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package_uri': 'git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives'.format(
                    git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                ),
            }],
            'algorithm_types': [
                metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING,
            ],
            'primitive_family': metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    # TODO: Because we can read only local files, we could change "can_accept" to inspect "location_base_uris" to assure it is a local URI.
    def _read_fileuri(self, metadata: frozendict.FrozenOrderedDict, fileuri: str) -> str:
        parsed_uri = url_parse.urlparse(fileuri, allow_fragments=False)

        if parsed_uri.scheme != 'file':
            raise exceptions.NotSupportedError("Only local files are supported, not '{fileuri}'.".format(fileuri=fileuri))

        if parsed_uri.netloc not in ['', 'localhost']:
            raise exceptions.InvalidArgumentValueError("Invalid hostname for a local file: {fileuri}".format(fileuri=fileuri))

        if not parsed_uri.path.startswith('/'):
            raise exceptions.InvalidArgumentValueError("Not an absolute path for a local file: {fileuri}".format(fileuri=fileuri))

        with open(parsed_uri.path, 'r', encoding='utf8') as file:
            text = file.read()

        text = container.List([text], {
            'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List,
        }, generate_metadata=False)

        text.metadata = text.metadata.update((), {
            'dimension': {
                'None': None,
            },
        })

        return text

