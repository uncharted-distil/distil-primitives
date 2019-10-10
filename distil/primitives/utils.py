import io
from typing import Sequence
import sys
import importlib.util
import importlib.machinery

import numpy as np

from d3m.metadata import base
from d3m import container

MISSING_VALUE_INDICATOR = '__miss_salt_8acf6447-fd14-480e-9cfb-0cb46accfafd'
SINGLETON_INDICATOR     = '__sing_salt_6df854b8-a0ba-41ba-b598-ddeba2edfb53'

CATEGORICALS = ('https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'https://metadata.datadrivendiscovery.org/types/OrdinalData',
                'http://schema.org/DateTime')


def metadata_to_str(metadata: base.Metadata, selector: base.Selector = None) -> str:
    buf = io.StringIO()
    metadata.pretty_print(selector, buf)
    return buf.getvalue()

def get_operating_columns(inputs: container.DataFrame, use_columns: Sequence[int],
                          semantic_types: Sequence[str], require_attribute: bool = True) -> Sequence[int]:
    # use caller supplied columns if supplied
    cols = set(use_columns)
    type_cols = set(inputs.metadata.list_columns_with_semantic_types(semantic_types))
    if require_attribute:
        attributes = set(inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Attribute',)))
        type_cols = type_cols & attributes

    if len(cols) > 0:
        cols = type_cols & cols
    else:
        cols = type_cols
    return list(cols)

def get_operating_columns_structural_type(inputs: container.DataFrame, use_columns: Sequence[int],
                                          structural_types: Sequence[str], require_attribute: bool = True) -> Sequence[int]:
    # use caller supplied columns if supplied
    cols = set(use_columns)
    type_cols = set(inputs.metadata.list_columns_with_structural_types(structural_types))
    if require_attribute:
        attributes = set(inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Attribute',)))
        type_cols = type_cols & attributes

    if len(cols) > 0:
        cols = type_cols & cols
    else:
        cols = type_cols
    return list(cols)

def lazy_load(fullname: str):
    # lazy load a module - needed for imports that trigger long running static model
    # loads
    if fullname in sys.modules:
        return sys.modules[fullname]
    else:
        spec = importlib.util.find_spec(fullname)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)
        # Make module with proper locking and get it inserted into sys.modules.
        loader.exec_module(module)
        return module