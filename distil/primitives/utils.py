#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import importlib.machinery
import importlib.util
import io
import sys
import cProfile
import pstats
from typing import Any, Callable, Dict, Sequence, Tuple
import logging
import time

from d3m import container
from d3m.metadata import base

MISSING_VALUE_INDICATOR = "__miss_salt_8acf6447-fd14-480e-9cfb-0cb46accfafd"
SINGLETON_INDICATOR = "__sing_salt_6df854b8-a0ba-41ba-b598-ddeba2edfb53"

CATEGORICALS = (
    "https://metadata.datadrivendiscovery.org/types/CategoricalData",
    "https://metadata.datadrivendiscovery.org/types/OrdinalData",
    "http://schema.org/DateTime",
    "http://schema.org/Boolean",
)

VECTOR = "https://metadata.datadrivendiscovery.org/types/FloatVector"


def metadata_to_str(metadata: base.Metadata, selector: base.Selector = None) -> str:
    buf = io.StringIO()
    metadata.pretty_print(selector, buf)
    return buf.getvalue()


def get_operating_columns(
    inputs: container.DataFrame,
    use_columns: Sequence[int],
    semantic_types: Sequence[str],
    require_attribute: bool = True,
) -> Sequence[int]:
    # use caller supplied columns if supplied
    cols = set(use_columns)
    type_cols = set(inputs.metadata.list_columns_with_semantic_types(semantic_types))
    if require_attribute:
        attributes = set(
            inputs.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/Attribute",)
            )
        )
        type_cols = type_cols & attributes

    if len(cols) > 0:
        cols = type_cols & cols
    else:
        cols = type_cols
    return list(cols)


def get_operating_columns_structural_type(
    inputs: container.DataFrame,
    use_columns: Sequence[int],
    structural_types: Sequence[str],
    require_attribute: bool = True,
) -> Sequence[int]:
    # use caller supplied columns if supplied
    cols = set(use_columns)
    type_cols = set(
        inputs.metadata.list_columns_with_structural_types(structural_types)
    )
    if require_attribute:
        attributes = set(
            inputs.metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/Attribute",)
            )
        )
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


# an annotation that can be added to function calls to time their execution
def timed(fcn: Callable) -> Callable:
    def wrapped(*args: Tuple, **kwargs: Dict[str, Any]) -> Any:
        logger = logging.getLogger(__name__)
        start = time.time()
        logger.debug(f"Executing: {fcn.__module__}.{fcn.__name__}")
        result = fcn(*args, **kwargs)
        end = time.time()
        logger.debug(f"Finished: {fcn.__module__}.{fcn.__name__} in {end - start} ms")
        return result

    return wrapped


# an annotation that can be added to function calls to fully profile their execution
def profiled(fcn: Callable) -> Callable:
    def wrapped(*args: Tuple, **kwargs: Dict[str, Any]) -> Any:
        pr = cProfile.Profile()
        pr.enable()
        fcn(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby).strip_dirs()
        ps.print_stats()
        print(s.getvalue())

    return wrapped
