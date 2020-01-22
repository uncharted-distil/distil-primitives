import os
import typing
from distil.utils import CYTHON_DEP
from d3m import container, exceptions, utils as d3m_utils
from d3m.base import utils as base_utils
from d3m.metadata import base as metadata_base, hyperparams
from d3m.primitive_interfaces import base, transformer
import common_primitives

__all__ = ("ExtractColumnsBySemanticTypesPrimitive",)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    semantic_types = hyperparams.Set(
        elements=hyperparams.Hyperparameter[str](""),
        default=("https://metadata.datadrivendiscovery.org/types/Attribute",),
        min_size=1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Semantic types to use to extract columns. If any of them matches, by default.",
    )
    match_logic = hyperparams.Enumeration(
        values=["set", "all", "any"],
        default="set",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description='Should a column have all of semantic types in "semantic_types" to be extracted, or any of them?',
    )
    negate = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description='Should columns which do not match semantic types in "semantic_types" be extracted?',
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified column does not match any semantic type, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description='A set of column indices to not operate on. Applicable only if "use_columns" is not provided.',
    )
    add_index_columns = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="Also include primary index columns if input data has them.",
    )


class RemoveColumnsBySemanticTypesPrimitive(
    transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]
):
    """
    A primitive which extracts columns from input data based on semantic types provided.
    Columns which match any of the listed semantic types are extracted.

    If you want to extract only attributes, you can use ``https://metadata.datadrivendiscovery.org/types/Attribute``
    semantic type (also default).

    For real targets (not suggested targets) use ``https://metadata.datadrivendiscovery.org/types/Target``.
    For this to work, columns have to be are marked as targets by the TA2 in a dataset before passing the dataset
    through a pipeline. Or something else has to mark them at some point in a pipeline.

    It uses ``use_columns`` and ``exclude_columns`` to control which columns it considers.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "a91a043d-a014-49da-b65b-eaa4fed72130",
            "version": "0.1.0",
            "name": "Delete columns by semantic type",
            "python_path": "d3m.primitives.data_transformation.remove_columns_by_semantic_types.RemoveColumnsBySemanticTypesPrimitive",
            "source": {
                "name": common_primitives.__author__,
                "contact": "mailto:balazs.horanyi@qntfy.com",
                "uris": [
                    "https://github.com/uncharted-distil/distil-primitives/distil/primitives/remove_columns_by_semantic_type.py",
                    "https://github.com/uncharted-distil/distil-primitives",
                ],
            },
            "installation": [
                CYTHON_DEP,
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/uncharted-distil/distil-primitives.git@{git_commit}#egg=distil-primitives".format(
                        git_commit=d3m_utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.ARRAY_SLICING, ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_TRANSFORMATION,
        },
    )

    def produce(
            self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> base.CallResult[Outputs]:
        columns_to_remove = self._get_columns(inputs.metadata, self.hyperparams)

        outputs = inputs.remove_columns(columns_to_remove)

        return base.CallResult(outputs)

    @classmethod
    def _can_use_column(
            cls,
            inputs_metadata: metadata_base.DataMetadata,
            column_index: int,
            hyperparams: Hyperparams,
    ) -> bool:
        column_metadata = inputs_metadata.query(
            (metadata_base.ALL_ELEMENTS, column_index)
        )

        semantic_types = column_metadata.get("semantic_types", [])

        if hyperparams["match_logic"] == "set":
            match = len(set(semantic_types) - set(hyperparams["semantic_types"])) == 0

        elif hyperparams["match_logic"] == "all":
            match = all(
                semantic_type in semantic_types
                for semantic_type in hyperparams["semantic_types"]
            )
        elif hyperparams["match_logic"] == "any":
            match = any(
                semantic_type in semantic_types
                for semantic_type in hyperparams["semantic_types"]
            )
        else:
            raise exceptions.UnexpectedValueError(
                'Unknown value of hyper-parameter "match_logic": {value}'.format(
                    value=hyperparams["match_logic"]
                )
            )

        if hyperparams["negate"]:
            return not match
        else:
            return match

    @classmethod
    def _get_columns(
            cls, inputs_metadata: metadata_base.DataMetadata, hyperparams: Hyperparams
    ) -> typing.Sequence[int]:
        def can_use_column(column_index: int) -> bool:
            return cls._can_use_column(inputs_metadata, column_index, hyperparams)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(
            inputs_metadata,
            hyperparams["use_columns"],
            hyperparams["exclude_columns"],
            can_use_column,
        )

        if hyperparams["use_columns"] and columns_not_to_use:
            cls.logger.warning(
                "Not all specified columns match semantic types. Skipping columns: %(columns)s",
                {"columns": columns_not_to_use, },
            )

        return columns_to_use

    @classmethod
    def can_accept(
            cls,
            *,
            method_name: str,
            arguments: typing.Dict[str, typing.Union[metadata_base.Metadata, type]],
            hyperparams: Hyperparams
    ) -> typing.Optional[metadata_base.DataMetadata]:
        output_metadata = super().can_accept(
            method_name=method_name, arguments=arguments, hyperparams=hyperparams
        )

        # If structural types didn't match, don't bother.
        if output_metadata is None:
            return None

        if method_name != "produce":
            return output_metadata

        if "inputs" not in arguments:
            return output_metadata

        inputs_metadata = typing.cast(metadata_base.DataMetadata, arguments["inputs"])

        columns_to_use = cls._get_columns(inputs_metadata, hyperparams)

        output_columns = inputs_metadata.select_columns(columns_to_use)

        return base_utils.combine_columns_metadata(
            inputs_metadata,
            columns_to_use,
            [output_columns],
            return_result="new",
            add_index_columns=hyperparams["add_index_columns"],
        )
