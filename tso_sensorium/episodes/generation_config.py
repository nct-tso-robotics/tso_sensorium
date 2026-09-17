"""Configuration schema for dataset generation.

Requires Python 3.9+; the episode assembly classes themselves stay
importable without it.
"""

from __future__ import annotations

import abc
import importlib.util
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, Field, field_validator, model_validator

from tso_sensorium.configuration import ConfigModel, OverrideValue, set_by_path
from tso_sensorium.episodes.dataset_transforms import AnyDatasetTransform
from tso_sensorium.episodes.legend import (
    DATASET_METADATA_FILE_NAME,
    DatasetMetadata,
    load_phase_legend,
)
from tso_sensorium.episodes.schema import DatasetSchema
from tso_sensorium.episodes.table_transforms import AnyTableTransform
from tso_sensorium.export.base import DatasetWriter
from tso_sensorium.export.csv_writer import CsvDatasetWriter
from tso_sensorium.processing.frame_transforms import AnyFrameTransform

if importlib.util.find_spec("lerobot") is not None:
    from tso_sensorium.export.lerobot_action_update_writer import (
        LeRobotActionUpdateWriter,
    )
    from tso_sensorium.export.lerobot_writer import LeRobotDatasetWriter
else:
    LeRobotActionUpdateWriter = None
    LeRobotDatasetWriter = None


class LanguageSource(str, Enum):
    """Source used to populate language labels during generation."""

    ANNOTATION = "annotation"
    PHASE_LEGEND = "phase_legend"


class LegendSource(str, Enum):
    """Source used to resolve phase definitions during generation."""

    AUTO = "auto"
    CONFIG = "config"


class AnnotationsConfig(ConfigModel):
    """How phase and language annotations are applied during generation.

    The phase column receives the integer labels; the language column is
    filled from the legend's instruction variants (one sampled per
    episode, seeded by the episode name so regeneration is reproducible).

    Args:
        file_name: Annotations file inside each episode folder.
        phase_column: Output column holding the integer phase label.
        language_column: Output column holding the language instruction.
        language_source: Whether segment annotations or the phase legend supply
            language labels.
        legend_source: Whether root metadata may override the configured legend.
        legend: Inline dataset metadata, a YAML path, or a ``package://`` asset
            reference. When ``None``, use metadata at the recordings root.
        metadata_file: Metadata file name at the recordings root.
        require_full_coverage: Whether uncovered timestamps discard the
            episode instead of receiving empty labels.
    """

    file_name: str = "annotations.json"
    phase_column: str = "phase"
    language_column: str = "language_instruction"
    language_source: LanguageSource = LanguageSource.ANNOTATION
    legend_source: LegendSource = LegendSource.AUTO
    legend: Optional[DatasetMetadata] = None
    metadata_file: str = DATASET_METADATA_FILE_NAME
    require_full_coverage: bool = False

    @field_validator("legend", mode="before")
    @classmethod
    def load_legend_reference(
        cls, value: DatasetMetadata | dict[str, Any] | str | Path | None
    ) -> DatasetMetadata | dict[str, Any] | None:
        """Resolve explicit legend references through the shared phase loader."""
        if isinstance(value, (str, Path)):
            return load_phase_legend(config_path=value)
        return value

    @model_validator(mode="after")
    def validate_legend_source(self) -> "AnnotationsConfig":
        """Require an inline legend when configuration is authoritative."""
        if self.legend_source == LegendSource.CONFIG and self.legend is None:
            raise ValueError(
                "annotations.legend is required when legend_source is 'config'"
            )
        return self

    def resolve_legend(self, recordings_root: Path) -> DatasetMetadata:
        """Resolve the effective dataset metadata for a recordings root.

        In auto mode, root metadata wins when it defines a legend and the
        inline legend seeds datasets without one. Config mode always uses the
        inline legend.

        Args:
            recordings_root: Directory containing one folder per episode.

        Returns:
            The effective dataset metadata.
        """
        if self.legend_source == LegendSource.CONFIG:
            if self.legend is None:
                raise RuntimeError(
                    "Validated annotations config is missing its required legend"
                )
            return self.legend
        metadata = DatasetMetadata.load(path=Path(recordings_root) / self.metadata_file)
        if metadata.phase_legend:
            return metadata
        if self.legend is not None:
            return self.legend
        return metadata


class VideoSourceConfig(ConfigModel):
    """A recorded video aligned into the episode as frame paths.

    Args:
        video_file: Video file name inside the episode directory.
        timestamps_file: Per-frame timestamp CSV inside the episode
            directory.
        frame_column: Episode table column holding the frame path.
        frames_directory: Directory name for extracted frames, inside the
            episode directory.
        preprocess: Frame transforms applied before saving, in order.
    """

    video_file: str = ""
    timestamps_file: str = ""
    frame_column: str = ""
    frames_directory: str = ""
    preprocess: List[AnyFrameTransform] = Field(default_factory=list)


class StateSourceConfig(ConfigModel):
    """A recorded state CSV aligned into the episode.

    Args:
        state_file: CSV file name inside the episode directory.
        columns: Columns kept in the aligned output.
    """

    state_file: str = ""
    columns: List[str] = Field(default_factory=list)


class WriterConfig(ConfigModel, abc.ABC):
    """Selects and configures the output dataset format."""

    @abc.abstractmethod
    def build(self, recordings_root: Path) -> DatasetWriter:
        """Create the writer for a dataset rooted at ``recordings_root``."""


class CsvWriterConfig(WriterConfig):
    """Per-episode CSV folders.

    Args:
        output_root: Output directory; defaults to the recordings root,
            preserving the historical layout.
    """

    type: Literal["csv"] = "csv"
    output_root: Optional[str] = None

    def build(self, recordings_root: Path) -> DatasetWriter:
        output_root = (
            Path(self.output_root) if self.output_root is not None else recordings_root
        )
        return CsvDatasetWriter(output_root=output_root)


class LeRobotWriterConfig(WriterConfig):
    """LeRobot dataset (format v3.0). Requires the lerobot extra.

    Args:
        output_root: Local directory for dataset storage.
        repo_id: Dataset repository id; defaults to
            "tso/<recordings root name>".
        use_videos: Whether to encode camera streams as videos instead of
            individual images.
    """

    type: Literal["lerobot"] = "lerobot"
    output_root: str = ""
    repo_id: Optional[str] = None
    use_videos: bool = True
    task_column: Optional[str] = None

    def build(self, recordings_root: Path) -> DatasetWriter:
        if LeRobotDatasetWriter is None:
            raise ImportError(
                "LeRobot export requires the lerobot extra:"
                " pip install 'tso-sensorium[lerobot]'"
            )
        if not self.output_root:
            raise ValueError("writer.output_root is required for lerobot export")
        repo_id = (
            self.repo_id if self.repo_id is not None else f"tso/{recordings_root.name}"
        )
        return LeRobotDatasetWriter(
            repo_id=repo_id,
            output_root=self.output_root,
            use_videos=self.use_videos,
            task_column=self.task_column,
        )


class LeRobotActionUpdateWriterConfig(WriterConfig):
    """Transactional action-only update of an existing LeRobot v3 dataset.

    Args:
        dataset_root: Existing LeRobot dataset root to update.
        task_column: Optional episode-table column used to validate task strings.
    """

    type: Literal["lerobot_action_update"] = "lerobot_action_update"
    dataset_root: str = ""
    task_column: Optional[str] = None

    def build(self, recordings_root: Path) -> DatasetWriter:
        if LeRobotActionUpdateWriter is None:
            raise ImportError(
                "LeRobot action update requires the lerobot extra:"
                " pip install 'tso-sensorium[lerobot]'"
            )
        if not self.dataset_root:
            raise ValueError(
                "writer.dataset_root is required for LeRobot action update"
            )
        return LeRobotActionUpdateWriter(
            dataset_root=self.dataset_root,
            task_column=self.task_column,
        )


AnyWriterConfig = Annotated[
    Union[
        CsvWriterConfig,
        LeRobotWriterConfig,
        LeRobotActionUpdateWriterConfig,
    ],
    Field(discriminator="type"),
]


class DatasetGenerationConfig(ConfigModel):
    """A full dataset generation run.

    Args:
        recordings_root: Directory containing one folder per episode.
        dataset_schema: Dataset schema shared by all episodes; the YAML
            key is ``schema``.
        videos: Video sources aligned into each episode.
        states: State CSV sources aligned into each episode.
        table_transforms: Transformations applied to each aligned table,
            in order.
        dataset_transforms: Transformations using all successfully assembled
            episodes.
        writer: Output format selection.
        annotations: Phase and language annotation join; disabled when
            ``None``.
        sync_column: Timestamp column used for synchronization.
        max_sync_difference_seconds: Largest tolerated gap between the
            reference timestamps and each source's closest message.
        save_frames: Whether to extract and save video frames.
        n_jobs: Parallel episode assembly jobs (-1 uses all cores).
        exclude_directory_substrings: Directory names containing any of
            these are not treated as episodes.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    recordings_root: str = ""
    dataset_schema: DatasetSchema = Field(
        alias="schema",
        default_factory=lambda: DatasetSchema(name="", fps=30, cameras=[], arms=[]),
    )
    videos: List[VideoSourceConfig] = Field(default_factory=list)
    states: List[StateSourceConfig] = Field(default_factory=list)
    table_transforms: List[AnyTableTransform] = Field(default_factory=list)
    dataset_transforms: List[AnyDatasetTransform] = Field(default_factory=list)
    writer: AnyWriterConfig = Field(default_factory=CsvWriterConfig)
    annotations: Optional[AnnotationsConfig] = None
    sync_column: str = "time"
    max_sync_difference_seconds: float = 0.1
    save_frames: bool = False
    n_jobs: int = -1
    exclude_directory_substrings: List[str] = Field(default_factory=lambda: [".zarr"])

    @model_validator(mode="after")
    def resolve_writer_task_column(self) -> "DatasetGenerationConfig":
        """Use the configured annotation language column for LeRobot tasks."""
        if self.annotations is None:
            return self
        if (
            isinstance(
                self.writer,
                (LeRobotWriterConfig, LeRobotActionUpdateWriterConfig),
            )
            and self.writer.task_column is None
        ):
            self.writer = self.writer.model_copy(
                update={"task_column": self.annotations.language_column}
            )
        return self


def apply_generation_overrides(
    config: DatasetGenerationConfig,
    overrides: Dict[str, OverrideValue],
) -> DatasetGenerationConfig:
    """Apply dot-path overrides onto a generation configuration.

    Paths address nested fields and list elements, e.g. ``save_frames``,
    ``writer.type``, or ``videos.0.frame_column``. The result is validated
    back into the model, so type errors fail loudly.

    Args:
        config: Base configuration.
        overrides: Mapping of dot path to replacement value.

    Returns:
        New configuration with the overrides applied.
    """
    encoded = config.model_dump(by_alias=True)
    for path, value in overrides.items():
        set_by_path(tree=encoded, path=path, value=value)
    return DatasetGenerationConfig.model_validate(encoded)
