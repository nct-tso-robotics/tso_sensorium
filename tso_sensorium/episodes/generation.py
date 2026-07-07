"""Config-driven dataset generation from recorded episode folders."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import pandas as pd

from tso_sensorium.episodes.annotations import EpisodeAnnotations
from tso_sensorium.episodes.legend import DatasetMetadata
from tso_sensorium.episodes.builder import EpisodeGenerator
from tso_sensorium.episodes.dataset_builder import BuildReport, DatasetBuilder
from tso_sensorium.episodes.generation_config import (
    AnnotationsConfig,
    DatasetGenerationConfig,
)
from tso_sensorium.episodes.schema import Episode
from tso_sensorium.processing.frame_transforms import compose_transforms


def create_episode_assembler(
    config: DatasetGenerationConfig,
) -> Callable[[Path], Episode]:
    """Create the function that turns one recording folder into an episode.

    Args:
        config: Dataset generation configuration.

    Returns:
        Function assembling an Episode from a recording directory.
    """

    def assemble_episode(episode_directory: Path) -> Episode:
        generator = EpisodeGenerator()
        for video in config.videos:
            generator.add_video(
                video_path=Path(episode_directory, video.video_file),
                timestamps_path=Path(episode_directory, video.timestamps_file),
                sync_col_name=config.sync_column,
                frames_output_path=Path(episode_directory, video.frames_directory),
                frame_col_name=video.frame_column,
                preprocess_fn=compose_transforms(transforms=video.preprocess),
                save_frames=config.save_frames,
                max_sync_difference_seconds=config.max_sync_difference_seconds,
            )
        for state in config.states:
            generator.add_state(
                state_data_path=Path(episode_directory, state.state_file),
                sync_col_name=config.sync_column,
                dataset_cols=state.columns,
                max_sync_difference_seconds=config.max_sync_difference_seconds,
            )
        generator.generate_dataset()
        table = generator.dataset
        if config.annotations is not None:
            table = apply_annotations(
                table=table,
                episode_directory=episode_directory,
                annotations_config=config.annotations,
                sync_column=config.sync_column,
            )
        for transform in config.table_transforms:
            table = transform.apply(table=table)
        return Episode(name=episode_directory.name, table=table)

    return assemble_episode


def apply_annotations(
    table: pd.DataFrame,
    episode_directory: Path,
    annotations_config: AnnotationsConfig,
    sync_column: str,
) -> pd.DataFrame:
    """Join phase and language annotations onto an aligned episode table.

    Args:
        table: Aligned episode table.
        episode_directory: Episode folder holding the annotations file.
        annotations_config: How the annotations are applied.
        sync_column: Timestamp column of the table.

    Returns:
        Table with the phase and language columns added.
    """
    annotations = EpisodeAnnotations.load(
        path=episode_directory / annotations_config.file_name
    )
    legend = annotations_config.legend
    if legend is None:
        legend = DatasetMetadata.load(
            path=episode_directory.parent / annotations_config.metadata_file
        )
    episode_rng = random.Random(episode_directory.name)
    sampled_instructions = {}
    for label, definition in sorted(legend.phase_legend.items()):
        if definition.instructions:
            sampled_instructions[label] = episode_rng.choice(definition.instructions)
    phases = []
    languages = []
    for timestamp in table[sync_column]:
        segment = annotations.segment_at(timestamp=timestamp)
        if segment is None:
            if annotations_config.require_full_coverage:
                raise ValueError(
                    f"Timestamp {timestamp} is not covered by annotations"
                    f" in {episode_directory.name}. Discarding episode."
                )
            phases.append(-1)
            languages.append("")
            continue
        phases.append(segment.phase)
        if segment.language is not None:
            language = segment.language
        else:
            language = sampled_instructions.get(segment.phase, "")
        languages.append(language)
    table[annotations_config.phase_column] = phases
    table[annotations_config.language_column] = languages
    return table


def generate_dataset(config: DatasetGenerationConfig) -> BuildReport:
    """Generate a dataset from a folder of recorded episodes.

    Args:
        config: Dataset generation configuration.

    Returns:
        Report with written episode names and per-episode failures.
    """
    if not config.recordings_root:
        raise ValueError("recordings_root is required")
    recordings_root = Path(config.recordings_root)
    dataset_metadata = None
    if config.annotations is not None:
        dataset_metadata = config.annotations.legend
        if dataset_metadata is None:
            dataset_metadata = DatasetMetadata.load(
                path=recordings_root / config.annotations.metadata_file
            )
    builder = DatasetBuilder(
        schema=config.dataset_schema,
        writer=config.writer.build(recordings_root=recordings_root),
        assemble_episode=create_episode_assembler(config=config),
        n_jobs=config.n_jobs,
        exclude_substrings=config.exclude_directory_substrings,
        dataset_metadata=dataset_metadata,
    )
    report = builder.build(recordings_root=recordings_root)
    return report
