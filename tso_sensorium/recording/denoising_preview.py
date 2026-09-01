"""Denoising statistics for the recording dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from tso_sensorium.episodes.dataset_builder import discover_episode_directories
from tso_sensorium.episodes.dataset_transforms import (
    JsonValue,
    PercentileDenoiseColumns,
)
from tso_sensorium.episodes.generation import create_episode_assembler
from tso_sensorium.episodes.generation_config import DatasetGenerationConfig
from tso_sensorium.episodes.schema import Episode

HISTOGRAM_BIN_COUNT = 100


@dataclass(frozen=True)
class DenoisingGroup:
    """Configured action component group and its generation override path."""

    name: str
    columns: tuple[str, ...]
    percentile: float
    override_path: str


@dataclass(frozen=True)
class DenoisingGroupSamples:
    """Action magnitudes and optional phase labels for one group."""

    group: DenoisingGroup
    magnitudes: NDArray[np.float64]
    phases: Optional[NDArray[np.int64]]


@dataclass(frozen=True)
class DenoisingPreviewData:
    """Cached action magnitudes assembled from the current recordings root."""

    groups: tuple[DenoisingGroupSamples, ...]
    episode_count: int
    failed: dict[str, str]

    def to_payload(
        self,
        percentiles: Optional[dict[str, float]] = None,
        phase_names: Optional[dict[int, str]] = None,
    ) -> dict[str, JsonValue]:
        """Calculate thresholds and zeroing counts for selected percentiles.

        Args:
            percentiles: Values keyed by generation override path. Missing
                paths retain their configured percentile.
            phase_names: Human-readable names keyed by phase label.

        Returns:
            JSON-compatible preview data for the dashboard.

        Raises:
            ValueError: If an override is unknown or outside ``[0, 100]``.
        """
        requested = percentiles or {}
        known_paths = {samples.group.override_path for samples in self.groups}
        unknown_paths = sorted(set(requested) - known_paths)
        if unknown_paths:
            raise ValueError(
                f"Unknown denoising percentile override paths: {unknown_paths}"
            )

        groups = []
        for samples in self.groups:
            percentile = requested.get(
                samples.group.override_path, samples.group.percentile
            )
            if (
                isinstance(percentile, bool)
                or not isinstance(percentile, Real)
                or not np.isfinite(percentile)
                or not 0.0 <= float(percentile) <= 100.0
            ):
                raise ValueError(
                    "Denoising percentile for "
                    f"'{samples.group.override_path}' must be a finite number in "
                    f"[0, 100], got {percentile}"
                )
            groups.append(
                _group_payload(
                    samples=samples,
                    percentile=float(percentile),
                    phase_names=phase_names or {},
                )
            )
        return {
            "available": bool(self.groups),
            "episode_count": self.episode_count,
            "failed": self.failed,
            "groups": groups,
        }


def configured_denoising_groups(
    config: DatasetGenerationConfig,
) -> list[DenoisingGroup]:
    """Discover percentile-denoising groups and their override paths.

    Args:
        config: Active dataset-generation configuration.

    Returns:
        Groups in configuration order.
    """
    groups = []
    for transform_index, transform in enumerate(config.dataset_transforms):
        if not isinstance(transform, PercentileDenoiseColumns):
            continue
        for group_name, group in transform.column_groups.items():
            groups.append(
                DenoisingGroup(
                    name=group_name,
                    columns=tuple(group.columns),
                    percentile=group.percentile,
                    override_path=(
                        f"dataset_transforms.{transform_index}.column_groups."
                        f"{group_name}.percentile"
                    ),
                )
            )
    return groups


def build_denoising_preview_data(
    config: DatasetGenerationConfig,
    episodes: list[Episode],
    failed: Optional[dict[str, str]] = None,
) -> DenoisingPreviewData:
    """Collect action magnitudes from assembled episodes.

    Args:
        config: Active dataset-generation configuration.
        episodes: Episodes after table-level transforms and before dataset
            transforms.
        failed: Episode assembly failures to expose in the preview.

    Returns:
        Cached magnitudes for interactive threshold previews.

    Raises:
        ValueError: If a configured action or phase column is missing.
    """
    groups = configured_denoising_groups(config=config)
    phase_column = (
        config.annotations.phase_column if config.annotations is not None else None
    )
    group_samples = []
    for group in groups:
        episode_magnitudes = []
        episode_phases = []
        for episode in episodes:
            missing_columns = [
                column for column in group.columns if column not in episode.table
            ]
            if missing_columns:
                raise ValueError(
                    f"Episode '{episode.name}' is missing denoising columns for "
                    f"group '{group.name}': {missing_columns}"
                )
            values = episode.table[list(group.columns)].to_numpy(dtype=float)
            episode_magnitudes.append(np.linalg.norm(x=values, axis=1))
            if phase_column is not None:
                if phase_column not in episode.table:
                    raise ValueError(
                        f"Episode '{episode.name}' is missing phase column "
                        f"'{phase_column}'"
                    )
                episode_phases.append(
                    episode.table[phase_column].to_numpy(dtype=np.int64)
                )
        magnitudes = (
            np.concatenate(episode_magnitudes).astype(np.float64, copy=False)
            if episode_magnitudes
            else np.empty(shape=(0,), dtype=np.float64)
        )
        phases = (
            np.concatenate(episode_phases).astype(np.int64, copy=False)
            if phase_column is not None and episode_phases
            else None
        )
        group_samples.append(
            DenoisingGroupSamples(
                group=group,
                magnitudes=magnitudes,
                phases=phases,
            )
        )
    return DenoisingPreviewData(
        groups=tuple(group_samples),
        episode_count=len(episodes),
        failed=failed or {},
    )


def load_denoising_preview_data(
    config: DatasetGenerationConfig,
    recordings_root: Path | str,
) -> DenoisingPreviewData:
    """Assemble the current recordings without applying dataset transforms.

    Args:
        config: Active dataset-generation configuration.
        recordings_root: Folder containing recorded episode directories.

    Returns:
        Cached action magnitudes and per-episode assembly failures.

    Raises:
        ValueError: If no episode can be assembled.
    """
    episode_directories = discover_episode_directories(
        recordings_root=recordings_root,
        exclude_substrings=config.exclude_directory_substrings,
    )
    assemble_episode = create_episode_assembler(config=config)
    episodes = []
    failed = {}
    for episode_directory in episode_directories:
        try:
            episodes.append(assemble_episode(episode_directory))
        except (ValueError, OSError) as error:
            failed[episode_directory.name] = str(error)
    if not episodes:
        raise ValueError(
            "No recorded episodes could be assembled for the denoising preview"
        )
    return build_denoising_preview_data(
        config=config,
        episodes=episodes,
        failed=failed,
    )


def _group_payload(
    samples: DenoisingGroupSamples,
    percentile: float,
    phase_names: dict[int, str],
) -> dict[str, JsonValue]:
    """Render one group at a selected percentile."""
    nonzero_magnitudes = samples.magnitudes[samples.magnitudes > 0]
    if len(nonzero_magnitudes) == 0:
        raise ValueError(
            f"Cannot preview denoising group '{samples.group.name}': "
            "the dataset contains no nonzero movement"
        )
    threshold = float(np.percentile(a=nonzero_magnitudes, q=percentile))
    log_magnitudes = np.log10(nonzero_magnitudes)
    raw_counts, bin_edges = np.histogram(
        a=log_magnitudes,
        bins=HISTOGRAM_BIN_COUNT,
    )
    retained_magnitudes = nonzero_magnitudes[nonzero_magnitudes >= threshold]
    denoised_counts, _ = np.histogram(
        a=np.log10(retained_magnitudes),
        bins=bin_edges,
    )
    phases = []
    if samples.phases is not None:
        for phase_label in sorted(np.unique(samples.phases).tolist()):
            phase_mask = samples.phases == phase_label
            phases.append(
                {
                    "label": int(phase_label),
                    "name": phase_names.get(int(phase_label), ""),
                    **_zeroing_summary(
                        magnitudes=samples.magnitudes[phase_mask],
                        threshold=threshold,
                    ),
                }
            )
    return {
        "name": samples.group.name,
        "columns": list(samples.group.columns),
        "override_path": samples.group.override_path,
        "percentile": percentile,
        "threshold": threshold,
        "histogram": {
            "bin_edges": [float(edge) for edge in bin_edges],
            "raw_counts": [int(count) for count in raw_counts],
            "denoised_counts": [int(count) for count in denoised_counts],
            "log10_threshold": float(np.log10(threshold)),
        },
        "overall": _zeroing_summary(
            magnitudes=samples.magnitudes,
            threshold=threshold,
        ),
        "phases": phases,
    }


def _zeroing_summary(
    magnitudes: NDArray[np.float64],
    threshold: float,
) -> dict[str, JsonValue]:
    """Count existing and newly suppressed zero actions."""
    sample_count = len(magnitudes)
    existing_zero = magnitudes == 0
    zero_after = magnitudes < threshold
    newly_zeroed = zero_after & ~existing_zero
    zeroed_count = int(zero_after.sum())
    return {
        "sample_count": sample_count,
        "existing_zero_count": int(existing_zero.sum()),
        "newly_zeroed_count": int(newly_zeroed.sum()),
        "zeroed_count": zeroed_count,
        "zeroed_percent": (
            100.0 * zeroed_count / sample_count if sample_count else 0.0
        ),
    }
