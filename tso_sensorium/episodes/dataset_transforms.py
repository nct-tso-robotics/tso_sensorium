"""Configurable transforms that use statistics from a complete dataset."""

from __future__ import annotations

import abc
from typing import Dict, List, Literal, Union

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from tso_sensorium.configuration import ConfigModel
from tso_sensorium.episodes.schema import Episode

JsonValue = Union[
    None,
    bool,
    int,
    float,
    str,
    List["JsonValue"],
    Dict[str, "JsonValue"],
]


class DatasetTransform(ConfigModel, abc.ABC):
    """A transformation computed over every assembled episode."""

    @abc.abstractmethod
    def apply(self, episodes: list[Episode]) -> list[Episode]:
        """Transform assembled episodes using dataset-level statistics.

        Args:
            episodes: Successfully assembled episodes.

        Returns:
            Transformed episodes in the same order.
        """

    @abc.abstractmethod
    def metadata_payload(self) -> dict[str, JsonValue]:
        """Describe the applied transform and its computed statistics.

        Returns:
            JSON-compatible transform provenance.

        Raises:
            RuntimeError: If called before the transform has been applied.
        """


class PercentileDenoisingColumnGroup(ConfigModel):
    """One vector whose small nonzero magnitudes should be suppressed.

    Args:
        columns: Component columns forming the vector.
        percentile: Percentile of nonzero L2 magnitudes used as the threshold.
    """

    columns: List[str] = Field(default_factory=list)
    percentile: float = 15.0

    @model_validator(mode="after")
    def validate_configuration(self) -> "PercentileDenoisingColumnGroup":
        """Validate component columns and percentile bounds."""
        if not self.columns:
            raise ValueError("Denoising columns cannot be empty")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(f"Denoising columns must be unique, got {self.columns}")
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError(
                f"Denoising percentile must be in [0, 100], got {self.percentile}"
            )
        return self


class PercentileDenoiseColumns(DatasetTransform):
    """Zero vector rows below global nonzero-magnitude percentiles.

    A separate threshold is computed for every named group by concatenating
    its L2 magnitudes across all successfully assembled episodes. Exact-zero
    rows are excluded from the percentile calculation.

    Args:
        column_groups: Named vector groups with independently configurable
            columns and percentiles.
    """

    type: Literal["percentile_denoise_columns"] = "percentile_denoise_columns"
    column_groups: Dict[str, PercentileDenoisingColumnGroup] = Field(
        default_factory=dict
    )
    _computed_thresholds: dict[str, float] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_column_groups(self) -> "PercentileDenoiseColumns":
        """Require at least one named column group."""
        if not self.column_groups:
            raise ValueError("Denoising column_groups cannot be empty")
        return self

    def apply(self, episodes: list[Episode]) -> list[Episode]:
        """Compute global thresholds and zero movements below them.

        Args:
            episodes: Successfully assembled episodes.

        Returns:
            Copies of the episodes with denoised column groups.

        Raises:
            ValueError: If a configured column is missing or a group has no
                nonzero movement in the dataset.
        """
        for episode in episodes:
            for group_name, group in self.column_groups.items():
                missing_columns = [
                    column for column in group.columns if column not in episode.table
                ]
                if missing_columns:
                    raise ValueError(
                        f"Episode '{episode.name}' is missing denoising columns for "
                        f"group '{group_name}': {missing_columns}"
                    )

        computed_thresholds: dict[str, float] = {}
        for group_name, group in self.column_groups.items():
            episode_values = [
                episode.table[group.columns].to_numpy(dtype=float)
                for episode in episodes
            ]
            values = (
                np.concatenate(episode_values, axis=0)
                if episode_values
                else np.empty(shape=(0, len(group.columns)), dtype=float)
            )
            magnitudes = np.linalg.norm(x=values, axis=1)
            nonzero_magnitudes = magnitudes[magnitudes > 0]
            if len(nonzero_magnitudes) == 0:
                raise ValueError(
                    f"Cannot compute denoising threshold for group '{group_name}': "
                    "the dataset contains no nonzero movement"
                )
            computed_thresholds[group_name] = float(
                np.percentile(a=nonzero_magnitudes, q=group.percentile)
            )

        transformed_episodes = [
            Episode(
                name=episode.name,
                table=episode.table.copy(),
                task=episode.task,
            )
            for episode in episodes
        ]
        for episode in transformed_episodes:
            for group_name, group in self.column_groups.items():
                values = episode.table[group.columns].to_numpy(dtype=float)
                magnitudes = np.linalg.norm(x=values, axis=1)
                below_threshold = magnitudes < computed_thresholds[group_name]
                episode.table.loc[below_threshold, group.columns] = 0.0

        self._computed_thresholds = computed_thresholds
        return transformed_episodes

    def metadata_payload(self) -> dict[str, JsonValue]:
        """Describe configured groups and their computed thresholds.

        Returns:
            JSON-compatible transform provenance.

        Raises:
            RuntimeError: If the transform has not been applied.
        """
        if set(self._computed_thresholds) != set(self.column_groups):
            raise RuntimeError(
                "Percentile denoising metadata is unavailable before apply()"
            )
        return {
            "type": self.type,
            "column_groups": {
                name: {
                    "columns": group.columns,
                    "percentile": group.percentile,
                    "threshold": self._computed_thresholds[name],
                }
                for name, group in sorted(self.column_groups.items())
            },
        }


AnyDatasetTransform = PercentileDenoiseColumns
