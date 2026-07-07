"""Dataset schema describing recorded features and their episode columns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from pydantic import ConfigDict

from tso_sensorium.configuration import ConfigModel


class CameraFeature(ConfigModel):
    """Camera stream of a dataset.

    Args:
        name: Camera name used in exported feature keys.
        frame_column: Episode table column holding the frame image path.
        height: Frame height in pixels.
        width: Frame width in pixels.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    frame_column: str
    height: int
    width: int


class ArmFeature(ConfigModel):
    """Robot arm of a dataset.

    Args:
        name: Arm name.
        state_columns: Episode table columns holding the arm's
            proprioceptive state.
        action_columns: Episode table columns holding the arm's actions.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    state_columns: list[str]
    action_columns: list[str]


class DatasetSchema(ConfigModel):
    """Describes dataset features and where they live in episode tables.

    The same schema drives episode assembly and export: cameras and arms
    declare which episode table columns hold their data, and single-arm
    versus multi-arm datasets differ only in the number of arm entries.

    Args:
        name: Dataset name.
        fps: Acquisition frame rate of aligned episodes.
        cameras: Camera streams included in the dataset.
        arms: Robot arms included in the dataset.
        task: Language description of the task, used as the default when an
            episode does not define its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fps: int
    cameras: list[CameraFeature]
    arms: list[ArmFeature]
    task: Optional[str] = None

    @property
    def state_columns(self) -> list[str]:
        """Proprioceptive state columns across all arms, in arm order."""
        return [column for arm in self.arms for column in arm.state_columns]

    @property
    def action_columns(self) -> list[str]:
        """Action columns across all arms, in arm order."""
        return [column for arm in self.arms for column in arm.action_columns]

    def validate_episode_table(self, table: pd.DataFrame) -> None:
        """Raise if the table is missing any column the schema references.

        Args:
            table: Aligned episode table to check.
        """
        required_columns = (
            self.state_columns
            + self.action_columns
            + [camera.frame_column for camera in self.cameras]
        )
        missing_columns = [
            column for column in required_columns if column not in table.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Episode table is missing schema columns: {missing_columns}"
            )


@dataclass(frozen=True)
class Episode:
    """A single aligned episode.

    Args:
        name: Episode identifier, used as the folder or episode name on
            export.
        table: Aligned per-timestep values, one row per frame.
        task: Episode-specific task description; falls back to the schema
            task when omitted.
    """

    name: str
    table: pd.DataFrame
    task: Optional[str] = None
