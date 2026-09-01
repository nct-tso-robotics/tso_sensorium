"""Dataset schema describing recorded features and their episode columns."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

import pandas as pd
from pydantic import ConfigDict, Field, model_validator

from tso_sensorium.configuration import ConfigModel


class FrameTemporality(str, Enum):
    """Whether a coordinate frame has a common basis across timesteps."""

    FIXED = "fixed"
    MOVING = "moving"
    UNKNOWN = "unknown"


class CoordinateFrameFeatureMetadata(ConfigModel):
    """Coordinate-frame metadata for a group of vector component columns.

    Args:
        columns: Ordered component columns forming the vector feature.
        frame: Dataset-defined name of the coordinate frame.
        frame_temporality: Whether the frame basis is fixed across time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: list[str]
    frame: str
    frame_temporality: FrameTemporality

    @model_validator(mode="after")
    def validate_fields(self) -> "CoordinateFrameFeatureMetadata":
        """Validate that the feature identifies columns and a frame."""
        if not self.columns:
            raise ValueError("Coordinate-frame feature columns cannot be empty")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                f"Coordinate-frame feature columns must be unique, got {self.columns}"
            )
        if not self.frame:
            raise ValueError("Coordinate-frame feature frame cannot be empty")
        return self


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


class AuxiliaryFeature(ConfigModel):
    """Additional tabular feature exported alongside state and action.

    Args:
        columns: Episode table columns forming the feature vector.
        dtype: Numeric dtype used by the destination dataset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    columns: list[str]
    dtype: Literal[
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "bool",
    ]

    @model_validator(mode="after")
    def validate_columns(self) -> "AuxiliaryFeature":
        """Validate that the feature contains unique columns."""
        if not self.columns:
            raise ValueError("Auxiliary feature columns cannot be empty")
        if len(self.columns) != len(set(self.columns)):
            raise ValueError(
                f"Auxiliary feature columns must be unique, got {self.columns}"
            )
        return self


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
        coordinate_frame_features: Vector component groups and their coordinate-frame
            temporality.
        auxiliary_features: Additional named numeric tabular features.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    fps: int
    cameras: list[CameraFeature]
    arms: list[ArmFeature]
    task: Optional[str] = None
    coordinate_frame_features: dict[str, CoordinateFrameFeatureMetadata] = Field(
        default_factory=dict
    )
    auxiliary_features: dict[str, AuxiliaryFeature] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_coordinate_frame_feature_columns(self) -> "DatasetSchema":
        """Ensure frame metadata references exported state or action columns."""
        exported_columns = set(self.state_columns + self.action_columns)
        for name, feature in self.coordinate_frame_features.items():
            missing_columns = [
                column for column in feature.columns if column not in exported_columns
            ]
            if missing_columns:
                raise ValueError(
                    f"Coordinate-frame feature '{name}' references columns outside "
                    "the schema: "
                    f"{missing_columns}"
                )
        return self

    @model_validator(mode="after")
    def validate_auxiliary_feature_names(self) -> "DatasetSchema":
        """Ensure auxiliary feature names do not collide with core features."""
        reserved_names = {"observation.state", "action", "task"}
        invalid_names = sorted(reserved_names & set(self.auxiliary_features))
        invalid_names.extend(
            sorted(
                name
                for name in self.auxiliary_features
                if name.startswith("observation.images.")
            )
        )
        if invalid_names:
            raise ValueError(
                f"Auxiliary feature names collide with core features: {invalid_names}"
            )
        return self

    @property
    def state_columns(self) -> list[str]:
        """Proprioceptive state columns across all arms, in arm order."""
        return [column for arm in self.arms for column in arm.state_columns]

    @property
    def action_columns(self) -> list[str]:
        """Action columns across all arms, in arm order."""
        return [column for arm in self.arms for column in arm.action_columns]

    @property
    def auxiliary_columns(self) -> list[str]:
        """Auxiliary columns across all named features."""
        return [
            column
            for feature in self.auxiliary_features.values()
            for column in feature.columns
        ]

    def validate_episode_table(self, table: pd.DataFrame) -> None:
        """Raise if the table is missing any column the schema references.

        Args:
            table: Aligned episode table to check.
        """
        required_columns = (
            self.state_columns
            + self.action_columns
            + self.auxiliary_columns
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
