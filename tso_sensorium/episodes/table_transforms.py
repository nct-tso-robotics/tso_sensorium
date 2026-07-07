"""Configurable transforms over aligned episode tables.

Each transform is selected by its ``type`` key in YAML and applied to the
episode table after alignment. This module requires Python 3.9+; the rest
of the episodes package stays importable without it.
"""

from __future__ import annotations

import abc
from typing import Annotated, Dict, List, Literal, Union

import numpy as np
import pandas as pd
from pydantic import Field
from scipy.spatial.transform import Rotation

from tso_sensorium.configuration import ConfigModel

VECTOR3_PREFIX_LENGTH = 3


def parse_vector3_string(vector_string: str) -> list[float]:
    """Parse a stringified ROS Vector3 message into its three floats.

    The recorded format has one axis per line, each as ``axis: value``.

    Args:
        vector_string: Stringified message as stored in the recording CSV.

    Returns:
        The [x, y, z] values.
    """
    lines = vector_string.split("\n")
    return [float(line[VECTOR3_PREFIX_LENGTH:]) for line in lines[:3]]


def rotate_point_with_quaternion(
    quaternion: list[float], point: list[float]
) -> np.ndarray:
    """Rotate a point by a quaternion given in x, y, z, w order.

    Args:
        quaternion: Rotation as [x, y, z, w].
        point: Point as [x, y, z].

    Returns:
        Rotated point, (3,).
    """
    x, y, z, w = quaternion
    rotation_matrix = np.array(
        [
            [1 - 2 * y**2 - 2 * z**2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x**2 - 2 * z**2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x**2 - 2 * y**2],
        ]
    )
    return rotation_matrix @ np.array(point)


class TableTransform(ConfigModel, abc.ABC):
    """A single transformation step over an aligned episode table."""

    @abc.abstractmethod
    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        """Return the transformed table."""


class ParseVector3Columns(TableTransform):
    """Split a stringified Vector3 column into three float columns.

    Args:
        column: Column holding the stringified messages.
        output_columns: Names of the three output columns, in x, y, z
            order.
    """

    type: Literal["parse_vector3"] = "parse_vector3"
    column: str = ""
    output_columns: List[str] = Field(default_factory=list)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        parsed = table[self.column].apply(parse_vector3_string)
        table[self.output_columns] = pd.DataFrame(parsed.tolist(), index=table.index)
        return table


class SumColumns(TableTransform):
    """Add two column groups elementwise into output columns.

    Args:
        first_columns: First operand columns.
        second_columns: Second operand columns, same length.
        output_columns: Names of the sums, same length.
    """

    type: Literal["sum_columns"] = "sum_columns"
    first_columns: List[str] = Field(default_factory=list)
    second_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        for first, second, output in zip(
            self.first_columns, self.second_columns, self.output_columns
        ):
            table[output] = table[first] + table[second]
        return table


class FixedTransformToCameraFrame(TableTransform):
    """Convert positions to the camera frame with a fixed calibrated transform.

    The camera-to-base homogeneous transform is inverted once; positions
    are mapped through it, and the resulting constant quaternion and
    translation are added as columns.

    Args:
        camera_to_base: 4x4 homogeneous transform, camera frame to base.
        position_columns: Base-frame position columns, x, y, z order.
        output_columns: Camera-frame position column names.
        quaternion_output_columns: Constant quaternion column names, in
            x, y, z, w order.
        translation_output_columns: Constant translation column names.
    """

    type: Literal["fixed_transform_to_camera_frame"] = "fixed_transform_to_camera_frame"
    camera_to_base: List[List[float]] = Field(default_factory=list)
    position_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)
    quaternion_output_columns: List[str] = Field(default_factory=list)
    translation_output_columns: List[str] = Field(default_factory=list)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        base_to_camera = np.linalg.inv(np.array(self.camera_to_base))
        rotation = base_to_camera[:3, :3]
        translation = base_to_camera[:3, 3]
        positions = table[self.position_columns].to_numpy()  # (N, 3)
        camera_positions = positions @ rotation.T + translation
        table[self.output_columns] = camera_positions
        quaternion = Rotation.from_matrix(rotation).as_quat()
        for column, value in zip(self.quaternion_output_columns, quaternion):
            table[column] = value
        for column, value in zip(self.translation_output_columns, translation):
            table[column] = value
        return table


class RotateByQuaternionColumns(TableTransform):
    """Rotate positions by a per-row quaternion stored in the table.

    Args:
        point_columns: Position columns, x, y, z order.
        quaternion_columns: Quaternion columns, x, y, z, w order.
        output_columns: Rotated position column names.
    """

    type: Literal["rotate_by_quaternion_columns"] = "rotate_by_quaternion_columns"
    point_columns: List[str] = Field(default_factory=list)
    quaternion_columns: List[str] = Field(default_factory=list)
    output_columns: List[str] = Field(default_factory=list)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        rotated = [
            rotate_point_with_quaternion(
                quaternion=list(row[self.quaternion_columns]),
                point=list(row[self.point_columns]),
            )
            for _, row in table.iterrows()
        ]
        table[self.output_columns] = np.array(rotated).reshape(len(table), 3)
        return table


class AddConstantColumns(TableTransform):
    """Add columns holding one constant value each.

    Args:
        values: Mapping of column name to constant value.
    """

    type: Literal["add_constant_columns"] = "add_constant_columns"
    values: Dict[str, float] = Field(default_factory=dict)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        for column, value in self.values.items():
            table[column] = value
        return table


class RenameColumns(TableTransform):
    """Rename columns.

    Args:
        mapping: Mapping of old column name to new column name.
    """

    type: Literal["rename_columns"] = "rename_columns"
    mapping: Dict[str, str] = Field(default_factory=dict)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.rename(columns=self.mapping)


class DropColumns(TableTransform):
    """Remove columns from the table.

    Args:
        columns: Columns to drop.
    """

    type: Literal["drop_columns"] = "drop_columns"
    columns: List[str] = Field(default_factory=list)

    def apply(self, table: pd.DataFrame) -> pd.DataFrame:
        return table.drop(columns=self.columns)


AnyTableTransform = Annotated[
    Union[
        ParseVector3Columns,
        SumColumns,
        FixedTransformToCameraFrame,
        RotateByQuaternionColumns,
        AddConstantColumns,
        RenameColumns,
        DropColumns,
    ],
    Field(discriminator="type"),
]
