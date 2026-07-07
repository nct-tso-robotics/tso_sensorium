"""Tests for tso_sensorium.episodes.table_transforms module."""

from pydantic import TypeAdapter
import numpy as np
import pandas as pd
import pytest

from tso_sensorium.episodes.table_transforms import (
    AddConstantColumns,
    AnyTableTransform,
    DropColumns,
    FixedTransformToCameraFrame,
    ParseVector3Columns,
    RenameColumns,
    RotateByQuaternionColumns,
    SumColumns,
    parse_vector3_string,
    rotate_point_with_quaternion,
)

IDENTITY_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


@pytest.fixture
def position_table_factory():
    def factory(length=2) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x": [0.1 * index for index in range(length)],
                "y": [0.2 * index for index in range(length)],
                "z": [0.3 * index for index in range(length)],
            }
        )

    return factory


class TestParsingHelpers:
    @pytest.mark.unit
    def test_parse_vector3_string_extracts_floats(self):
        assert parse_vector3_string("x: 1.5\ny: -2.0\nz: 0.25") == [
            1.5,
            -2.0,
            0.25,
        ]

    @pytest.mark.unit
    def test_identity_quaternion_leaves_point_unchanged(self):
        rotated = rotate_point_with_quaternion(
            quaternion=[0.0, 0.0, 0.0, 1.0], point=[1.0, 2.0, 3.0]
        )
        np.testing.assert_allclose(rotated, [1.0, 2.0, 3.0])


class TestParseVector3Columns:
    @pytest.mark.unit
    def test_splits_string_column_into_floats(self):
        table = pd.DataFrame(
            {"vector": ["x: 1.0\ny: 2.0\nz: 3.0", "x: 4.0\ny: 5.0\nz: 6.0"]}
        )
        transform = ParseVector3Columns(
            column="vector", output_columns=["vx", "vy", "vz"]
        )
        result = transform.apply(table=table)
        assert result["vx"].tolist() == [1.0, 4.0]
        assert result["vz"].tolist() == [3.0, 6.0]


class TestSumColumns:
    @pytest.mark.unit
    def test_adds_column_pairs(self, position_table_factory):
        table = position_table_factory(length=2)
        table[["ox", "oy", "oz"]] = 1.0
        transform = SumColumns(
            first_columns=["x", "y", "z"],
            second_columns=["ox", "oy", "oz"],
            output_columns=["sx", "sy", "sz"],
        )
        result = transform.apply(table=table)
        np.testing.assert_allclose(result["sx"].tolist(), [1.0, 1.1])
        np.testing.assert_allclose(result["sz"].tolist(), [1.0, 1.3])


class TestFixedTransformToCameraFrame:
    @pytest.mark.unit
    def test_identity_transform_copies_positions_and_adds_constants(
        self, position_table_factory
    ):
        table = position_table_factory(length=2)
        transform = FixedTransformToCameraFrame(
            camera_to_base=IDENTITY_TRANSFORM,
            position_columns=["x", "y", "z"],
            output_columns=["cx", "cy", "cz"],
            quaternion_output_columns=["qx", "qy", "qz", "qw"],
            translation_output_columns=["tx", "ty", "tz"],
        )
        result = transform.apply(table=table)
        np.testing.assert_allclose(result["cx"], result["x"])
        np.testing.assert_allclose(result["qw"].tolist(), [1.0, 1.0])
        np.testing.assert_allclose(result["qx"].tolist(), [0.0, 0.0])
        np.testing.assert_allclose(result["tx"].tolist(), [0.0, 0.0])

    @pytest.mark.unit
    def test_translation_shifts_positions(self, position_table_factory):
        table = position_table_factory(length=1)
        camera_to_base = [
            [1.0, 0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, -3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        transform = FixedTransformToCameraFrame(
            camera_to_base=camera_to_base,
            position_columns=["x", "y", "z"],
            output_columns=["cx", "cy", "cz"],
            quaternion_output_columns=["qx", "qy", "qz", "qw"],
            translation_output_columns=["tx", "ty", "tz"],
        )
        result = transform.apply(table=table)
        # base_to_camera is the inverse, so positions shift by +(1, 2, 3).
        np.testing.assert_allclose(
            result[["cx", "cy", "cz"]].iloc[0].tolist(), [1.0, 2.0, 3.0]
        )


class TestRotateByQuaternionColumns:
    @pytest.mark.unit
    def test_identity_quaternion_copies_positions(self, position_table_factory):
        table = position_table_factory(length=2)
        table[["qx", "qy", "qz"]] = 0.0
        table["qw"] = 1.0
        transform = RotateByQuaternionColumns(
            point_columns=["x", "y", "z"],
            quaternion_columns=["qx", "qy", "qz", "qw"],
            output_columns=["cx", "cy", "cz"],
        )
        result = transform.apply(table=table)
        np.testing.assert_allclose(result["cx"], result["x"])
        np.testing.assert_allclose(result["cy"], result["y"])


class TestSimpleColumnOperations:
    @pytest.mark.unit
    def test_add_constant_columns(self, position_table_factory):
        table = position_table_factory(length=2)
        result = AddConstantColumns(values={"phase": 1.0}).apply(table=table)
        assert result["phase"].tolist() == [1.0, 1.0]

    @pytest.mark.unit
    def test_rename_and_drop_columns(self, position_table_factory):
        table = position_table_factory(length=1)
        result = RenameColumns(mapping={"x": "position_x"}).apply(table=table)
        result = DropColumns(columns=["y"]).apply(table=result)
        assert "position_x" in result.columns
        assert "x" not in result.columns
        assert "y" not in result.columns


@pytest.mark.unit
def test_choice_registry_decodes_by_type_key():
    transform = TypeAdapter(AnyTableTransform).validate_python(
        {"type": "drop_columns", "columns": ["a", "b"]}
    )
    assert transform == DropColumns(columns=["a", "b"])


@pytest.mark.unit
def test_rotate_point_with_quaternion_applies_forward_rotation():
    # +90 degrees about z maps (1,0,0) -> (0,1,0).
    result = rotate_point_with_quaternion(
        quaternion=[0.0, 0.0, 0.7071067811865476, 0.7071067811865476],
        point=[1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)
