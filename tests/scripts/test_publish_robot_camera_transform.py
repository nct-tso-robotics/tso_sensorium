"""Tests for tso_sensorium.scripts.publish_robot_camera_transform module."""

import pytest

pytest.importorskip("rospy")

from tso_sensorium.scripts.publish_robot_camera_transform import (  # noqa: E402
    parse_arguments,
)


@pytest.mark.unit
def test_subscribed_cli_preserves_toolkit_command_shape() -> None:
    arguments = parse_arguments(
        arguments=[
            "--subscribe",
            "/robot_state",
            "--topic",
            "/robot_camera_transform",
            "--calibration_path",
            "calibration.yaml",
        ]
    )

    assert arguments.subscribe == "/robot_state"
    assert arguments.euler is None
    assert arguments.topic == "/robot_camera_transform"
    assert arguments.calibration_path == "calibration.yaml"
