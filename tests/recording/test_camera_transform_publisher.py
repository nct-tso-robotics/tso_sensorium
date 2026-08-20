"""Tests for tso_sensorium.recording.ros1.camera_transform_publisher module."""

import re
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("rospy")

from tso_sensorium.processing.camera_transform import (  # noqa: E402
    CameraMountCalibration,
)
from tso_sensorium.recording.ros1.camera_transform_publisher import (  # noqa: E402
    CameraTransformPublisher,
    discover_message_type,
)

CAMERA_QUATERNION_PATH = (
    "tso_sensorium.recording.ros1.camera_transform_publisher."
    "camera_quaternion_from_end_effector"
)
PUBLISHER_PATH = (
    "tso_sensorium.recording.ros1.camera_transform_publisher.rospy.Publisher"
)
SUBSCRIBER_PATH = (
    "tso_sensorium.recording.ros1.camera_transform_publisher.rospy.Subscriber"
)
TOPIC_CLASS_PATH = (
    "tso_sensorium.recording.ros1.camera_transform_publisher.rostopic.get_topic_class"
)
MONOTONIC_PATH = (
    "tso_sensorium.recording.ros1.camera_transform_publisher.time.monotonic"
)
IS_SHUTDOWN_PATH = (
    "tso_sensorium.recording.ros1.camera_transform_publisher.rospy.is_shutdown"
)


@pytest.fixture
def camera_mount_calibration() -> CameraMountCalibration:
    return CameraMountCalibration(
        aligned_camera_from_end_effector=np.eye(3),
        optical_tilt_radians=0.0,
    )


@pytest.fixture
def camera_transform_publisher_factory(camera_mount_calibration):
    def factory() -> CameraTransformPublisher:
        with patch(PUBLISHER_PATH):
            return CameraTransformPublisher(
                topic_name="/output",
                parent_frame="base",
                child_frame="camera",
                translation=(1.0, 2.0, 3.0),
                rate_hz=20.0,
                calibration=camera_mount_calibration,
            )

    return factory


@pytest.mark.unit
def test_callback_calibrates_end_effector_orientation(
    camera_transform_publisher_factory,
    camera_mount_calibration,
) -> None:
    publisher = camera_transform_publisher_factory()
    message = MagicMock()
    message.robot_pose.orientation.x = 0.1
    message.robot_pose.orientation.y = 0.2
    message.robot_pose.orientation.z = 0.3
    message.robot_pose.orientation.w = 0.9
    calibrated_quaternion = np.array([0.4, 0.5, 0.6, 0.7])

    with patch(CAMERA_QUATERNION_PATH, return_value=calibrated_quaternion) as compute:
        publisher.robot_state_callback(message=message)

    compute.assert_called_once_with(
        quaternion=(0.1, 0.2, 0.3, 0.9),
        calibration=camera_mount_calibration,
    )
    rotation = publisher.transform.transform.rotation
    assert (rotation.x, rotation.y, rotation.z, rotation.w) == (0.4, 0.5, 0.6, 0.7)
    assert publisher.pose_received


@pytest.mark.unit
def test_subscribed_mode_registers_callback_before_publishing(
    camera_transform_publisher_factory,
) -> None:
    publisher = camera_transform_publisher_factory()
    message_type = MagicMock()
    with (
        patch(SUBSCRIBER_PATH) as subscriber,
        patch.object(publisher, "publish") as publish,
    ):
        publisher.run_subscribed(
            subscribe_topic="/robot_state",
            message_type=message_type,
        )

    subscriber.assert_called_once_with(
        "/robot_state",
        message_type,
        publisher.robot_state_callback,
        queue_size=1,
    )
    publish.assert_called_once_with()


@pytest.mark.unit
def test_discovers_live_topic_message_type() -> None:
    message_type = MagicMock()
    with (
        patch(MONOTONIC_PATH, side_effect=[0.0, 0.0]),
        patch(IS_SHUTDOWN_PATH, return_value=False),
        patch(TOPIC_CLASS_PATH, return_value=(message_type, "/robot_state", None)),
    ):
        discovered = discover_message_type(
            topic_name="/robot_state",
            timeout_seconds=5.0,
        )

    assert discovered == message_type


@pytest.mark.unit
def test_topic_discovery_has_bounded_timeout() -> None:
    with (
        patch(MONOTONIC_PATH, side_effect=[0.0, 2.0]),
        patch(IS_SHUTDOWN_PATH, return_value=False),
        pytest.raises(
            RuntimeError,
            match=re.escape("No message type found for /missing within 1 seconds"),
        ),
    ):
        discover_message_type(topic_name="/missing", timeout_seconds=1.0)
