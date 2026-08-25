"""ROS 1 publisher for robot-to-camera transforms."""

from __future__ import annotations

import time
from typing import Protocol, cast

from geometry_msgs.msg import TransformStamped
import rospy
import rostopic

from tso_sensorium.processing.camera_transform import (
    CameraMountCalibration,
    camera_quaternion_from_end_effector,
)

TOPIC_DISCOVERY_POLL_SECONDS = 0.1


class QuaternionMessage(Protocol):
    """Quaternion fields required from a ROS message."""

    x: float
    y: float
    z: float
    w: float


class PoseMessage(Protocol):
    """Pose fields required from a ROS message."""

    orientation: QuaternionMessage


class RobotStateMessage(Protocol):
    """Robot-state fields required by the transform publisher."""

    robot_pose: PoseMessage


def discover_message_type(
    *, topic_name: str, timeout_seconds: float
) -> type[RobotStateMessage]:
    """Discover a live topic's message class within a bounded time.

    Args:
        topic_name: ROS topic whose type should be discovered.
        timeout_seconds: Maximum discovery duration.

    Returns:
        Message class published on the topic.

    Raises:
        RuntimeError: If the topic type is unavailable before the deadline.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline and not rospy.is_shutdown():
        message_type, _, _ = rostopic.get_topic_class(topic_name, blocking=False)
        if message_type is not None:
            return cast(type[RobotStateMessage], message_type)
        rospy.sleep(TOPIC_DISCOVERY_POLL_SECONDS)
    raise RuntimeError(
        f"No message type found for {topic_name} within {timeout_seconds:g} seconds"
    )


class CameraTransformPublisher:
    """Publish static or live robot-to-camera rotations."""

    def __init__(
        self,
        *,
        topic_name: str,
        parent_frame: str,
        child_frame: str,
        translation: tuple[float, float, float],
        rate_hz: float,
        calibration: CameraMountCalibration,
    ) -> None:
        self.topic_name = topic_name
        self.rate_hz = rate_hz
        self.calibration = calibration
        self.pose_received = False
        self.publisher = rospy.Publisher(
            topic_name,
            TransformStamped,
            queue_size=1,
        )
        self.transform = TransformStamped()
        self.transform.header.frame_id = parent_frame
        self.transform.child_frame_id = child_frame
        self.transform.transform.translation.x = translation[0]
        self.transform.transform.translation.y = translation[1]
        self.transform.transform.translation.z = translation[2]

    def set_quaternion(self, *, quaternion: tuple[float, float, float, float]) -> None:
        """Update the published rotation in xyzw order."""
        rotation = self.transform.transform.rotation
        rotation.x, rotation.y, rotation.z, rotation.w = quaternion
        self.pose_received = True

    def robot_state_callback(self, message: RobotStateMessage) -> None:
        """Calibrate and store an incoming end-effector orientation."""
        orientation = message.robot_pose.orientation
        quaternion = camera_quaternion_from_end_effector(
            quaternion=(orientation.x, orientation.y, orientation.z, orientation.w),
            calibration=self.calibration,
        )
        self.set_quaternion(quaternion=tuple(float(value) for value in quaternion))

    def publish(self) -> None:
        """Publish transforms at the configured rate until ROS shuts down."""
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            if self.pose_received:
                self.transform.header.stamp = rospy.Time.now()
                self.publisher.publish(self.transform)
            rate.sleep()

    def run_static(self, *, quaternion: tuple[float, float, float, float]) -> None:
        """Continuously publish a fixed rotation."""
        self.set_quaternion(quaternion=quaternion)
        rospy.loginfo("Publishing static transform to %s", self.topic_name)
        self.publish()

    def run_subscribed(
        self,
        *,
        subscribe_topic: str,
        message_type: type[RobotStateMessage],
    ) -> None:
        """Republish calibrated orientations from a robot-state topic."""
        rospy.Subscriber(
            subscribe_topic,
            message_type,
            self.robot_state_callback,
            queue_size=1,
        )
        rospy.loginfo("Subscribed to %s", subscribe_topic)
        rospy.loginfo("Publishing live transform to %s", self.topic_name)
        self.publish()
