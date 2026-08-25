"""Publish a static or live robot-to-camera transform on ROS 1."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import rospy

from tso_sensorium.processing.camera_transform import (
    camera_quaternion_from_euler,
    load_camera_mount_calibration,
)
from tso_sensorium.recording.ros1.camera_transform_publisher import (
    CameraTransformPublisher,
    discover_message_type,
)

DEFAULT_TOPIC = "/robot_camera_transform"
DEFAULT_PARENT_FRAME = "robot_base"
DEFAULT_CHILD_FRAME = "camera"
DEFAULT_RATE_HZ = 10.0
DEFAULT_DISCOVERY_TIMEOUT_SECONDS = 5.0
ROS_NODE_NAME = "robot_camera_transform_publisher"


def parse_vector3(value: str) -> tuple[float, float, float]:
    """Parse three comma-separated floating-point values."""
    values = value.replace(" ", "").split(",")
    if len(values) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected three comma-separated values, got {value}"
        )
    return float(values[0]), float(values[1]), float(values[2])


def parse_arguments(*, arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse transform publisher command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Publish a static or live robot-to-camera transform"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--euler",
        "-e",
        type=parse_vector3,
        help="Static roll,pitch,yaw angles in radians",
    )
    source.add_argument(
        "--subscribe",
        "-s",
        metavar="TOPIC",
        help="Topic containing a robot_pose.orientation field",
    )
    parser.add_argument("--calibration_path", required=True)
    parser.add_argument("--topic", "-t", default=DEFAULT_TOPIC)
    parser.add_argument("--parent", "--parent-frame", default=DEFAULT_PARENT_FRAME)
    parser.add_argument("--child", "--child-frame", default=DEFAULT_CHILD_FRAME)
    parser.add_argument(
        "--translation",
        "--trans",
        type=parse_vector3,
        default=(0.0, 0.0, 0.0),
    )
    parser.add_argument("--rate", "-r", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument(
        "--discovery-timeout",
        type=float,
        default=DEFAULT_DISCOVERY_TIMEOUT_SECONDS,
    )
    return parser.parse_args(args=arguments)


def run(*, arguments: argparse.Namespace) -> None:
    """Run the configured transform publisher."""
    if arguments.rate <= 0:
        raise ValueError(f"rate must be positive, got {arguments.rate}")
    if arguments.discovery_timeout <= 0:
        raise ValueError(
            f"discovery_timeout must be positive, got {arguments.discovery_timeout}"
        )
    calibration = load_camera_mount_calibration(
        calibration_path=arguments.calibration_path
    )
    rospy.init_node(ROS_NODE_NAME, anonymous=True)
    publisher = CameraTransformPublisher(
        topic_name=arguments.topic,
        parent_frame=arguments.parent,
        child_frame=arguments.child,
        translation=arguments.translation,
        rate_hz=arguments.rate,
        calibration=calibration,
    )
    if arguments.euler is not None:
        roll, pitch, yaw = arguments.euler
        quaternion = camera_quaternion_from_euler(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            calibration=calibration,
        )
        publisher.run_static(quaternion=tuple(float(value) for value in quaternion))
        return
    message_type = discover_message_type(
        topic_name=arguments.subscribe,
        timeout_seconds=arguments.discovery_timeout,
    )
    publisher.run_subscribed(
        subscribe_topic=arguments.subscribe,
        message_type=message_type,
    )


def main() -> None:
    """Parse command-line arguments and publish transforms."""
    run(arguments=parse_arguments())


if __name__ == "__main__":
    main()
