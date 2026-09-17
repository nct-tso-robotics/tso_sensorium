"""Synthetic ROS 2 sensors for trying the recording stack without hardware.

Publishes a moving test-pattern camera, a circular tool pose, and a
toggling gripper, matching ``configs/recording/mock_ui.yaml``:

    python -m tso_sensorium.scripts.mock_sensors_ros2
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool

from tso_sensorium.scripts.mock_scene import (
    CAMERA_INFO_TOPIC,
    CAMERA_TOPIC,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GRIPPER_TOPIC,
    POSE_TOPIC,
    PUBLISH_RATE_HZ,
    gripper_open,
    render_frame,
    tool_position,
)

QUEUE_DEPTH = 1


def main() -> None:
    """Publish the mock sensors until interrupted."""
    rclpy.init()
    node = rclpy.create_node("mock_sensors")
    camera = node.create_publisher(Image, CAMERA_TOPIC, QUEUE_DEPTH)
    camera_info = node.create_publisher(CameraInfo, CAMERA_INFO_TOPIC, QUEUE_DEPTH)
    pose = node.create_publisher(PoseStamped, POSE_TOPIC, QUEUE_DEPTH)
    gripper = node.create_publisher(Bool, GRIPPER_TOPIC, QUEUE_DEPTH)

    base_y, base_x = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    state = {"tick": 0}

    def publish_once() -> None:
        tick = state["tick"]
        now = node.get_clock().now().to_msg()
        frame = render_frame(tick=tick, base_x=base_x, base_y=base_y)
        image_message = Image()
        image_message.height = FRAME_HEIGHT
        image_message.width = FRAME_WIDTH
        image_message.encoding = "bgr8"
        image_message.step = FRAME_WIDTH * 3
        image_message.data = frame.tobytes()
        image_message.header.stamp = now
        camera.publish(image_message)
        camera_info.publish(
            CameraInfo(
                header=image_message.header, height=FRAME_HEIGHT, width=FRAME_WIDTH
            )
        )

        pose_message = PoseStamped()
        pose_message.header.stamp = now
        position_x, position_y, position_z = tool_position(tick=tick)
        pose_message.pose.position.x = position_x
        pose_message.pose.position.y = position_y
        pose_message.pose.position.z = position_z
        pose_message.pose.orientation.w = 1.0
        pose.publish(pose_message)

        gripper.publish(Bool(data=gripper_open(tick=tick)))
        state["tick"] = tick + 1

    node.create_timer(1.0 / PUBLISH_RATE_HZ, publish_once)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
