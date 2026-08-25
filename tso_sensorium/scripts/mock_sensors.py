"""Synthetic ROS 1 sensors for trying the recording stack without hardware.

Publishes a moving test-pattern camera, a circular tool pose, and a
toggling gripper, matching ``configs/recording/mock_ui.yaml``:

    python -m tso_sensorium.scripts.mock_sensors
"""

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from tso_sensorium.scripts.mock_scene import (
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


def main() -> None:
    """Publish the mock sensors until interrupted."""
    rospy.init_node("mock_sensors", anonymous=True)
    camera = rospy.Publisher(CAMERA_TOPIC, Image, queue_size=1)
    pose = rospy.Publisher(POSE_TOPIC, PoseStamped, queue_size=1)
    gripper = rospy.Publisher(GRIPPER_TOPIC, Bool, queue_size=1)

    rate = rospy.Rate(PUBLISH_RATE_HZ)
    base_y, base_x = np.mgrid[0:FRAME_HEIGHT, 0:FRAME_WIDTH]
    tick = 0
    while not rospy.is_shutdown():
        now = rospy.Time.now()
        frame = render_frame(tick=tick, base_x=base_x, base_y=base_y)
        image_message = Image()
        image_message.height = FRAME_HEIGHT
        image_message.width = FRAME_WIDTH
        image_message.encoding = "bgr8"
        image_message.step = FRAME_WIDTH * 3
        image_message.data = frame.tobytes()
        image_message.header.stamp = now
        camera.publish(image_message)

        pose_message = PoseStamped()
        pose_message.header.stamp = now
        position_x, position_y, position_z = tool_position(tick=tick)
        pose_message.pose.position.x = position_x
        pose_message.pose.position.y = position_y
        pose_message.pose.position.z = position_z
        pose_message.pose.orientation.w = 1.0
        pose.publish(pose_message)

        gripper.publish(Bool(data=gripper_open(tick=tick)))
        tick += 1
        rate.sleep()


if __name__ == "__main__":
    main()
