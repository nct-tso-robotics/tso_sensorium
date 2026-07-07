"""Synthetic ROS 1 sensors for trying the recording stack without hardware.

Publishes a moving test-pattern camera, a circular tool pose, and a
toggling gripper, matching ``configs/recording/mock_service.yaml``:

    python -m tso_sensorium.scripts.mock_sensors
"""

import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

FRAME_HEIGHT = 540
FRAME_WIDTH = 960
PUBLISH_RATE_HZ = 15
GRIPPER_INTERVAL_TICKS = 45
CAMERA_TOPIC = "/mock/camera"
POSE_TOPIC = "/mock/pose"
GRIPPER_TOPIC = "/mock/gripper"


def render_frame(tick: int, base_x: np.ndarray, base_y: np.ndarray) -> np.ndarray:
    """Render one synthetic camera frame.

    Args:
        tick: Frame counter driving the animation.
        base_x: Column index grid, (H, W).
        base_y: Row index grid, (H, W).

    Returns:
        BGR frame, (H, W, 3).
    """
    phase = tick * 4
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = ((base_x + phase) % 512 // 2).astype(np.uint8)
    frame[:, :, 1] = ((base_y + phase // 2) % 512 // 2).astype(np.uint8)
    frame[:, :, 2] = 40
    center = (
        int(FRAME_WIDTH / 2 + 260 * np.cos(tick / 20)),
        int(FRAME_HEIGHT / 2 + 160 * np.sin(tick / 20)),
    )
    cv2.circle(frame, center, 42, (60, 200, 90), -1)
    cv2.putText(
        frame,
        f"mock camera  {time.strftime('%H:%M:%S')}",
        (24, FRAME_HEIGHT - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
    )
    return frame


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
        pose_message.pose.position.x = float(np.cos(tick / 20))
        pose_message.pose.position.y = float(np.sin(tick / 20))
        pose_message.pose.position.z = 0.1
        pose_message.pose.orientation.w = 1.0
        pose.publish(pose_message)

        gripper.publish(Bool(data=(tick // GRIPPER_INTERVAL_TICKS) % 2 == 0))
        tick += 1
        rate.sleep()


if __name__ == "__main__":
    main()
