#!/usr/bin/env python3
"""
Random ROS Topic Publisher

Publishes simulated data to ModelPoseSetter-compatible topics.
Run with: python random_publisher.py [options]
Example:
```
python -m tso_sensorium.recording.ros1.random_publisher --robot_state_topic=/fake_robot/RobotState
--interlaced_image_topic=/fake_camera --gripper_topic=/fake_koala/open --transform_topic=/fake_rct
```
Model Pose Setter compatible call:
```
roslaunch robot_pose_setters model_pose_setter.launch bind_ip:=10.136.27.158 bind_port:=5555 robot_to_camera_transform_topic_name:=/fake_rct
left_image_topic:=/fake_camera right_image_topic:=/fake_camera robot_state_topic_name:=/fake_robot/RobotState set_
robot_pose_topic_name:=/fake_robot/set_tip_pose gripper_state_topic_name:=/fake_koala/open use_interlaced_image:=false
"""

from __future__ import annotations

import argparse
import random
import rospy
import numpy as np
import threading
import time
from typing import Any, Callable, Optional
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Twist, Point, Wrench, TransformStamped
from testbed_msgs.msg import RobotState, KoalaCommand


class RandomPublisher:
    """
    Publishes random messages to ROS topics at specified frequencies.

    Creates random data using a provided function and publishes it to a ROS topic
    at a specified rate in a separate thread to avoid blocking the main thread.
    """

    def __init__(
        self,
        topic_name: str,
        msg_type: Any,
        create_msg_fn: Callable[[], Any],
        hz: float = 10.0,
        queue_size: int = 10,
    ):
        """Standalone init with direct args."""
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.create_msg_fn = create_msg_fn
        self.hz = hz
        self.publisher = rospy.Publisher(topic_name, msg_type, queue_size=queue_size)
        self.publish_thread: Optional[threading.Thread] = None
        rospy.loginfo(
            f"RandomPublisher: {topic_name} @ {hz} Hz (type: {msg_type.__name__})"
        )

    def _publish_loop(self) -> None:
        period = 1.0 / self.hz
        while not rospy.is_shutdown():
            start_time = time.time()
            msg = self.create_msg_fn()
            self.publisher.publish(msg)
            elapsed = time.time() - start_time
            sleep_time = max(0, period - elapsed)
            time.sleep(sleep_time)

    def start_publishing(self) -> threading.Thread:
        """Start publishing in a daemon thread."""
        self.publish_thread = threading.Thread(target=self._publish_loop, daemon=True)
        self.publish_thread.start()
        rospy.on_shutdown(self._stop_publishing)  # Hook for clean shutdown
        return self.publish_thread

    def _stop_publishing(self) -> None:
        """Stop the publishing thread on shutdown."""
        if self.publish_thread and self.publish_thread.is_alive():
            self.publish_thread.join(timeout=1.0)  # Brief wait; daemon ensures exit


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone Random Publisher for ModelPoseSetter"
    )
    parser.add_argument(
        "--robot_state_topic",
        default="/ur5e_rcm_twist_controller/RobotState",
        help="RobotState topic",
    )
    parser.add_argument(
        "--interlaced_image_topic",
        default="/endoscope/capture/image_raw",
        help="Interlaced image topic",
    )
    parser.add_argument(
        "--gripper_topic", default="/koala_grasper/open", help="Gripper topic"
    )
    parser.add_argument(
        "--transform_topic", default="/robot_camera_transform", help="Transform topic"
    )
    parser.add_argument("--robot_hz", type=float, default=500.0, help="RobotState Hz")
    parser.add_argument("--image_hz", type=float, default=30.0, help="Image Hz")
    parser.add_argument("--gripper_hz", type=float, default=10.0, help="Gripper Hz")
    parser.add_argument("--transform_hz", type=float, default=5.0, help="Transform Hz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rospy.init_node("random_publisher", anonymous=True)

    def create_robot_state_msg() -> RobotState:
        msg = RobotState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""
        # Joints state (UR5-like: 6 DOF)
        msg.joints_state = JointState()
        msg.joints_state.header.stamp = msg.header.stamp
        msg.joints_state.name = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        msg.joints_state.position = [np.random.uniform(-np.pi, np.pi) for _ in range(6)]
        msg.joints_state.velocity = [np.random.uniform(-1, 1) for _ in range(6)]
        msg.joints_state.effort = [np.random.uniform(-10, 10) for _ in range(6)]
        # Robot pose (end-effector)
        msg.robot_pose = Pose()
        msg.robot_pose.position.x = np.random.uniform(-0.5, 0.5)
        msg.robot_pose.position.y = np.random.uniform(-0.5, 0.5)
        msg.robot_pose.position.z = np.random.uniform(0, 0.5)
        quat = np.random.uniform(-1, 1, 4)  # Random vec4
        quat /= np.linalg.norm(quat)  # Normalize to unit quat
        (
            msg.robot_pose.orientation.x,
            msg.robot_pose.orientation.y,
            msg.robot_pose.orientation.z,
            msg.robot_pose.orientation.w,
        ) = quat
        # End-effector twist
        msg.end_effector_twist = Twist()
        msg.end_effector_twist.linear.x = np.random.uniform(-0.05, 0.05)
        msg.end_effector_twist.linear.y = np.random.uniform(-0.05, 0.05)
        msg.end_effector_twist.linear.z = np.random.uniform(-0.05, 0.05)
        msg.end_effector_twist.angular.x = np.random.uniform(-0.1, 0.1)
        msg.end_effector_twist.angular.y = np.random.uniform(-0.1, 0.1)
        msg.end_effector_twist.angular.z = np.random.uniform(-0.1, 0.1)
        # Robot space tip position
        msg.robot_space_tip_position = Point()
        msg.robot_space_tip_position.x = np.random.uniform(-0.5, 0.5)
        msg.robot_space_tip_position.y = np.random.uniform(-0.5, 0.5)
        msg.robot_space_tip_position.z = np.random.uniform(0, 0.5)
        # Robot space tip twist
        msg.robot_space_tip_twist = Twist()
        msg.robot_space_tip_twist.linear.x = np.random.uniform(-0.05, 0.05)
        msg.robot_space_tip_twist.linear.y = np.random.uniform(-0.05, 0.05)
        msg.robot_space_tip_twist.linear.z = np.random.uniform(-0.05, 0.05)
        msg.robot_space_tip_twist.angular.x = np.random.uniform(-0.1, 0.1)
        msg.robot_space_tip_twist.angular.y = np.random.uniform(-0.1, 0.1)
        msg.robot_space_tip_twist.angular.z = np.random.uniform(-0.1, 0.1)
        # Robot space initial tip position
        msg.robot_space_initial_tip_position = Point()
        msg.robot_space_initial_tip_position.x = 0.0  # Fixed initial
        msg.robot_space_initial_tip_position.y = 0.0
        msg.robot_space_initial_tip_position.z = 0.0
        # Robot space RCM position
        msg.robot_space_rcm_position = Point()
        msg.robot_space_rcm_position.x = np.random.uniform(-0.1, 0.1)
        msg.robot_space_rcm_position.y = np.random.uniform(-0.1, 0.1)
        msg.robot_space_rcm_position.z = np.random.uniform(0, 0.1)
        # Relative tip position
        msg.relative_tip_position = Point()
        msg.relative_tip_position.x = np.random.uniform(-0.1, 0.1)
        msg.relative_tip_position.y = np.random.uniform(-0.1, 0.1)
        msg.relative_tip_position.z = np.random.uniform(-0.1, 0.1)
        # Relative pivot RPY
        msg.relative_pivot_RPY = Point()
        msg.relative_pivot_RPY.x = np.random.uniform(-0.1, 0.1)  # Roll
        msg.relative_pivot_RPY.y = np.random.uniform(-0.1, 0.1)  # Pitch
        msg.relative_pivot_RPY.z = np.random.uniform(-0.1, 0.1)  # Yaw
        # Relative pivot translation
        msg.relative_pivot_translation = np.random.uniform(-0.05, 0.05)
        # Robot space wrench
        msg.robot_space_wrench = Wrench()
        msg.robot_space_wrench.force.x = np.random.uniform(-1, 1)
        msg.robot_space_wrench.force.y = np.random.uniform(-1, 1)
        msg.robot_space_wrench.force.z = np.random.uniform(-1, 1)
        msg.robot_space_wrench.torque.x = np.random.uniform(-0.5, 0.5)
        msg.robot_space_wrench.torque.y = np.random.uniform(-0.5, 0.5)
        msg.robot_space_wrench.torque.z = np.random.uniform(-0.5, 0.5)
        return msg

    def create_interlaced_image_msg() -> Image:
        h, w = 1080, 1920
        img = np.random.randint(0, 255, (h, w, 4), dtype=np.uint8)
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""
        msg.height, msg.width = h, w
        msg.encoding = "bgra8"
        msg.is_bigendian = 0
        msg.step = w * 4
        msg.data = img.tobytes()
        return msg

    def create_gripper_msg() -> KoalaCommand:
        msg = KoalaCommand()
        msg.header.stamp = rospy.Time.now()
        msg.open = bool(random.getrandbits(1))
        return msg

    def create_transform_msg() -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = ""
        msg.child_frame_id = "camera_frame"
        # Random translation (fixed offset + noise)
        msg.transform.translation.x = 0.1 + np.random.uniform(-0.01, 0.01)
        msg.transform.translation.y = 0.0 + np.random.uniform(-0.01, 0.01)
        msg.transform.translation.z = 0.0 + np.random.uniform(-0.01, 0.01)
        # Random unit quaternion (no scipy)
        quat = np.random.uniform(-1, 1, 4)
        quat /= np.linalg.norm(quat)
        (
            msg.transform.rotation.x,
            msg.transform.rotation.y,
            msg.transform.rotation.z,
            msg.transform.rotation.w,
        ) = quat
        return msg

    publishers = [
        RandomPublisher(
            args.robot_state_topic, RobotState, create_robot_state_msg, args.robot_hz
        ),
        RandomPublisher(
            args.interlaced_image_topic,
            Image,
            create_interlaced_image_msg,
            args.image_hz,
        ),
        RandomPublisher(
            args.gripper_topic, KoalaCommand, create_gripper_msg, args.gripper_hz
        ),
        RandomPublisher(
            args.transform_topic,
            TransformStamped,
            create_transform_msg,
            args.transform_hz,
        ),
    ]
    threads = [pub.start_publishing() for pub in publishers]
    rospy.loginfo("Publishers active. Ctrl+C to exit.")
    rospy.spin()
