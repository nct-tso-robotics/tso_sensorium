"""
Script to publish robot-camera transform as a quaternion computed from Euler RPY angles expressed in radians.
This script receives the robot base roll, pitch, yaw as input parameters, computes the quaternion,
and publishes it as a ROS transform message for the robot-camera transformation.
"""

import rospy
import argparse
import sys
from geometry_msgs.msg import TransformStamped
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotation_matrix_to_quaternion(R_matrix):
    # Create a Rotation object from the 3x3 rotation matrix
    rotation = R.from_matrix(R_matrix)
    # Convert the rotation to a quaternion
    quaternion = rotation.as_quat()  # returns [x, y, z, w] scalar last
    return quaternion


def rotation_x(theta):
    """Rotation matrix around x-axis."""
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ]
    )


def rotation_y(theta):
    """Rotation matrix around y-axis."""
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )


def rotation_z(theta):
    """Rotation matrix around z-axis."""
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def compute_base_to_camera_quaternion(roll, pitch, yaw):
    # Step 1: Camera to aligned camera (correct 30° tilt around x-axis)
    tilt_angle = -np.pi / 6  # -30 degrees
    R_camera_to_aligned = rotation_x(tilt_angle)
    # Step 2: Aligned camera to end effector
    R_align_to_ee = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    # Step 3: End effector to base (inverse of base to EE)
    R_base_to_ee = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    # camera_cut = x= 0.5459517832815843 y=0.792021 z=-0.229473 w=0.148255
    R_base_to_camera = R_base_to_ee @ R_align_to_ee.T @ R_camera_to_aligned
    # R_base_to_camera = R_camera_to_aligned@R_align_to_ee.T@R_base_to_ee#( R_align_to_ee.T @ R_camera_to_aligned @ R_base_to_ee)
    # Step 4: Total rotation
    # R_cam_to_base = R_ee_to_base @ R_align_to_ee @ R_camera_to_aligned
    R_cam_to_base = R_base_to_camera.T
    R_base_to_camera = R_cam_to_base.T
    quaternion = rotation_matrix_to_quaternion(R_base_to_camera)
    return quaternion


def parse_euler(euler_str):
    """
    Parse Euler angles from string input.
    Args:
        euler_str: String containing Euler angles (roll,pitch,yaw) in radians
    Returns:
        tuple: (roll, pitch, yaw) as floats
    Raises:
        ValueError: If format is invalid
    """
    try:
        # Remove any whitespace and split by comma
        values = euler_str.replace(" ", "").split(",")
        if len(values) != 3:
            raise ValueError("Euler angles must have exactly 3 values (roll,pitch,yaw)")
        roll, pitch, yaw = map(float, values)
        return roll, pitch, yaw
    except ValueError as e:
        raise ValueError(
            f"Invalid Euler format: {euler_str}. Expected format: roll,pitch,yaw"
        ) from e


def publish_transform(
    quaternion,
    topic_name="/robot_camera_transform",
    parent_frame="robot_base",
    child_frame="camera",
    translation=(0.0, 0.0, 0.0),
    rate=10,
):
    """
    Publish the transform with the given quaternion.
    Args:
        quaternion: Tuple of (x, y, z, w) quaternion values
        topic_name: Name of the topic to publish to (default: "/robot_camera_transform")
        parent_frame: Name of the parent frame (default: "robot_base")
        child_frame: Name of the child frame (default: "camera")
        translation: Tuple of (x, y, z) translation values (default: (0,0,0))
        rate: Publishing rate in Hz (default: 10)
    """
    # Initialize ROS node
    rospy.init_node("robot_camera_transform_publisher", anonymous=True)
    # Create publisher
    publisher = rospy.Publisher(topic_name, TransformStamped, queue_size=1)
    # Wait for publisher to be ready
    rospy.sleep(0.1)
    # Create transform message
    transform = TransformStamped()
    transform.header.frame_id = parent_frame
    transform.child_frame_id = child_frame
    # Set translation
    transform.transform.translation.x = translation[0]
    transform.transform.translation.y = translation[1]
    transform.transform.translation.z = translation[2]
    # Set rotation (quaternion)
    transform.transform.rotation.x = quaternion[0]
    transform.transform.rotation.y = quaternion[1]
    transform.transform.rotation.z = quaternion[2]
    transform.transform.rotation.w = quaternion[3]
    # Set timestamp
    transform.header.stamp = rospy.Time.now()
    # Create rate object
    rate_obj = rospy.Rate(rate)
    rospy.loginfo(f"Publishing transform to topic: {topic_name}")
    rospy.loginfo(f"Transform from {parent_frame} to {child_frame}")
    rospy.loginfo(f"Translation: {translation}")
    rospy.loginfo(f"Rotation (quaternion): {quaternion}")
    rospy.loginfo(f"Publishing rate: {rate} Hz")
    # Publish continuously
    while not rospy.is_shutdown():
        # Update timestamp
        transform.header.stamp = rospy.Time.now()
        # Publish transform
        publisher.publish(transform)
        # Sleep to maintain rate
        rate_obj.sleep()


def main():
    """Main function to parse arguments and publish transform."""
    parser = argparse.ArgumentParser(
        description="Publish robot-camera transform as a quaternion computed from Euler angles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--euler",
        "-e",
        type=str,
        required=True,
        help="Euler angles as comma-separated values (roll,pitch,yaw) in radians",
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default="/robot_camera_transform",
        help="Topic name to publish to (default: /robot_camera_transform)",
    )
    parser.add_argument(
        "--parent",
        "--parent-frame",
        dest="parent_frame",
        default="robot_base",
        help="Parent frame name (default: robot_base)",
    )
    parser.add_argument(
        "--child",
        "--child-frame",
        dest="child_frame",
        default="camera",
        help="Child frame name (default: camera)",
    )
    parser.add_argument(
        "--translation",
        "--trans",
        type=str,
        default="0.0,0.0,0.0",
        help="Translation as comma-separated values (x,y,z) (default: 0.0,0.0,0.0)",
    )
    parser.add_argument(
        "--rate", "-r", type=int, default=10, help="Publishing rate in Hz (default: 10)"
    )
    args = parser.parse_args()
    try:
        # Parse Euler angles
        roll, pitch, yaw = parse_euler(args.euler)
        # Compute quaternion
        quaternion = compute_base_to_camera_quaternion(roll, pitch, yaw)
        # Parse translation
        translation_str = args.translation.replace(" ", "").split(",")
        if len(translation_str) != 3:
            raise ValueError("Translation must have exactly 3 values (x,y,z)")
        translation = tuple(map(float, translation_str))
        # Validate quaternion (should be unit quaternion)
        quat_norm = np.sqrt(sum(q * q for q in quaternion))
        if abs(quat_norm - 1.0) > 1e-6:
            rospy.logwarn(
                f"Warning: Quaternion is not normalized (norm={quat_norm:.6f}). Normalizing..."
            )
            quaternion = tuple(q / quat_norm for q in quaternion)
        # Publish transform
        publish_transform(
            quaternion=quaternion,
            topic_name=args.topic,
            parent_frame=args.parent_frame,
            child_frame=args.child_frame,
            translation=translation,
            rate=args.rate,
        )
    except ValueError as e:
        rospy.logerr(f"Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        rospy.loginfo("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
