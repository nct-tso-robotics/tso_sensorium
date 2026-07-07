"""Live display of ROS 1 image topics with composable preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

from tso_sensorium.processing import Rectifier
from tso_sensorium.processing.image import (
    convert_stereo_imgs_to_anaglyph,
    deinterlace_cv_image,
)
from tso_sensorium.resources import STORZ_ENDOSCOPE_CALIBRATION_PATH

ESCAPE_KEY = 27
STORZ_FRAME_SIZE = (960, 540)
PreprocessFn = Callable[[np.ndarray], np.ndarray]


def stream_image_topic(
    topic_name: str,
    get_cv_img_from_msg_fn: Callable[[Any], np.ndarray],
    is_compressed: bool,
    preprocess_img_fns: Optional[list[PreprocessFn]] = None,
    title: str = "Camera Feed",
    stop_key: int = ESCAPE_KEY,
) -> None:
    """Display a ROS image topic in an OpenCV window until stopped.

    Expects the ROS node to already exist; this function only subscribes
    and spins.

    Args:
        topic_name: ROS topic to subscribe to.
        get_cv_img_from_msg_fn: Converts the ROS message to a numpy image,
            handling decompression when needed.
        is_compressed: Whether the topic publishes compressed images,
            which determines the subscribed message type.
        preprocess_img_fns: Transforms applied in order before display.
        title: OpenCV window title.
        stop_key: Key code that stops the feed.
    """
    if preprocess_img_fns is None:
        preprocess_img_fns = []

    def image_callback(msg: Any) -> None:
        cv_image = get_cv_img_from_msg_fn(msg)
        for preprocess_img_fn in preprocess_img_fns:
            cv_image = preprocess_img_fn(cv_image)
        cv2.imshow(title, cv_image)
        key = cv2.waitKey(1)
        if key == stop_key:
            cv2.destroyAllWindows()
            rospy.signal_shutdown("Shutdown requested")

    msg_type = CompressedImage if is_compressed else Image
    rospy.Subscriber(topic_name, msg_type, image_callback)
    rospy.spin()
    cv2.destroyAllWindows()


def convert_raw_img_to_cv_img(msg: Any) -> np.ndarray:
    """Convert a raw ROS image message to a numpy array.

    The pixel layout is taken from the message's own encoding field.

    Args:
        msg: ``sensor_msgs/Image`` message.

    Returns:
        Image as a numpy array.
    """
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(msg, desired_encoding=msg.encoding)


def convert_jpg_compressed_img_to_cv_img(msg: Any) -> np.ndarray:
    """Convert a JPEG-compressed ROS image message to a numpy array.

    Args:
        msg: ``sensor_msgs/CompressedImage`` message with JPEG data.

    Returns:
        Decoded BGR image.
    """
    compressed_data = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)


def create_add_grid_to_img_preprocess_fn(
    cell_size: int = 50,
    color: tuple = (128, 128, 128),
    thickness: int = 1,
) -> PreprocessFn:
    """Create a preprocessing function that draws a grid overlay.

    Useful as a visual reference for calibration or measurement.

    Args:
        cell_size: Size of each grid cell in pixels.
        color: BGR color of the grid lines.
        thickness: Thickness of the grid lines in pixels.

    Returns:
        Function drawing the grid onto the given image.
    """

    def preprocess_fn(cv_image: np.ndarray) -> np.ndarray:
        for x in range(0, cv_image.shape[1], cell_size):
            cv2.line(cv_image, (x, 0), (x, cv_image.shape[0]), color, thickness)
        for y in range(0, cv_image.shape[0], cell_size):
            cv2.line(cv_image, (0, y), (cv_image.shape[1], y), color, thickness)
        return cv_image

    return preprocess_fn


def create_rectify_img_preprocess_fn(
    calibration_path: Path | str, left: bool = True
) -> PreprocessFn:
    """Create a preprocessing function that rectifies one camera's images.

    Args:
        calibration_path: OpenCV stereo calibration file.
        left: Whether to rectify as the left camera; otherwise the right.

    Returns:
        Function rectifying the given image.
    """
    rectifier = Rectifier(calibration_file_path=calibration_path)

    def preprocess_fn(cv_image: np.ndarray) -> np.ndarray:
        if left:
            return rectifier.rectify_left(image=cv_image)
        return rectifier.rectify_right(image=cv_image)

    return preprocess_fn


def create_storz_endoscope_img_to_anaglyph_preprocess_fn() -> PreprocessFn:
    """Create a preprocessing function for the interlaced STORZ endoscope.

    Deinterlaces the incoming image, rectifies both halves with the
    packaged calibration, and combines them into an anaglyph.

    Returns:
        Function converting an interlaced stereo image to an anaglyph.
    """
    rectifier = Rectifier(calibration_file_path=STORZ_ENDOSCOPE_CALIBRATION_PATH)

    def preprocess_fn(cv_image: np.ndarray) -> np.ndarray:
        left_img, right_img = deinterlace_cv_image(image=cv_image, left_odd=True)
        left_img = cv2.resize(left_img, STORZ_FRAME_SIZE)
        right_img = cv2.resize(right_img, STORZ_FRAME_SIZE)
        left_img = rectifier.rectify_left(image=left_img)
        right_img = rectifier.rectify_right(image=right_img)
        return convert_stereo_imgs_to_anaglyph(left_img=left_img, right_img=right_img)

    return preprocess_fn
