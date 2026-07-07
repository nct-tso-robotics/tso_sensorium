"""Stereo camera rectification using OpenCV calibration files."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LEFT_CAMERA_MATRIX_NODE = "cam1"
RIGHT_CAMERA_MATRIX_NODE = "cam2"
LEFT_DISTORTION_NODE = "dis1"
RIGHT_DISTORTION_NODE = "dis2"
LEFT_ROTATION_NODE = "R1"
RIGHT_ROTATION_NODE = "R2"
LEFT_PROJECTION_NODE = "P1"
RIGHT_PROJECTION_NODE = "P2"
IMAGE_SIZE_NODE = "imageSize"


class Rectifier:
    """Undistorts and aligns stereo images using calibration parameters.

    Loads camera calibration parameters and creates rectification maps to
    correct lens distortion and align epipolar lines for stereo matching.

    Args:
        calibration_file_path: OpenCV calibration file (.xml or .yml) with
            camera matrices, distortion coefficients, rectification
            rotations, projection matrices, and image size.
    """

    def __init__(self, calibration_file_path: Path | str):
        storage = cv2.FileStorage(str(calibration_file_path), cv2.FILE_STORAGE_READ)
        left_camera_matrix = storage.getNode(LEFT_CAMERA_MATRIX_NODE).mat()
        right_camera_matrix = storage.getNode(RIGHT_CAMERA_MATRIX_NODE).mat()
        left_distortion = storage.getNode(LEFT_DISTORTION_NODE).mat()
        right_distortion = storage.getNode(RIGHT_DISTORTION_NODE).mat()
        left_rotation = storage.getNode(LEFT_ROTATION_NODE).mat()
        right_rotation = storage.getNode(RIGHT_ROTATION_NODE).mat()
        left_projection = storage.getNode(LEFT_PROJECTION_NODE).mat()
        right_projection = storage.getNode(RIGHT_PROJECTION_NODE).mat()
        image_size = storage.getNode(IMAGE_SIZE_NODE).mat()
        storage.release()
        width = int(image_size[0][0])
        height = int(image_size[1][0])

        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            left_camera_matrix,
            left_distortion,
            left_rotation,
            left_projection,
            (width, height),
            cv2.CV_32FC1,
        )
        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            right_camera_matrix,
            right_distortion,
            right_rotation,
            right_projection,
            (width, height),
            cv2.CV_32FC1,
        )

    def rectify_left(self, image: np.ndarray) -> np.ndarray:
        """Rectify a left camera image.

        Args:
            image: Raw left camera image.

        Returns:
            Rectified left image.
        """
        return cv2.remap(image, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR)

    def rectify_right(self, image: np.ndarray) -> np.ndarray:
        """Rectify a right camera image.

        Args:
            image: Raw right camera image.

        Returns:
            Rectified right image.
        """
        return cv2.remap(image, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR)

    def rectify(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rectify both stereo images simultaneously.

        Args:
            left: Raw left camera image.
            right: Raw right camera image.

        Returns:
            Tuple of (rectified_left, rectified_right) images.
        """
        return self.rectify_left(image=left), self.rectify_right(image=right)
