"""Robot-to-camera rotation calculations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROTATION_MATRIX_SHAPE = (3, 3)
ROTATION_VALIDATION_TOLERANCE = 1e-6


@dataclass(frozen=True)
class CameraMountCalibration:
    """Fixed rotation from an end effector to an optical camera.

    Args:
        aligned_camera_from_end_effector: Axis mapping from the end-effector
            frame to the aligned camera frame.
        optical_tilt_radians: Optical-frame tilt around its x-axis.
    """

    aligned_camera_from_end_effector: np.ndarray
    optical_tilt_radians: float

    def __post_init__(self) -> None:
        if self.aligned_camera_from_end_effector.shape != ROTATION_MATRIX_SHAPE:
            raise ValueError(
                "aligned_camera_from_end_effector must have shape (3, 3), "
                f"got {self.aligned_camera_from_end_effector.shape}"
            )
        matrix = self.aligned_camera_from_end_effector
        is_orthonormal = np.allclose(
            matrix.T @ matrix,
            np.eye(ROTATION_MATRIX_SHAPE[0]),
            atol=ROTATION_VALIDATION_TOLERANCE,
        )
        has_positive_determinant = np.isclose(
            np.linalg.det(matrix),
            1.0,
            atol=ROTATION_VALIDATION_TOLERANCE,
        )
        if not is_orthonormal or not has_positive_determinant:
            raise ValueError(
                "aligned_camera_from_end_effector must be a proper rotation matrix"
            )


def load_camera_mount_calibration(
    *, calibration_path: Path | str
) -> CameraMountCalibration:
    """Load a camera mount calibration from YAML."""
    with Path(calibration_path).open(encoding="utf-8") as calibration_file:
        values = yaml.safe_load(calibration_file)
    return CameraMountCalibration(
        aligned_camera_from_end_effector=np.asarray(
            values["aligned_camera_from_end_effector"],
            dtype=float,
        ),
        optical_tilt_radians=float(values["optical_tilt_radians"]),
    )


def rotation_x(*, angle: float) -> np.ndarray:
    """Return a rotation matrix around the x-axis."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(angle), -np.sin(angle)],
            [0.0, np.sin(angle), np.cos(angle)],
        ]
    )


def rotation_y(*, angle: float) -> np.ndarray:
    """Return a rotation matrix around the y-axis."""
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )


def rotation_z(*, angle: float) -> np.ndarray:
    """Return a rotation matrix around the z-axis."""
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def compose_base_to_camera_rotation(
    *,
    base_to_end_effector: np.ndarray,
    calibration: CameraMountCalibration,
) -> np.ndarray:
    """Compose an EE orientation with a camera mount calibration."""
    end_effector_from_aligned_camera = calibration.aligned_camera_from_end_effector.T
    aligned_camera_from_optical_camera = rotation_x(
        angle=calibration.optical_tilt_radians
    )
    return (
        base_to_end_effector
        @ end_effector_from_aligned_camera
        @ aligned_camera_from_optical_camera
    )


def camera_quaternion_from_euler(
    *,
    roll: float,
    pitch: float,
    yaw: float,
    calibration: CameraMountCalibration,
) -> np.ndarray:
    """Compute the base-to-camera quaternion from EE Euler angles."""
    base_to_end_effector = (
        rotation_z(angle=yaw) @ rotation_y(angle=pitch) @ rotation_x(angle=roll)
    )
    base_to_camera = compose_base_to_camera_rotation(
        base_to_end_effector=base_to_end_effector,
        calibration=calibration,
    )
    return Rotation.from_matrix(base_to_camera).as_quat()


def camera_quaternion_from_end_effector(
    *,
    quaternion: tuple[float, float, float, float],
    calibration: CameraMountCalibration,
) -> np.ndarray:
    """Compute the base-to-camera quaternion from an EE quaternion."""
    base_to_end_effector = Rotation.from_quat(quaternion).as_matrix()
    base_to_camera = compose_base_to_camera_rotation(
        base_to_end_effector=base_to_end_effector,
        calibration=calibration,
    )
    return Rotation.from_matrix(base_to_camera).as_quat()
