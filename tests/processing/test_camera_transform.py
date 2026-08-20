"""Tests for tso_sensorium.processing.camera_transform module."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tso_sensorium.processing.camera_transform import (
    CameraMountCalibration,
    camera_quaternion_from_end_effector,
    camera_quaternion_from_euler,
    compose_base_to_camera_rotation,
    load_camera_mount_calibration,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TESTBED_CALIBRATION_PATH = (
    REPOSITORY_ROOT / "configs" / "calibration" / "tso_endoscope_mount.yaml"
)
EXPECTED_AXIS_MAPPING = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
)


@pytest.fixture
def camera_mount_calibration_factory():
    def factory(
        aligned_camera_from_end_effector: np.ndarray | None = None,
        optical_tilt_radians: float = 0.0,
    ) -> CameraMountCalibration:
        if aligned_camera_from_end_effector is None:
            aligned_camera_from_end_effector = np.eye(3)
        return CameraMountCalibration(
            aligned_camera_from_end_effector=aligned_camera_from_end_effector,
            optical_tilt_radians=optical_tilt_radians,
        )

    return factory


@pytest.mark.unit
def test_rejects_non_rotation_axis_mapping(camera_mount_calibration_factory) -> None:
    invalid_matrix = np.diag([1.0, 1.0, 2.0])
    with pytest.raises(
        ValueError,
        match="aligned_camera_from_end_effector must be a proper rotation matrix",
    ):
        camera_mount_calibration_factory(
            aligned_camera_from_end_effector=invalid_matrix
        )


@pytest.mark.unit
def test_composition_maps_verified_camera_axes_to_end_effector_axes(
    camera_mount_calibration_factory,
) -> None:
    calibration = camera_mount_calibration_factory(
        aligned_camera_from_end_effector=EXPECTED_AXIS_MAPPING
    )

    base_to_camera = compose_base_to_camera_rotation(
        base_to_end_effector=np.eye(3),
        calibration=calibration,
    )

    np.testing.assert_allclose(base_to_camera[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(base_to_camera[:, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(base_to_camera[:, 2], [1.0, 0.0, 0.0])


@pytest.mark.unit
def test_euler_and_quaternion_inputs_share_calibration(
    camera_mount_calibration_factory,
) -> None:
    calibration = camera_mount_calibration_factory(
        aligned_camera_from_end_effector=EXPECTED_AXIS_MAPPING,
        optical_tilt_radians=-np.pi / 6,
    )
    roll, pitch, yaw = 0.2, -0.3, 0.4
    end_effector_quaternion = tuple(
        Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    )

    from_euler = camera_quaternion_from_euler(
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        calibration=calibration,
    )
    from_quaternion = camera_quaternion_from_end_effector(
        quaternion=end_effector_quaternion,
        calibration=calibration,
    )

    np.testing.assert_allclose(from_quaternion, from_euler)


@pytest.mark.unit
def test_testbed_calibration_preserves_legacy_zero_pose_quaternion() -> None:
    calibration = load_camera_mount_calibration(
        calibration_path=TESTBED_CALIBRATION_PATH
    )

    quaternion = camera_quaternion_from_euler(
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        calibration=calibration,
    )

    np.testing.assert_allclose(
        quaternion,
        [0.35355339, 0.35355339, 0.61237244, 0.61237244],
    )


@pytest.mark.integration
def test_shipped_testbed_calibration_preserves_verified_geometry() -> None:
    calibration = load_camera_mount_calibration(
        calibration_path=TESTBED_CALIBRATION_PATH
    )

    np.testing.assert_array_equal(
        calibration.aligned_camera_from_end_effector,
        EXPECTED_AXIS_MAPPING,
    )
    assert calibration.optical_tilt_radians == pytest.approx(-np.pi / 6)
