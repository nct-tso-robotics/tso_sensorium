"""Tests for tso_sensorium.processing.rectification module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from tso_sensorium.processing.rectification import Rectifier

FILE_STORAGE_PATH = "tso_sensorium.processing.rectification.cv2.FileStorage"
INIT_MAP_PATH = "tso_sensorium.processing.rectification.cv2.initUndistortRectifyMap"
REMAP_PATH = "tso_sensorium.processing.rectification.cv2.remap"


@pytest.fixture
def calibration_nodes() -> dict:
    return {
        "cam1": np.diag([100.0, 100.0, 1.0]),
        "cam2": np.diag([200.0, 200.0, 1.0]),
        "dis1": np.zeros((1, 5)),
        "dis2": np.ones((1, 5)),
        "R1": np.eye(3),
        "R2": 2.0 * np.eye(3),
        "P1": np.hstack([np.diag([100.0, 100.0, 1.0]), np.zeros((3, 1))]),
        "P2": np.hstack([np.diag([200.0, 200.0, 1.0]), np.zeros((3, 1))]),
        "imageSize": np.array([[640], [480]]),
    }


@pytest.fixture
def mock_file_storage(calibration_nodes) -> MagicMock:
    storage = MagicMock()

    def get_node(name: str) -> MagicMock:
        node = MagicMock()
        node.mat.return_value = calibration_nodes[name]
        return node

    storage.getNode.side_effect = get_node
    return storage


@pytest.fixture
def rectifier_factory(mock_file_storage):
    def factory(
        left_maps=("left_x", "left_y"), right_maps=("right_x", "right_y")
    ) -> Rectifier:
        with (
            patch(FILE_STORAGE_PATH, return_value=mock_file_storage),
            patch(INIT_MAP_PATH, side_effect=[left_maps, right_maps]),
        ):
            return Rectifier(calibration_file_path=Path("calibration.yml"))

    return factory


class TestRectifierInitialization:
    @pytest.mark.unit
    def test_reads_calibration_file_and_builds_both_maps(
        self, mock_file_storage, calibration_nodes
    ):
        with (
            patch(FILE_STORAGE_PATH, return_value=mock_file_storage) as file_storage,
            patch(INIT_MAP_PATH, side_effect=[("lx", "ly"), ("rx", "ry")]) as init_map,
        ):
            Rectifier(calibration_file_path=Path("calibration.yml"))
        file_storage.assert_called_once_with("calibration.yml", cv2.FILE_STORAGE_READ)
        mock_file_storage.release.assert_called_once_with()
        left_call, right_call = init_map.call_args_list
        expected_left = (
            calibration_nodes["cam1"],
            calibration_nodes["dis1"],
            calibration_nodes["R1"],
            calibration_nodes["P1"],
        )
        expected_right = (
            calibration_nodes["cam2"],
            calibration_nodes["dis2"],
            calibration_nodes["R2"],
            calibration_nodes["P2"],
        )
        for actual, expected in zip(left_call.args[:4], expected_left):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(right_call.args[:4], expected_right):
            np.testing.assert_array_equal(actual, expected)
        assert left_call.args[4] == (640, 480)
        assert right_call.args[4] == (640, 480)
        assert left_call.args[5] == cv2.CV_32FC1

    @pytest.mark.unit
    def test_stores_left_and_right_maps(self, rectifier_factory):
        rectifier = rectifier_factory(left_maps=("lx", "ly"), right_maps=("rx", "ry"))
        assert (rectifier.left_map_x, rectifier.left_map_y) == ("lx", "ly")
        assert (rectifier.right_map_x, rectifier.right_map_y) == ("rx", "ry")


class TestRectify:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "method_name, expected_maps",
        [
            ("rectify_left", ("lx", "ly")),
            ("rectify_right", ("rx", "ry")),
        ],
    )
    def test_single_image_remaps_with_matching_maps(
        self, rectifier_factory, method_name, expected_maps
    ):
        rectifier = rectifier_factory(left_maps=("lx", "ly"), right_maps=("rx", "ry"))
        image = MagicMock()
        with patch(REMAP_PATH, return_value="rectified") as remap:
            result = getattr(rectifier, method_name)(image=image)
        remap.assert_called_once_with(
            image, expected_maps[0], expected_maps[1], cv2.INTER_LINEAR
        )
        assert result == "rectified"

    @pytest.mark.unit
    def test_pair_rectifies_left_and_right(self, rectifier_factory):
        rectifier = rectifier_factory()
        with patch(REMAP_PATH, side_effect=["rectified_left", "rectified_right"]):
            left, right = rectifier.rectify(left=MagicMock(), right=MagicMock())
        assert (left, right) == ("rectified_left", "rectified_right")


@pytest.mark.integration
def test_identity_calibration_leaves_image_unchanged(tmp_path, rng):
    width, height = 64, 48
    camera_matrix = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    projection = np.hstack([camera_matrix, np.zeros((3, 1))])
    storage = cv2.FileStorage(str(tmp_path / "calibration.yml"), cv2.FILE_STORAGE_WRITE)
    for side in ("1", "2"):
        storage.write(f"cam{side}", camera_matrix)
        storage.write(f"dis{side}", np.zeros((1, 5)))
        storage.write(f"R{side}", np.eye(3))
        storage.write(f"P{side}", projection)
    storage.write("imageSize", np.array([[width], [height]], dtype=np.int32))
    storage.release()

    rectifier = Rectifier(calibration_file_path=tmp_path / "calibration.yml")
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    rectified_left, rectified_right = rectifier.rectify(left=image, right=image)
    np.testing.assert_allclose(rectified_left, image, atol=1)
    np.testing.assert_allclose(rectified_right, image, atol=1)
