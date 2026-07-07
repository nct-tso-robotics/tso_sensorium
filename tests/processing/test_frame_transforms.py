"""Tests for tso_sensorium.processing.frame_transforms module."""

import re
from unittest.mock import MagicMock, patch

from pydantic import TypeAdapter
import numpy as np
import pytest

from tso_sensorium.processing.frame_transforms import (
    AnyFrameTransform,
    DeinterlaceHalf,
    RectifySide,
    Resize,
    compose_transforms,
)

RECTIFIER_PATH = "tso_sensorium.processing.frame_transforms.Rectifier"


class TestDeinterlaceHalf:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "side, expected_rows",
        [
            ("left", [1, 3, 5]),
            ("right", [0, 2, 4]),
        ],
    )
    def test_keeps_requested_half(self, side, expected_rows):
        row_values = np.arange(6, dtype=np.uint8).reshape(6, 1, 1)
        image = np.broadcast_to(row_values, (6, 2, 3)).copy()
        transform = DeinterlaceHalf(side=side, left_odd=True)
        np.testing.assert_array_equal(
            transform(image)[:, 0, 0], np.array(expected_rows)
        )

    @pytest.mark.unit
    def test_rejects_unknown_side(self):
        with pytest.raises(
            ValueError,
            match=re.escape("side must be one of ('left', 'right'), got 'top'"),
        ):
            DeinterlaceHalf(side="top")


@pytest.mark.unit
def test_resize_outputs_requested_size(rng):
    image = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)
    resized = Resize(width=32, height=16)(image)
    assert resized.shape == (16, 32, 3)


class TestRectifySide:
    @pytest.mark.unit
    def test_builds_rectifier_lazily_and_routes_by_side(self, rng):
        image = rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8)
        transform = RectifySide(side="right", calibration_path="calibration.yml")
        with patch(RECTIFIER_PATH) as rectifier_class:
            rectifier_class.return_value.rectify_right.return_value = "rectified"
            first = transform(image)
            second = transform(image)
        rectifier_class.assert_called_once_with(calibration_file_path="calibration.yml")
        assert (first, second) == ("rectified", "rectified")

    @pytest.mark.unit
    def test_requires_calibration_path(self):
        with pytest.raises(
            ValueError,
            match=re.escape("rectify transform requires a calibration_path"),
        ):
            RectifySide(side="left")


class TestComposeTransforms:
    @pytest.mark.unit
    def test_applies_transforms_in_order(self):
        first = MagicMock(side_effect=lambda frame: frame + 1)
        second = MagicMock(side_effect=lambda frame: frame * 2)
        preprocess = compose_transforms(transforms=[first, second])
        assert preprocess(np.array([1])) == np.array([4])

    @pytest.mark.unit
    def test_empty_list_returns_none(self):
        assert compose_transforms(transforms=[]) is None


@pytest.mark.unit
def test_choice_registry_decodes_by_type_key():
    transform = TypeAdapter(AnyFrameTransform).validate_python(
        {"type": "resize", "width": 100, "height": 50}
    )
    assert transform == Resize(width=100, height=50)
