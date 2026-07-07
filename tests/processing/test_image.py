"""Tests for tso_sensorium.processing.image module."""

import numpy as np
import pytest

from tso_sensorium.processing.image import (
    convert_stereo_imgs_to_anaglyph,
    deinterlace_cv_image,
)


@pytest.fixture
def interlaced_image_factory():
    def factory(height: int = 6, width: int = 2) -> np.ndarray:
        row_values = np.arange(height, dtype=np.uint8).reshape(height, 1, 1)
        return np.broadcast_to(row_values, (height, width, 3)).copy()

    return factory


class TestDeinterlaceCvImage:
    @pytest.mark.unit
    @pytest.mark.parametrize("left_odd", [True, False])
    def test_splits_odd_and_even_rows(self, interlaced_image_factory, left_odd):
        image = interlaced_image_factory(height=6)
        left, right = deinterlace_cv_image(image=image, left_odd=left_odd)
        odd_row_values = np.array([1, 3, 5])
        even_row_values = np.array([0, 2, 4])
        expected_left = odd_row_values if left_odd else even_row_values
        expected_right = even_row_values if left_odd else odd_row_values
        np.testing.assert_array_equal(left[:, 0, 0], expected_left)
        np.testing.assert_array_equal(right[:, 0, 0], expected_right)

    @pytest.mark.unit
    def test_output_shapes_are_half_height(self, interlaced_image_factory):
        image = interlaced_image_factory(height=8, width=4)
        left, right = deinterlace_cv_image(image=image, left_odd=True)
        assert left.shape == (4, 4, 3)
        assert right.shape == (4, 4, 3)


@pytest.mark.unit
def test_anaglyph_takes_red_from_left_and_blue_green_from_right():
    left = np.full((2, 2, 3), fill_value=(10, 20, 30), dtype=np.uint8)
    right = np.full((2, 2, 3), fill_value=(40, 50, 60), dtype=np.uint8)
    anaglyph = convert_stereo_imgs_to_anaglyph(left_img=left, right_img=right)
    np.testing.assert_array_equal(anaglyph[:, :, 0], 40)  # Blue from right
    np.testing.assert_array_equal(anaglyph[:, :, 1], 50)  # Green from right
    np.testing.assert_array_equal(anaglyph[:, :, 2], 30)  # Red from left
