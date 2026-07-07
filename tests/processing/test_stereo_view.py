"""Tests for tso_sensorium.processing.stereo_view module."""

import re

import numpy as np
import pytest

from tso_sensorium.processing.stereo_view import StereoViewProcessor


@pytest.fixture
def interlaced_frame_factory(rng):
    def factory(height=8, width=6):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[1::2] = 200  # odd rows = left eye
        frame[::2] = 50
        return frame

    return factory


class TestStereoViewProcessor:
    @pytest.mark.unit
    def test_rejects_unknown_mode(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "mode must be one of ('interlaced', 'duplicate'), got 'depth'"
            ),
        ):
            StereoViewProcessor(mode="depth")

    @pytest.mark.unit
    def test_duplicate_mode_mirrors_frame_to_both_eyes(self):
        processor = StereoViewProcessor(mode="duplicate")
        frame = np.full((4, 6, 3), 7, dtype=np.uint8)
        side_by_side = processor.side_by_side(frame=frame)
        assert side_by_side.shape == (4, 12, 3)
        np.testing.assert_array_equal(side_by_side[:, :6], side_by_side[:, 6:])

    @pytest.mark.unit
    def test_interlaced_mode_splits_rows(self, interlaced_frame_factory):
        processor = StereoViewProcessor(mode="interlaced", left_odd=True)
        side_by_side = processor.side_by_side(frame=interlaced_frame_factory())
        assert side_by_side.shape == (4, 12, 3)
        assert side_by_side[:, :6].max() == 200  # left = odd rows
        assert side_by_side[:, 6:].max() == 50

    @pytest.mark.unit
    def test_anaglyph_takes_red_from_left_cyan_from_right(
        self, interlaced_frame_factory
    ):
        processor = StereoViewProcessor(mode="interlaced", left_odd=True)
        anaglyph = processor.anaglyph(frame=interlaced_frame_factory())
        assert anaglyph.shape == (4, 6, 3)
        assert anaglyph[..., 2].max() == 200  # red channel from left
        assert anaglyph[..., 0].max() == 50  # blue channel from right
