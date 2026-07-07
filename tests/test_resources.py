"""Tests for tso_sensorium.resources module."""

import pytest

from tso_sensorium.resources import (
    ASSETS_DIR,
    STORZ_ENDOSCOPE_CALIBRATION_PATH,
)


@pytest.mark.unit
def test_assets_dir_points_to_packaged_assets():
    assert ASSETS_DIR.name == "assets"
    assert ASSETS_DIR.is_dir()


@pytest.mark.unit
def test_storz_calibration_file_is_packaged():
    assert STORZ_ENDOSCOPE_CALIBRATION_PATH.is_file()
