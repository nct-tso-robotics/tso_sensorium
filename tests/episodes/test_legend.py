"""Tests for tso_sensorium.episodes.legend module."""

import json

import pytest

from tso_sensorium.episodes.legend import DatasetMetadata
from tso_sensorium.episodes.schema import (
    CoordinateFrameFeatureMetadata,
    FrameTemporality,
)


@pytest.mark.unit
def test_coordinate_frame_feature_metadata_round_trips_as_json_values(tmp_path):
    metadata_path = tmp_path / "dataset_metadata.json"
    metadata = DatasetMetadata(
        dataset_name="moving_camera",
        coordinate_frame_features={
            "camera_position": CoordinateFrameFeatureMetadata(
                columns=["camera_x", "camera_y", "camera_z"],
                frame="camera",
                frame_temporality=FrameTemporality.MOVING,
            )
        },
    )

    metadata.save(path=metadata_path)

    payload = json.loads(metadata_path.read_text())
    assert payload["coordinate_frame_features"] == {
        "camera_position": {
            "columns": ["camera_x", "camera_y", "camera_z"],
            "frame": "camera",
            "frame_temporality": "moving",
        }
    }
    assert DatasetMetadata.load(path=metadata_path) == metadata
