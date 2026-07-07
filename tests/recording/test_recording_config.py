"""Tests for tso_sensorium.recording.config module."""

from pathlib import Path

import pytest

from tso_sensorium.configuration import load_config, parse_config_from_cli
from tso_sensorium.recording.config import (
    RecordingSessionConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TSO_TESTBED_CONFIG = REPOSITORY_ROOT / "configs" / "recording" / "tso_testbed.yaml"


@pytest.mark.unit
def test_shipped_tso_testbed_config_decodes():
    config = load_config(
        config_class=RecordingSessionConfig, config_path=TSO_TESTBED_CONFIG
    )
    assert len(config.recorders) == 8
    topic_recorders = [
        recorder
        for recorder in config.recorders
        if isinstance(recorder, TopicRecorderConfig)
    ]
    video_recorders = [
        recorder
        for recorder in config.recorders
        if isinstance(recorder, VideoRecorderConfig)
    ]
    assert len(topic_recorders) == 5
    assert len(video_recorders) == 3
    assert topic_recorders[0].message_type == "testbed_msgs.msg.RobotState"
    assert video_recorders[2].lossless_compression is True
    assert "/ur5e_rcm_twist_controller/RobotState" in config.rosbag_topics


@pytest.mark.unit
def test_cli_overrides_apply_on_top_of_yaml():
    config = parse_config_from_cli(
        config_class=RecordingSessionConfig,
        arguments=[
            "--config_path",
            str(TSO_TESTBED_CONFIG),
            "--output_folder",
            "/data/recordings",
            "--record_rosbag",
            "true",
        ],
    )
    assert config.output_folder == "/data/recordings"
    assert config.record_rosbag is True
