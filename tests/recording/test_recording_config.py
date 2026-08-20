"""Tests for tso_sensorium.recording.config module."""

from pathlib import Path

import pytest

from tso_sensorium.configuration import load_config, parse_config_from_cli
from tso_sensorium.recording.config import (
    RecordingServiceConfig,
    RecordingSessionConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TSO_TESTBED_CONFIG = REPOSITORY_ROOT / "configs" / "recording" / "tso_testbed.yaml"
TSO_TESTBED_SERVICE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "tso_testbed_service.yaml"
)
ENDOSCOPE_GUIDANCE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "endoscope_guidance.yaml"
)
ENDOSCOPE_GUIDANCE_SERVICE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "endoscope_guidance_service.yaml"
)


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
    assert topic_recorders[0].queue_size == 100
    assert video_recorders[2].frames_per_second == 30
    assert video_recorders[2].lossless_compression is True
    assert "/ur5e_rcm_twist_controller/RobotState" in config.rosbag_topics


@pytest.mark.unit
def test_shipped_tso_testbed_service_config_decodes_nested_includes():
    config = load_config(
        config_class=RecordingServiceConfig, config_path=TSO_TESTBED_SERVICE_CONFIG
    )

    assert config.session.output_folder == ""
    assert len(config.session.recorders) == 8
    assert config.generation is not None
    assert config.generation.annotations is not None
    assert config.generation.annotations.legend.dataset_name == "bowel_retraction"


@pytest.mark.unit
def test_shipped_endoscope_guidance_config_contains_only_guidance_streams():
    config = load_config(
        config_class=RecordingSessionConfig,
        config_path=ENDOSCOPE_GUIDANCE_CONFIG,
    )

    assert [recorder.topic_name for recorder in config.recorders] == [
        "/ur5e_rcm_twist_controller/RobotState",
        "/robot_camera_transform",
        "/laparoscope/camera/left/image_raw",
        "/laparoscope/camera/right/image_raw",
        "/stereo/camera_driver/image_raw",
    ]
    assert all(
        excluded_topic not in config.rosbag_topics
        for excluded_topic in (
            "/koala_grasper/state",
            "/pedal_board/state",
            "/bota_force_sensor_state",
        )
    )


@pytest.mark.unit
def test_shipped_endoscope_guidance_service_uses_left_camera_preview():
    config = load_config(
        config_class=RecordingServiceConfig,
        config_path=ENDOSCOPE_GUIDANCE_SERVICE_CONFIG,
    )

    assert config.camera_topic == "/laparoscope/camera/left/image_raw"
    assert config.generation is None
    assert config.session.output_folder == ""


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
