"""Tests for tso_sensorium.recording.config module."""

from pathlib import Path

import pytest

from tso_sensorium.configuration import load_config, parse_config_from_cli
from tso_sensorium.recording.config import (
    RecordingSessionConfig,
    RecordingUIConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TSO_TESTBED_CONFIG = REPOSITORY_ROOT / "configs" / "recording" / "tso_testbed.yaml"
TSO_TESTBED_UI_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "tso_testbed_ui.yaml"
)
ENDOSCOPE_GUIDANCE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "endoscope_guidance.yaml"
)
ENDOSCOPE_GUIDANCE_UI_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "endoscope_guidance_ui.yaml"
)
FORCE_SESSION_CONFIG = REPOSITORY_ROOT / "configs" / "recording" / "force_session.yaml"
FORCE_SESSION_UI_CONFIG = (
    REPOSITORY_ROOT / "configs" / "recording" / "force_session_ui.yaml"
)


@pytest.mark.unit
def test_shipped_tso_testbed_config_decodes():
    config = load_config(
        config_class=RecordingSessionConfig, config_path=TSO_TESTBED_CONFIG
    )
    assert len(config.recorders) == 7
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
    assert len(topic_recorders) == 4
    assert len(video_recorders) == 3
    assert topic_recorders[0].message_type == "testbed_msgs.msg.RobotState"
    assert topic_recorders[0].queue_size == 100
    assert video_recorders[2].frames_per_second == 10
    assert video_recorders[2].lossless_compression is True
    assert "/ur5e_rcm_twist_controller/RobotState" in config.rosbag_topics
    assert all(
        recorder.topic_name != "/bota_force_sensor_state"
        for recorder in config.recorders
    )
    assert "/bota_force_sensor_state" not in config.rosbag_topics


@pytest.mark.unit
def test_shipped_tso_testbed_ui_config_decodes_nested_includes():
    config = load_config(
        config_class=RecordingUIConfig, config_path=TSO_TESTBED_UI_CONFIG
    )

    assert config.session.output_folder == ""
    assert len(config.session.recorders) == 7
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
def test_shipped_endoscope_guidance_ui_uses_left_camera_preview():
    config = load_config(
        config_class=RecordingUIConfig,
        config_path=ENDOSCOPE_GUIDANCE_UI_CONFIG,
    )

    assert config.camera_topic == "/laparoscope/camera/left/image_raw"
    assert config.generation is None
    assert config.session.output_folder == ""


@pytest.mark.unit
def test_shipped_force_session_config_records_force_setup():
    config = load_config(
        config_class=RecordingSessionConfig,
        config_path=FORCE_SESSION_CONFIG,
    )

    assert config.output_folder == "~/tso_sensorium_recordings"
    assert [recorder.topic_name for recorder in config.recorders] == [
        "/ur5e_rcm_twist_controller/RobotState",
        "/robot_camera_transform",
        "/laparoscope/camera/left/image_raw",
        "/laparoscope/camera/right/image_raw",
        "/stereo/camera_driver/image_raw",
        "/bota_sensor_publisher/wrench",
        "/bota_sensor_publisher/imu",
    ]
    assert config.recorders[4].lossless_compression is True
    assert config.recorders[4].frames_per_second == 10.0
    assert config.recorders[5].message_type == "geometry_msgs.msg.WrenchStamped"
    assert config.recorders[5].required is True
    assert config.recorders[6].message_type == "sensor_msgs.msg.Imu"
    assert config.recorders[6].required is True
    assert config.recorders[0].fields == [
        "joints_state",
        "robot_pose",
        "robot_space_initial_tip_position",
        "relative_tip_position",
        "robot_space_rcm_position",
        "robot_space_wrench",
        "relative_pivot_RPY",
        "relative_pivot_translation",
    ]
    assert config.recorders[0].csv_header == [
        "joints_state",
        "robot_pose",
        "robot_space_initial_tip_position",
        "relative_tip_position",
        "robot_space_rcm_position",
        "robot_space_wrench",
        "relative_pivot_rpy",
        "relative_pivot_translation",
    ]
    assert config.recorders[1].csv_header == [
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    ]


@pytest.mark.unit
def test_shipped_force_session_ui_uses_home_recordings_folder():
    config = load_config(
        config_class=RecordingUIConfig,
        config_path=FORCE_SESSION_UI_CONFIG,
    )

    assert config.session.output_folder == "~/tso_sensorium_recordings"
    assert config.camera_topic == "/laparoscope/camera/left/image_raw"
    assert config.generation is not None
    assert config.generation.dataset_schema.name == "bowel_retraction"


@pytest.mark.unit
@pytest.mark.parametrize(
    "config_path",
    [TSO_TESTBED_CONFIG, ENDOSCOPE_GUIDANCE_CONFIG, FORCE_SESSION_CONFIG],
)
def test_hardware_recording_configs_preserve_toolkit_kinematics(config_path):
    config = load_config(
        config_class=RecordingSessionConfig,
        config_path=config_path,
    )
    robot_recorder = next(
        recorder
        for recorder in config.recorders
        if recorder.message_type == "testbed_msgs.msg.RobotState"
    )
    camera_transform_recorder = next(
        recorder
        for recorder in config.recorders
        if recorder.topic_name == "/robot_camera_transform"
    )
    combined_video_recorder = next(
        recorder
        for recorder in config.recorders
        if recorder.file_name == "laparoscope_combined"
    )

    assert {
        "robot_space_initial_tip_position",
        "relative_tip_position",
        "relative_pivot_RPY",
        "relative_pivot_translation",
    }.issubset(robot_recorder.fields)
    assert "relative_pivot_rpy" in robot_recorder.csv_header
    assert all(column_name.islower() for column_name in robot_recorder.csv_header)
    assert camera_transform_recorder.csv_header == [
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    ]
    assert combined_video_recorder.frames_per_second == 10.0
    assert combined_video_recorder.lossless_compression is True


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
