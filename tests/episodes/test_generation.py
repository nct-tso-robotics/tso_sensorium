"""Tests for tso_sensorium.episodes.generation module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tso_sensorium.episodes.generation_config import (
    CsvWriterConfig,
    DatasetGenerationConfig,
    LanguageSource,
    StateSourceConfig,
    VideoSourceConfig,
)
from tso_sensorium.episodes.schema import (
    ArmFeature,
    CameraFeature,
    CoordinateFrameFeatureMetadata,
    DatasetSchema,
    FrameTemporality,
)
from tso_sensorium.episodes.table_transforms import (
    DropColumns,
    DropTerminalRows,
    FixedTransformToCameraFrame,
    ForwardDifferenceColumns,
    ParseVector3Columns,
    RotateByQuaternionColumns,
    SumColumns,
    WrappedAngleDifferenceColumns,
)
from tso_sensorium.recording.core import VideoFileWriter
from tso_sensorium.episodes.annotations import (
    EpisodeAnnotations,
    PhaseSegment,
)
import json

from tso_sensorium.episodes.generation import generate_dataset
from tso_sensorium.episodes.generation_config import AnnotationsConfig
from tso_sensorium.episodes.legend import DatasetMetadata, PhaseDefinition

IDENTITY_TRANSFORM = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]
POSITION_COLUMNS = [
    "camera_frame_tip_position_x",
    "camera_frame_tip_position_y",
    "camera_frame_tip_position_z",
]
ACTION_COLUMNS = [
    "camera_frame_tip_delta_x",
    "camera_frame_tip_delta_y",
    "camera_frame_tip_delta_z",
    "relative_pivot_roll_delta",
]


def _write_synthetic_recording(episode_directory: Path, frame_count: int) -> None:
    episode_directory.mkdir(parents=True)
    frame_rng = np.random.default_rng(seed=7)
    timestamps = [int(index * 1e8) for index in range(frame_count)]
    for camera in ("left", "right"):
        video_writer = VideoFileWriter(
            output_folder=episode_directory,
            file_name=f"laparoscope_{camera}",
            frames_per_second=30.0,
        )
        for timestamp_nanoseconds in timestamps:
            video_writer.write_frame(
                frame=frame_rng.integers(
                    0,
                    255,
                    size=(540, 960, 3),
                    dtype=np.uint8,
                ),
                timestamp_nanoseconds=timestamp_nanoseconds,
            )
        video_writer.close()
        pd.DataFrame({"time": timestamps}).to_csv(
            episode_directory / f"laparoscope_{camera}.csv", index=False
        )
    pd.DataFrame(
        {
            "time": timestamps,
            "relative_tip_position": [
                f"x: {0.01 * index}\ny: 0.0\nz: 0.0" for index in range(frame_count)
            ],
            "robot_space_initial_tip_position": [
                "x: 0.1\ny: 0.2\nz: 0.3" for _ in range(frame_count)
            ],
        }
    ).to_csv(episode_directory / "ur5e.csv", index=False)


def _synthetic_config(recordings_root: Path) -> DatasetGenerationConfig:
    return DatasetGenerationConfig(
        recordings_root=str(recordings_root),
        schema=DatasetSchema(
            name="synthetic",
            fps=30,
            cameras=[
                CameraFeature(
                    name="left",
                    frame_column="frameLeftRectifiedPath",
                    height=540,
                    width=960,
                ),
                CameraFeature(
                    name="right",
                    frame_column="frameRightRectifiedPath",
                    height=540,
                    width=960,
                ),
            ],
            arms=[
                ArmFeature(
                    name="tool",
                    state_columns=POSITION_COLUMNS,
                    action_columns=POSITION_COLUMNS,
                )
            ],
            task="synthetic task",
        ),
        videos=[
            VideoSourceConfig(
                video_file="laparoscope_left.mp4",
                timestamps_file="laparoscope_left.csv",
                frame_column="frameLeftRectifiedPath",
                frames_directory="framesLeftRectified",
            ),
            VideoSourceConfig(
                video_file="laparoscope_right.mp4",
                timestamps_file="laparoscope_right.csv",
                frame_column="frameRightRectifiedPath",
                frames_directory="framesRightRectified",
            ),
        ],
        states=[
            StateSourceConfig(
                state_file="ur5e.csv",
                columns=[
                    "relative_tip_position",
                    "robot_space_initial_tip_position",
                ],
            )
        ],
        table_transforms=[
            ParseVector3Columns(
                column="relative_tip_position",
                output_columns=[
                    "relative_tip_position_x",
                    "relative_tip_position_y",
                    "relative_tip_position_z",
                ],
            ),
            ParseVector3Columns(
                column="robot_space_initial_tip_position",
                output_columns=[
                    "robot_space_initial_tip_position_x",
                    "robot_space_initial_tip_position_y",
                    "robot_space_initial_tip_position_z",
                ],
            ),
            SumColumns(
                first_columns=[
                    "robot_space_initial_tip_position_x",
                    "robot_space_initial_tip_position_y",
                    "robot_space_initial_tip_position_z",
                ],
                second_columns=[
                    "relative_tip_position_x",
                    "relative_tip_position_y",
                    "relative_tip_position_z",
                ],
                output_columns=[
                    "base_frame_tip_position_x",
                    "base_frame_tip_position_y",
                    "base_frame_tip_position_z",
                ],
            ),
            FixedTransformToCameraFrame(
                camera_to_base=IDENTITY_TRANSFORM,
                position_columns=[
                    "base_frame_tip_position_x",
                    "base_frame_tip_position_y",
                    "base_frame_tip_position_z",
                ],
                output_columns=POSITION_COLUMNS,
                quaternion_output_columns=[
                    "base_to_camera_quat_x",
                    "base_to_camera_quat_y",
                    "base_to_camera_quat_z",
                    "base_to_camera_quat_w",
                ],
                translation_output_columns=[
                    "base_to_camera_trans_x",
                    "base_to_camera_trans_y",
                    "base_to_camera_trans_z",
                ],
            ),
            DropColumns(columns=["relative_tip_position"]),
        ],
        writer=CsvWriterConfig(),
        save_frames=True,
        n_jobs=1,
    )


def _moving_camera_config(recordings_root: Path) -> DatasetGenerationConfig:
    quaternion_columns = [
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    ]
    return DatasetGenerationConfig(
        recordings_root=str(recordings_root),
        schema=DatasetSchema(
            name="moving_camera",
            fps=5,
            cameras=[
                CameraFeature(
                    name="left",
                    frame_column="frameLeftRectifiedPath",
                    height=540,
                    width=960,
                ),
                CameraFeature(
                    name="right",
                    frame_column="frameRightRectifiedPath",
                    height=540,
                    width=960,
                ),
            ],
            arms=[
                ArmFeature(
                    name="camera",
                    state_columns=POSITION_COLUMNS + ["relative_pivot_roll"],
                    action_columns=ACTION_COLUMNS,
                )
            ],
            coordinate_frame_features={
                "camera_frame_tip_position": CoordinateFrameFeatureMetadata(
                    columns=POSITION_COLUMNS,
                    frame="camera",
                    frame_temporality=FrameTemporality.MOVING,
                ),
                "camera_frame_tip_delta": CoordinateFrameFeatureMetadata(
                    columns=ACTION_COLUMNS[:3],
                    frame="camera",
                    frame_temporality=FrameTemporality.MOVING,
                ),
            },
            task="move the camera",
        ),
        videos=[
            VideoSourceConfig(
                video_file="laparoscope_left.mp4",
                timestamps_file="laparoscope_left.csv",
                frame_column="frameLeftRectifiedPath",
                frames_directory="framesLeftRectified",
            ),
            VideoSourceConfig(
                video_file="laparoscope_right.mp4",
                timestamps_file="laparoscope_right.csv",
                frame_column="frameRightRectifiedPath",
                frames_directory="framesRightRectified",
            ),
        ],
        states=[
            StateSourceConfig(
                state_file="ur5e.csv",
                columns=["relative_tip_position", "relative_pivot_rpy"],
            ),
            StateSourceConfig(
                state_file="robot_camera_transform.csv",
                columns=quaternion_columns,
            ),
        ],
        table_transforms=[
            ParseVector3Columns(
                column="relative_tip_position",
                output_columns=[
                    "relative_tip_position_x",
                    "relative_tip_position_y",
                    "relative_tip_position_z",
                ],
            ),
            ParseVector3Columns(
                column="relative_pivot_rpy",
                output_columns=[
                    "relative_pivot_roll",
                    "relative_pivot_pitch",
                    "relative_pivot_yaw",
                ],
            ),
            ForwardDifferenceColumns(
                columns=[
                    "relative_tip_position_x",
                    "relative_tip_position_y",
                    "relative_tip_position_z",
                ],
                output_columns=["base_dx", "base_dy", "base_dz"],
            ),
            WrappedAngleDifferenceColumns(
                columns=["relative_pivot_roll"],
                output_columns=["relative_pivot_roll_delta"],
            ),
            DropTerminalRows(count=1),
            RotateByQuaternionColumns(
                point_columns=[
                    "relative_tip_position_x",
                    "relative_tip_position_y",
                    "relative_tip_position_z",
                ],
                quaternion_columns=quaternion_columns,
                output_columns=POSITION_COLUMNS,
                inverse=True,
            ),
            RotateByQuaternionColumns(
                point_columns=["base_dx", "base_dy", "base_dz"],
                quaternion_columns=quaternion_columns,
                output_columns=ACTION_COLUMNS[:3],
                inverse=True,
            ),
            DropColumns(
                columns=[
                    "relative_tip_position",
                    "relative_pivot_rpy",
                    "relative_pivot_pitch",
                    "relative_pivot_yaw",
                ]
            ),
        ],
        writer=CsvWriterConfig(),
        save_frames=True,
        n_jobs=1,
    )


@pytest.mark.unit
def test_generation_requires_recordings_root():
    with pytest.raises(ValueError, match="recordings_root is required"):
        generate_dataset(config=DatasetGenerationConfig())


@pytest.mark.integration
def test_end_to_end_generates_csv_episode_from_recording(tmp_path):
    recordings_root = tmp_path / "recordings"
    _write_synthetic_recording(
        episode_directory=recordings_root / "episode_000", frame_count=4
    )

    report = generate_dataset(config=_synthetic_config(recordings_root=recordings_root))

    assert report.written == ["episode_000"]
    assert report.failed == {}
    episode_table = pd.read_csv(recordings_root / "episode_000" / "episode.csv")
    assert len(episode_table) == 4
    for column in POSITION_COLUMNS + [
        "frameLeftRectifiedPath",
        "frameRightRectifiedPath",
        "base_to_camera_quat_x",
    ]:
        assert column in episode_table.columns
    assert "relative_tip_position" not in episode_table.columns
    # Identity camera transform: camera positions equal base-frame sums.
    np.testing.assert_allclose(
        episode_table["camera_frame_tip_position_x"].tolist(),
        [0.1 + 0.01 * index for index in range(4)],
    )
    left_frames = list(
        (recordings_root / "episode_000" / "framesLeftRectified").glob("*.png")
    )
    assert len(left_frames) == 4


@pytest.mark.integration
def test_end_to_end_aligns_recorded_language_into_episode_csv(tmp_path):
    recordings_root = tmp_path / "recordings"
    episode_directory = recordings_root / "episode_000"
    episode_directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "time": [0, int(1e8), int(2e8)],
            "position": [0.0, 1.0, 2.0],
        }
    ).to_csv(episode_directory / "robot_state.csv", index=False)
    pd.DataFrame(
        {
            "time": [0, int(2e8)],
            "language_instruction": ["Zoom in.", "Do not move."],
        }
    ).to_csv(episode_directory / "language_instruction.csv", index=False)
    config = DatasetGenerationConfig(
        recordings_root=str(recordings_root),
        schema=DatasetSchema(
            name="language_alignment",
            fps=5,
            cameras=[],
            arms=[],
        ),
        states=[
            StateSourceConfig(
                state_file="robot_state.csv",
                columns=["position"],
            ),
            StateSourceConfig(
                state_file="language_instruction.csv",
                columns=["language_instruction"],
            ),
        ],
        writer=CsvWriterConfig(),
        n_jobs=1,
    )

    report = generate_dataset(config=config)

    assert report.written == ["episode_000"]
    episode_table = pd.read_csv(episode_directory / "episode.csv")
    assert episode_table["language_instruction"].tolist() == [
        "Zoom in.",
        "Zoom in.",
        "Do not move.",
    ]


@pytest.mark.integration
def test_end_to_end_derives_actions_in_transition_start_camera_frame(tmp_path):
    recordings_root = tmp_path / "recordings"
    episode_directory = recordings_root / "episode_000"
    _write_synthetic_recording(episode_directory=episode_directory, frame_count=3)
    timestamps = [0, int(1e8), int(2e8)]
    pd.DataFrame(
        {
            "time": timestamps,
            "relative_tip_position": [
                "x: 1.0\ny: 0.0\nz: 0.0",
                "x: 1.0\ny: 1.0\nz: 0.0",
                "x: 2.0\ny: 1.0\nz: 0.0",
            ],
            "relative_pivot_rpy": [
                f"x: {np.pi - 0.1}\ny: 0.0\nz: 0.0",
                f"x: {-np.pi + 0.2}\ny: 0.0\nz: 0.0",
                f"x: {-np.pi + 0.4}\ny: 0.0\nz: 0.0",
            ],
        }
    ).to_csv(episode_directory / "ur5e.csv", index=False)
    pd.DataFrame(
        {
            "time": timestamps,
            "quaternion_x": [0.0, 0.0, 0.0],
            "quaternion_y": [0.0, 0.0, 0.0],
            "quaternion_z": [np.sqrt(0.5), 0.0, 0.0],
            "quaternion_w": [np.sqrt(0.5), 1.0, 1.0],
        }
    ).to_csv(episode_directory / "robot_camera_transform.csv", index=False)

    report = generate_dataset(
        config=_moving_camera_config(recordings_root=recordings_root)
    )

    assert report.written == ["episode_000"]
    episode_table = pd.read_csv(episode_directory / "episode.csv")
    assert len(episode_table) == 2
    np.testing.assert_allclose(
        episode_table[POSITION_COLUMNS].to_numpy(),
        [[0.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        episode_table[ACTION_COLUMNS[:3]].to_numpy(),
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        episode_table["relative_pivot_roll_delta"].to_numpy(),
        [0.3, 0.2],
        atol=1e-7,
    )
    metadata = json.loads((recordings_root / "dataset_metadata.json").read_text())
    assert metadata["coordinate_frame_features"] == {
        "camera_frame_tip_delta": {
            "columns": ACTION_COLUMNS[:3],
            "frame": "camera",
            "frame_temporality": "moving",
        },
        "camera_frame_tip_position": {
            "columns": POSITION_COLUMNS,
            "frame": "camera",
            "frame_temporality": "moving",
        },
    }


@pytest.mark.integration
def test_end_to_end_applies_annotations(tmp_path):
    recordings_root = tmp_path / "recordings"
    _write_synthetic_recording(
        episode_directory=recordings_root / "episode_000", frame_count=4
    )
    EpisodeAnnotations(
        segments=[
            PhaseSegment(start=0, end=int(2e8), phase=0),
            PhaseSegment(
                start=int(2e8),
                end=int(4e8),
                phase=1,
                language="pull back now",
            ),
        ]
    ).save(path=recordings_root / "episode_000" / "annotations.json")

    config = _synthetic_config(recordings_root=recordings_root)
    config.annotations = AnnotationsConfig(
        legend=DatasetMetadata(
            dataset_name="synthetic",
            task="synthetic task",
            phase_legend={
                0: PhaseDefinition(
                    name="approach", instructions=["move to the target"]
                ),
                1: PhaseDefinition(name="retract", instructions=["retreat"]),
            },
        )
    )
    report = generate_dataset(config=config)

    assert report.written == ["episode_000"]
    episode_table = pd.read_csv(recordings_root / "episode_000" / "episode.csv")
    assert episode_table["phase"].tolist() == [0, 0, 1, 1]
    metadata = json.loads((recordings_root / "dataset_metadata.json").read_text())
    assert metadata["phase_legend"]["0"]["name"] == "approach"
    assert metadata["generation"]["written"] == ["episode_000"]
    assert episode_table["language_instruction"].tolist() == [
        "move to the target",
        "move to the target",
        "pull back now",
        "pull back now",
    ]


@pytest.mark.integration
def test_end_to_end_can_use_phase_legend_instead_of_segment_language(tmp_path):
    recordings_root = tmp_path / "recordings"
    episode_directory = recordings_root / "episode_000"
    _write_synthetic_recording(
        episode_directory=episode_directory,
        frame_count=2,
    )
    EpisodeAnnotations(
        segments=[
            PhaseSegment(
                start=0,
                end=int(2e8),
                phase=0,
                language="legacy variant",
            )
        ]
    ).save(path=episode_directory / "annotations.json")
    config = _synthetic_config(recordings_root=recordings_root)
    config.annotations = AnnotationsConfig(
        legend=DatasetMetadata(
            dataset_name="synthetic",
            task="synthetic task",
            phase_legend={
                0: PhaseDefinition(
                    name="approach",
                    instructions=["move to the target"],
                )
            },
        ),
        language_source=LanguageSource.PHASE_LEGEND,
    )

    report = generate_dataset(config=config)

    assert report.written == ["episode_000"]
    episode_table = pd.read_csv(episode_directory / "episode.csv")
    assert episode_table["language_instruction"].tolist() == [
        "move to the target",
        "move to the target",
    ]


@pytest.mark.integration
def test_generation_without_annotations_preserves_curated_metadata(tmp_path):
    from tso_sensorium.episodes.legend import DatasetMetadata, PhaseDefinition

    recordings_root = tmp_path / "recordings"
    _write_synthetic_recording(
        episode_directory=recordings_root / "episode_000", frame_count=4
    )
    DatasetMetadata(
        dataset_name="curated",
        phase_legend={0: PhaseDefinition(name="approach", instructions=["go"])},
    ).save(path=recordings_root / "dataset_metadata.json")

    config = _synthetic_config(recordings_root=recordings_root)
    config.annotations = None
    generate_dataset(config=config)

    reloaded = DatasetMetadata.load(path=recordings_root / "dataset_metadata.json")
    assert reloaded.dataset_name == "curated"
    assert reloaded.phase_legend[0].name == "approach"


@pytest.mark.unit
def test_duplicate_frames_directory_rejected(tmp_path):
    from tso_sensorium.episodes.generation_config import VideoSourceConfig

    config = _synthetic_config(recordings_root=tmp_path)
    config.save_frames = True
    config.videos = [
        VideoSourceConfig(
            video_file="a.mp4", timestamps_file="a.csv", frame_column="a"
        ),
        VideoSourceConfig(
            video_file="b.mp4", timestamps_file="b.csv", frame_column="b"
        ),
    ]
    with pytest.raises(ValueError, match="distinct frames_directory"):
        generate_dataset(config=config)
