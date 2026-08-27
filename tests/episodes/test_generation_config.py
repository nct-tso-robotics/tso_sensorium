"""Tests for tso_sensorium.episodes.generation_config module."""

import importlib.util
import re
from pathlib import Path

from tso_sensorium.configuration import load_config
import pytest

from tso_sensorium.episodes.legend import DatasetMetadata, PhaseDefinition
from tso_sensorium.episodes.generation_config import (
    AnnotationsConfig,
    CsvWriterConfig,
    DatasetGenerationConfig,
    LeRobotWriterConfig,
    apply_generation_overrides,
)
from tso_sensorium.export.csv_writer import CsvDatasetWriter

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOWEL_RETRACTION_CONFIG = (
    REPOSITORY_ROOT / "configs" / "dataset" / "bowel_retraction.yaml"
)
ENDOSCOPE_GUIDANCE_CONFIG = (
    REPOSITORY_ROOT / "configs" / "dataset" / "endoscope_guidance.yaml"
)


@pytest.mark.unit
def test_shipped_bowel_retraction_config_decodes():
    config = load_config(
        config_class=DatasetGenerationConfig, config_path=BOWEL_RETRACTION_CONFIG
    )
    assert config.dataset_schema.name == "bowel_retraction"
    assert config.dataset_schema.fps == 30
    assert [camera.name for camera in config.dataset_schema.cameras] == [
        "left",
        "right",
    ]
    assert len(config.videos) == 2
    assert config.videos[0].preprocess[0].side == "left"
    assert [type(t).__name__ for t in config.table_transforms] == [
        "ParseVector3Columns",
        "ParseVector3Columns",
        "SumColumns",
        "FixedTransformToCameraFrame",
        "DropColumns",
    ]
    assert type(config.writer).__name__ == "CsvWriterConfig"
    assert config.annotations.phase_column == "task_phase"
    assert sorted(config.annotations.legend.phase_legend) == [0, 1, 2, 3, 4]
    assert all(
        len(definition.instructions) == 10
        for definition in config.annotations.legend.phase_legend.values()
    )


@pytest.mark.unit
def test_shipped_endoscope_guidance_config_includes_roll_in_pose():
    config = load_config(
        config_class=DatasetGenerationConfig,
        config_path=ENDOSCOPE_GUIDANCE_CONFIG,
    )

    expected_pose_columns = [
        "camera_frame_tip_position_x",
        "camera_frame_tip_position_y",
        "camera_frame_tip_position_z",
        "relative_pivot_roll",
    ]
    assert config.dataset_schema.name == "endoscope_guidance"
    assert config.dataset_schema.fps == 10
    assert config.dataset_schema.state_columns == expected_pose_columns
    assert config.dataset_schema.action_columns == expected_pose_columns
    assert config.states[0].state_file == "camera_robot_state.csv"
    assert "relative_pivot_rpy" in config.states[0].columns
    roll_transform = config.table_transforms[2]
    assert roll_transform.column == "relative_pivot_rpy"
    assert roll_transform.output_columns == [
        "relative_pivot_roll",
        "relative_pivot_pitch",
        "relative_pivot_yaw",
    ]
    assert sorted(config.annotations.legend.phase_legend) == [0, 1, 2, 3, 4, 5]


class TestCsvWriterConfig:
    @pytest.mark.unit
    def test_defaults_to_recordings_root(self, tmp_path):
        writer = CsvWriterConfig().build(recordings_root=tmp_path)
        assert isinstance(writer, CsvDatasetWriter)
        assert writer.output_root == tmp_path

    @pytest.mark.unit
    def test_explicit_output_root_wins(self, tmp_path):
        writer = CsvWriterConfig(output_root=str(tmp_path / "out")).build(
            recordings_root=tmp_path
        )
        assert writer.output_root == tmp_path / "out"


class TestLeRobotWriterConfig:
    @pytest.mark.unit
    @pytest.mark.skipif(
        importlib.util.find_spec("lerobot") is not None,
        reason="lerobot installed; the missing-dependency path cannot trigger",
    )
    def test_missing_lerobot_raises_import_error(self, tmp_path):
        with pytest.raises(
            ImportError,
            match=re.escape(
                "LeRobot export requires the lerobot extra:"
                " pip install 'tso-sensorium[lerobot]'"
            ),
        ):
            LeRobotWriterConfig(output_root="out").build(recordings_root=tmp_path)

    @pytest.mark.unit
    @pytest.mark.skipif(
        importlib.util.find_spec("lerobot") is None,
        reason="requires lerobot",
    )
    def test_requires_output_root(self, tmp_path):
        with pytest.raises(
            ValueError,
            match=re.escape("writer.output_root is required for lerobot export"),
        ):
            LeRobotWriterConfig().build(recordings_root=tmp_path)

    @pytest.mark.unit
    @pytest.mark.skipif(
        importlib.util.find_spec("lerobot") is None,
        reason="requires lerobot",
    )
    def test_repo_id_defaults_to_recordings_root_name(self, tmp_path):
        recordings_root = tmp_path / "bowel_retraction"
        recordings_root.mkdir()
        writer = LeRobotWriterConfig(output_root=str(tmp_path / "out")).build(
            recordings_root=recordings_root
        )
        assert writer.repo_id == "tso/bowel_retraction"


class TestApplyGenerationOverrides:
    @pytest.mark.unit
    def test_scalar_and_list_paths(self):
        config = load_config(
            config_class=DatasetGenerationConfig, config_path=BOWEL_RETRACTION_CONFIG
        )
        updated = apply_generation_overrides(
            config=config,
            overrides={
                "save_frames": True,
                "max_sync_difference_seconds": 0.25,
                "videos.0.frame_column": "customLeftFrame",
            },
        )
        assert updated.save_frames is True
        assert updated.max_sync_difference_seconds == 0.25
        assert updated.videos[0].frame_column == "customLeftFrame"
        assert config.videos[0].frame_column == "frameLeftRectifiedPath"

    @pytest.mark.unit
    def test_writer_replacement_changes_subclass(self):
        config = DatasetGenerationConfig()
        updated = apply_generation_overrides(
            config=config,
            overrides={
                "writer": {
                    "type": "lerobot",
                    "output_root": "/data/lerobot",
                    "repo_id": None,
                    "use_videos": True,
                }
            },
        )
        assert type(updated.writer).__name__ == "LeRobotWriterConfig"
        assert updated.writer.output_root == "/data/lerobot"

    @pytest.mark.unit
    def test_invalid_path_rejected(self):
        with pytest.raises(
            ValueError, match=re.escape("Invalid override path 'missing.field'")
        ):
            apply_generation_overrides(
                config=DatasetGenerationConfig(),
                overrides={"missing.field": 1},
            )


class TestResolveLegend:
    @pytest.mark.unit
    def test_metadata_file_wins_over_inline_legend(self, tmp_path):
        DatasetMetadata(
            dataset_name="edited",
            phase_legend={2: PhaseDefinition(name="new_phase", instructions=["go"])},
        ).save(path=tmp_path / "dataset_metadata.json")
        config = AnnotationsConfig(
            legend=DatasetMetadata(
                dataset_name="inline",
                phase_legend={0: PhaseDefinition(name="old")},
            )
        )
        resolved = config.resolve_legend(recordings_root=tmp_path)
        assert resolved.dataset_name == "edited"
        assert resolved.phase_legend[2].instructions == ["go"]

    @pytest.mark.unit
    def test_inline_legend_seeds_when_no_file(self, tmp_path):
        config = AnnotationsConfig(
            legend=DatasetMetadata(
                dataset_name="inline",
                phase_legend={0: PhaseDefinition(name="old")},
            )
        )
        resolved = config.resolve_legend(recordings_root=tmp_path)
        assert resolved.dataset_name == "inline"

    @pytest.mark.unit
    def test_empty_everywhere_gives_empty_metadata(self, tmp_path):
        resolved = AnnotationsConfig().resolve_legend(recordings_root=tmp_path)
        assert resolved.phase_legend == {}
