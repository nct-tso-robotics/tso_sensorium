"""Tests for tso_sensorium.episodes.generation_config module."""

import importlib.util
import re
from pathlib import Path
from unittest.mock import patch

from tso_sensorium.configuration import load_config
import pytest
from pydantic import ValidationError

from tso_sensorium.episodes.legend import (
    DatasetMetadata,
    PhaseDefinition,
    load_phase_instructions,
)
from tso_sensorium.episodes.generation_config import (
    AnnotationsConfig,
    CsvWriterConfig,
    DatasetGenerationConfig,
    LanguageSource,
    LegendSource,
    LeRobotActionUpdateWriterConfig,
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
def test_legend_asset_reference_uses_the_public_phase_loader() -> None:
    asset_reference = "package://instructions/endoscope_guidance.yaml"
    metadata = DatasetMetadata(
        phase_legend={5: PhaseDefinition(name="stop", instructions=["Hold."])}
    )
    with patch(
        "tso_sensorium.episodes.generation_config.load_phase_legend",
        return_value=metadata,
    ) as load_legend:
        result = AnnotationsConfig.load_legend_reference(value=asset_reference)

    load_legend.assert_called_once_with(config_path=asset_reference)
    assert result.phase_legend[5].instructions == ["Hold."]


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_path,asset_reference",
    [
        (
            ENDOSCOPE_GUIDANCE_CONFIG,
            "package://instructions/endoscope_guidance.yaml",
        ),
        (
            BOWEL_RETRACTION_CONFIG,
            "package://instructions/bowel_retraction_phantom.yaml",
        ),
    ],
)
def test_bundled_generation_uses_same_mapping_as_live_publisher(
    config_path: Path, asset_reference: str, tmp_path: Path
) -> None:
    DatasetMetadata(
        phase_legend={99: PhaseDefinition(name="unrelated", instructions=["Old."])}
    ).save(path=tmp_path / "dataset_metadata.json")
    config = load_config(config_class=DatasetGenerationConfig, config_path=config_path)
    metadata = config.annotations.resolve_legend(recordings_root=tmp_path)
    publisher_phases = load_phase_instructions(config_path=asset_reference)

    assert {
        label: tuple(definition.instructions)
        for label, definition in metadata.phase_legend.items()
    } == publisher_phases


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
def test_shipped_endoscope_guidance_config_derives_moving_frame_actions():
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
    expected_action_columns = [
        "camera_frame_tip_delta_x",
        "camera_frame_tip_delta_y",
        "camera_frame_tip_delta_z",
        "relative_pivot_roll_delta",
    ]
    assert config.dataset_schema.fps == 5
    assert config.dataset_schema.state_columns == expected_pose_columns
    assert config.dataset_schema.action_columns == expected_action_columns
    assert config.states[0].state_file == "robot_state.csv"
    assert "relative_pivot_rpy" in config.states[0].columns
    assert config.states[1].state_file == "robot_camera_transform.csv"
    assert config.states[1].columns == [
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "quaternion_w",
    ]
    assert config.states[2].state_file == "language_instruction.csv"
    assert config.states[2].columns == ["language_instruction"]
    roll_transform = config.table_transforms[1]
    assert roll_transform.column == "relative_pivot_rpy"
    assert roll_transform.output_columns == [
        "relative_pivot_roll",
        "relative_pivot_pitch",
        "relative_pivot_yaw",
    ]
    assert [type(transform).__name__ for transform in config.table_transforms] == [
        "ParseVector3Columns",
        "ParseVector3Columns",
        "ForwardDifferenceColumns",
        "WrappedAngleDifferenceColumns",
        "DropTerminalRows",
        "RotateByQuaternionColumns",
        "RotateByQuaternionColumns",
        "DropColumns",
    ]
    assert config.table_transforms[5].inverse is True
    assert config.table_transforms[6].inverse is True
    assert {
        name: feature.model_dump(mode="json")
        for name, feature in config.dataset_schema.coordinate_frame_features.items()
    } == {
        "camera_frame_tip_position": {
            "columns": expected_pose_columns[:3],
            "frame": "camera",
            "frame_temporality": "moving",
        },
        "camera_frame_tip_delta": {
            "columns": expected_action_columns[:3],
            "frame": "camera",
            "frame_temporality": "moving",
        },
    }
    assert config.dataset_schema.auxiliary_features["task_phase"].dtype == "int64"
    assert sorted(config.annotations.legend.phase_legend) == [0, 1, 2, 3, 4, 5]
    assert config.annotations.language_source == LanguageSource.PHASE_LEGEND
    assert config.annotations.legend_source == LegendSource.CONFIG
    assert {
        label: definition.instructions
        for label, definition in config.annotations.legend.phase_legend.items()
    } == {
        0: ["Zoom in toward the region between the surgical instruments."],
        1: ["Zoom out to obtain a broad view of the operative field."],
        2: ["Rotate the camera until the horizon is level."],
        3: ["Follow the surgical instrument tip along the anatomical contour."],
        4: ["Search the scene for the tip of the surgical instrument."],
        5: ["Do not move the camera."],
    }
    denoising_transform = config.dataset_transforms[0]
    assert denoising_transform.type == "percentile_denoise_columns"
    assert (
        denoising_transform.column_groups["translation"].columns
        == (expected_action_columns[:3])
    )
    assert denoising_transform.column_groups["translation"].percentile == 30.0
    assert denoising_transform.column_groups["roll"].columns == [
        "relative_pivot_roll_delta"
    ]
    assert denoising_transform.column_groups["roll"].percentile == 20.0


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


class TestLeRobotActionUpdateWriterConfig:
    @pytest.mark.unit
    @pytest.mark.skipif(
        importlib.util.find_spec("lerobot") is None,
        reason="requires lerobot",
    )
    def test_requires_dataset_root(self, tmp_path):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "writer.dataset_root is required for LeRobot action update"
            ),
        ):
            LeRobotActionUpdateWriterConfig().build(recordings_root=tmp_path)

    @pytest.mark.unit
    def test_annotation_language_column_defaults_both_lerobot_writers(self):
        annotations = AnnotationsConfig(language_column="phase_instruction")

        full_config = DatasetGenerationConfig(
            writer=LeRobotWriterConfig(output_root="/data/full"),
            annotations=annotations,
        )
        update_config = DatasetGenerationConfig(
            writer=LeRobotActionUpdateWriterConfig(dataset_root="/data/existing"),
            annotations=annotations,
        )

        assert full_config.writer.task_column == "phase_instruction"
        assert update_config.writer.task_column == "phase_instruction"

    @pytest.mark.unit
    def test_explicit_task_column_is_preserved(self):
        config = DatasetGenerationConfig(
            writer=LeRobotActionUpdateWriterConfig(
                dataset_root="/data/existing",
                task_column="explicit_task",
            ),
            annotations=AnnotationsConfig(language_column="phase_instruction"),
        )

        assert config.writer.task_column == "explicit_task"


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
    def test_dot_overrides_select_action_update_and_derive_task_column(self):
        config = load_config(
            config_class=DatasetGenerationConfig,
            config_path=ENDOSCOPE_GUIDANCE_CONFIG,
        )

        updated = apply_generation_overrides(
            config=config,
            overrides={
                "writer.type": "lerobot_action_update",
                "writer.dataset_root": "/data/lerobot",
            },
        )

        assert updated.writer.type == "lerobot_action_update"
        assert updated.writer.dataset_root == "/data/lerobot"
        assert updated.writer.task_column == config.annotations.language_column

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
    def test_config_source_keeps_inline_legend_when_metadata_file_exists(
        self, tmp_path
    ):
        DatasetMetadata(
            dataset_name="edited",
            phase_legend={2: PhaseDefinition(name="metadata", instructions=["old"])},
        ).save(path=tmp_path / "dataset_metadata.json")
        config = AnnotationsConfig(
            legend=DatasetMetadata(
                dataset_name="inline",
                phase_legend={0: PhaseDefinition(name="config", instructions=["new"])},
            ),
            legend_source=LegendSource.CONFIG,
        )

        resolved = config.resolve_legend(recordings_root=tmp_path)

        assert resolved.dataset_name == "inline"
        assert resolved.phase_legend[0].instructions == ["new"]

    @pytest.mark.unit
    def test_config_source_requires_inline_legend(self):
        with pytest.raises(ValidationError) as error_info:
            AnnotationsConfig(legend=None, legend_source=LegendSource.CONFIG)

        assert error_info.value.errors()[0]["msg"] == (
            "Value error, annotations.legend is required when legend_source is 'config'"
        )

    @pytest.mark.unit
    def test_empty_everywhere_gives_empty_metadata(self, tmp_path):
        resolved = AnnotationsConfig().resolve_legend(recordings_root=tmp_path)
        assert resolved.phase_legend == {}
