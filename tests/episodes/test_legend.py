"""Tests for tso_sensorium.episodes.legend module."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tso_sensorium.episodes.legend import (
    DatasetMetadata,
    load_phase_instructions,
    load_phase_legend,
)
from tso_sensorium.episodes.schema import (
    CoordinateFrameFeatureMetadata,
    FrameTemporality,
)

ENDOSCOPE_LEGEND = "package://instructions/endoscope_guidance.yaml"
PHANTOM_LEGEND = "package://instructions/bowel_retraction_phantom.yaml"
PORCINE_LEGEND = "package://instructions/bowel_retraction_porcine.yaml"


@pytest.fixture
def phase_metadata_factory() -> Callable[..., MagicMock]:
    def factory(*, phases: dict[int, list[str]]) -> MagicMock:
        metadata = MagicMock()
        metadata.phase_legend = {
            label: SimpleNamespace(instructions=instructions)
            for label, instructions in phases.items()
        }
        return metadata

    return factory


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


@pytest.mark.unit
@pytest.mark.parametrize("config_path", [Path("custom.yaml"), ENDOSCOPE_LEGEND])
def test_load_phase_legend_resolves_and_validates_with_shared_schema(
    config_path: Path | str,
    phase_metadata_factory: Callable[..., MagicMock],
) -> None:
    payload = {"phase_legend": {2: {"instructions": ["Hold position."]}}}
    metadata = phase_metadata_factory(phases={2: ["Hold position."]})
    with (
        patch(
            "tso_sensorium.episodes.legend.resolve_asset_path",
            return_value="/resolved/legend.yaml",
        ) as resolve_asset,
        patch(
            "tso_sensorium.episodes.legend.load_yaml_with_includes",
            return_value=payload,
        ) as load_yaml,
        patch(
            "tso_sensorium.episodes.legend.DatasetMetadata.model_validate",
            return_value=metadata,
        ) as validate,
    ):
        result = load_phase_legend(config_path=config_path)

    resolve_asset.assert_called_once_with(path=config_path)
    load_yaml.assert_called_once_with(path="/resolved/legend.yaml")
    validate.assert_called_once_with(payload)
    assert result.phase_legend[2].instructions == ["Hold position."]


@pytest.mark.unit
@pytest.mark.parametrize(
    "phases,expected_message",
    [
        ({}, "No phases found in custom.yaml"),
        ({2: []}, "Phases without instructions in custom.yaml: 2"),
        (
            {5: [" "], 2: ["Keep still.", ""]},
            "Phases without instructions in custom.yaml: 2, 5",
        ),
    ],
)
def test_load_phase_legend_rejects_unusable_phases(
    phases: dict[int, list[str]],
    expected_message: str,
    phase_metadata_factory: Callable[..., MagicMock],
) -> None:
    metadata = phase_metadata_factory(phases=phases)
    with (
        patch("tso_sensorium.episodes.legend.resolve_asset_path"),
        patch("tso_sensorium.episodes.legend.load_yaml_with_includes"),
        patch(
            "tso_sensorium.episodes.legend.DatasetMetadata.model_validate",
            return_value=metadata,
        ),
        pytest.raises(ValueError, match=re.escape(expected_message)),
    ):
        load_phase_legend(config_path="custom.yaml")


@pytest.mark.unit
def test_instruction_mapping_uses_the_shared_legend_loader(
    phase_metadata_factory: Callable[..., MagicMock],
) -> None:
    metadata = phase_metadata_factory(phases={2: ["Move.", "Advance."], 5: ["Hold."]})
    with patch(
        "tso_sensorium.episodes.legend.load_phase_legend", return_value=metadata
    ) as load_legend:
        phases = load_phase_instructions(config_path=PHANTOM_LEGEND)

    load_legend.assert_called_once_with(config_path=PHANTOM_LEGEND)
    assert phases == {2: ("Move.", "Advance."), 5: ("Hold.",)}


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_path,expected_counts",
    [
        (ENDOSCOPE_LEGEND, {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}),
        (PHANTOM_LEGEND, {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}),
        (PORCINE_LEGEND, {0: 10, 1: 10, 2: 10, 3: 10, 4: 10}),
    ],
)
def test_packaged_legends_load_without_a_repository_working_directory(
    config_path: str,
    expected_counts: dict[int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    metadata = load_phase_legend(config_path=config_path)
    phases = load_phase_instructions(config_path=config_path)

    assert {label: len(instructions) for label, instructions in phases.items()} == (
        expected_counts
    )
    assert phases == {
        label: tuple(definition.instructions)
        for label, definition in metadata.phase_legend.items()
    }


@pytest.mark.integration
def test_packaged_endoscope_legend_preserves_all_six_instructions() -> None:
    phases = load_phase_instructions(config_path=ENDOSCOPE_LEGEND)

    assert phases == {
        0: ("Zoom in toward the region between the surgical instruments.",),
        1: ("Zoom out to obtain a broad view of the operative field.",),
        2: ("Rotate the camera until the horizon is level.",),
        3: ("Follow the surgical instrument tip along the anatomical contour.",),
        4: ("Search the scene for the tip of the surgical instrument.",),
        5: ("Do not move the camera.",),
    }


@pytest.mark.integration
def test_custom_legend_resolves_includes_relative_to_the_file(tmp_path: Path) -> None:
    (tmp_path / "phases.yaml").write_text(
        "2:\n  name: hold\n  instructions: ['Hold position.']\n"
    )
    config_path = tmp_path / "custom.yaml"
    config_path.write_text("phase_legend: !include phases.yaml\n")

    phases = load_phase_instructions(config_path=config_path)

    assert phases == {2: ("Hold position.",)}
