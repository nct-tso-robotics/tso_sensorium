"""Tests for tso_sensorium.recording.denoising_preview module."""

import re
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from tso_sensorium.episodes.dataset_transforms import (
    PercentileDenoiseColumns,
    PercentileDenoisingColumnGroup,
)
from tso_sensorium.episodes.generation_config import (
    AnnotationsConfig,
    DatasetGenerationConfig,
)
from tso_sensorium.episodes.schema import Episode
from tso_sensorium.recording.denoising_preview import (
    build_denoising_preview_data,
    configured_denoising_groups,
    load_denoising_preview_data,
)

TRANSLATION_OVERRIDE_PATH = "dataset_transforms.0.column_groups.translation.percentile"


@pytest.fixture
def denoising_config_factory():
    def factory(percentile=50.0, with_phases=True):
        annotations = (
            AnnotationsConfig(phase_column="task_phase") if with_phases else None
        )
        return DatasetGenerationConfig(
            annotations=annotations,
            dataset_transforms=[
                PercentileDenoiseColumns(
                    column_groups={
                        "translation": PercentileDenoisingColumnGroup(
                            columns=["dx", "dy", "dz"],
                            percentile=percentile,
                        )
                    }
                )
            ],
        )

    return factory


@pytest.fixture
def episode_factory():
    def factory(name="episode", magnitudes=None, phases=None):
        configured_magnitudes = magnitudes or [0.0, 1.0, 2.0, 3.0, 4.0]
        configured_phases = phases or [0, 0, 1, 1, 1]
        return Episode(
            name=name,
            table=pd.DataFrame(
                {
                    "dx": configured_magnitudes,
                    "dy": [0.0] * len(configured_magnitudes),
                    "dz": [0.0] * len(configured_magnitudes),
                    "task_phase": configured_phases,
                }
            ),
        )

    return factory


@pytest.mark.unit
def test_discovers_generation_override_paths(denoising_config_factory):
    config = denoising_config_factory(percentile=37.5, with_phases=False)

    groups = configured_denoising_groups(config=config)

    assert groups[0].name == "translation"
    assert groups[0].columns == ("dx", "dy", "dz")
    assert groups[0].percentile == 37.5
    assert groups[0].override_path == TRANSLATION_OVERRIDE_PATH


@pytest.mark.integration
def test_preview_reports_exact_global_and_per_phase_zeroing(
    denoising_config_factory,
    episode_factory,
):
    preview = build_denoising_preview_data(
        config=denoising_config_factory(percentile=50.0, with_phases=True),
        episodes=[episode_factory()],
    )

    payload = preview.to_payload(phase_names={0: "stop", 1: "focus"})

    group = payload["groups"][0]
    assert group["threshold"] == 2.5
    histogram = group["histogram"]
    assert len(histogram["bin_edges"]) == 101
    assert len(histogram["raw_counts"]) == 100
    assert len(histogram["denoised_counts"]) == 100
    assert sum(histogram["raw_counts"]) == 4
    assert sum(histogram["denoised_counts"]) == 2
    assert histogram["log10_threshold"] == pytest.approx(np.log10(2.5))
    assert group["overall"] == {
        "sample_count": 5,
        "existing_zero_count": 1,
        "newly_zeroed_count": 2,
        "zeroed_count": 3,
        "zeroed_percent": 60.0,
    }
    assert group["phases"] == [
        {
            "label": 0,
            "name": "stop",
            "sample_count": 2,
            "existing_zero_count": 1,
            "newly_zeroed_count": 1,
            "zeroed_count": 2,
            "zeroed_percent": 100.0,
        },
        {
            "label": 1,
            "name": "focus",
            "sample_count": 3,
            "existing_zero_count": 0,
            "newly_zeroed_count": 1,
            "zeroed_count": 1,
            "zeroed_percent": 100.0 / 3.0,
        },
    ]


@pytest.mark.integration
def test_preview_recomputes_threshold_from_override_without_reassembly(
    denoising_config_factory,
    episode_factory,
):
    preview = build_denoising_preview_data(
        config=denoising_config_factory(percentile=50.0, with_phases=False),
        episodes=[episode_factory()],
    )

    payload = preview.to_payload(percentiles={TRANSLATION_OVERRIDE_PATH: 75.0})

    group = payload["groups"][0]
    assert group["percentile"] == 75.0
    assert group["threshold"] == 3.25
    assert sum(group["histogram"]["raw_counts"]) == 4
    assert sum(group["histogram"]["denoised_counts"]) == 1
    assert group["overall"]["newly_zeroed_count"] == 3
    assert group["overall"]["zeroed_count"] == 4


@pytest.mark.parametrize(
    "percentiles, expected_error",
    [
        (
            {"dataset_transforms.4.column_groups.unknown.percentile": 10.0},
            "Unknown denoising percentile override paths: "
            "['dataset_transforms.4.column_groups.unknown.percentile']",
        ),
        (
            {TRANSLATION_OVERRIDE_PATH: -1.0},
            "Denoising percentile for "
            f"'{TRANSLATION_OVERRIDE_PATH}' must be a finite number in "
            "[0, 100], got -1.0",
        ),
    ],
)
@pytest.mark.unit
def test_preview_rejects_invalid_percentile_overrides(
    denoising_config_factory,
    episode_factory,
    percentiles,
    expected_error,
):
    preview = build_denoising_preview_data(
        config=denoising_config_factory(percentile=50.0, with_phases=False),
        episodes=[episode_factory()],
    )

    with pytest.raises(ValueError, match=re.escape(expected_error)):
        preview.to_payload(percentiles=percentiles)


@pytest.mark.unit
def test_load_assembles_each_discovered_episode_and_reports_failures(
    denoising_config_factory,
    episode_factory,
    tmp_path,
):
    config = denoising_config_factory(percentile=50.0, with_phases=False)
    episode_directories = [tmp_path / "valid", tmp_path / "invalid"]
    valid_episode = episode_factory()
    assemble_episode = MagicMock(
        side_effect=[valid_episode, ValueError("missing state.csv")]
    )

    with (
        patch(
            "tso_sensorium.recording.denoising_preview.discover_episode_directories",
            return_value=episode_directories,
        ) as discover,
        patch(
            "tso_sensorium.recording.denoising_preview.create_episode_assembler",
            return_value=assemble_episode,
        ) as create_assembler,
    ):
        preview = load_denoising_preview_data(
            config=config,
            recordings_root=tmp_path,
        )

    discover.assert_called_once_with(
        recordings_root=tmp_path,
        exclude_substrings=config.exclude_directory_substrings,
    )
    create_assembler.assert_called_once_with(config=config)
    assert assemble_episode.call_args_list == [
        call(Path(episode_directories[0])),
        call(Path(episode_directories[1])),
    ]
    assert preview.episode_count == 1
    assert preview.failed == {"invalid": "missing state.csv"}
