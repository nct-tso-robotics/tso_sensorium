"""Tests for tso_sensorium.episodes.dataset_transforms module."""

import re
from typing import Callable, Dict, List

import pandas as pd
import pytest

from tso_sensorium.episodes.dataset_transforms import (
    PercentileDenoiseColumns,
    PercentileDenoisingColumnGroup,
)
from tso_sensorium.episodes.schema import Episode


@pytest.fixture
def episode_factory() -> Callable[[str, Dict[str, List[float]]], Episode]:
    def factory(name: str, columns: Dict[str, List[float]]) -> Episode:
        return Episode(
            name=name,
            table=pd.DataFrame(data=columns),
            task="move the camera",
        )

    return factory


@pytest.mark.integration
def test_percentile_denoising_uses_shared_nonzero_dataset_thresholds(
    episode_factory: Callable[[str, Dict[str, List[float]]], Episode],
) -> None:
    first_episode = episode_factory(
        "episode_a",
        {
            "dx": [0.0, 1.0, 2.0],
            "dy": [0.0, 0.0, 0.0],
            "roll": [0.0, 0.1, 0.2],
        },
    )
    second_episode = episode_factory(
        "episode_b",
        {
            "dx": [3.0, 4.0],
            "dy": [0.0, 0.0],
            "roll": [0.3, 0.4],
        },
    )
    transform = PercentileDenoiseColumns(
        column_groups={
            "translation": PercentileDenoisingColumnGroup(
                columns=["dx", "dy"],
                percentile=50.0,
            ),
            "roll": PercentileDenoisingColumnGroup(
                columns=["roll"],
                percentile=75.0,
            ),
        }
    )

    transformed = transform.apply(episodes=[first_episode, second_episode])

    assert transformed[0].table["dx"].tolist() == [0.0, 0.0, 0.0]
    assert transformed[1].table["dx"].tolist() == [3.0, 4.0]
    assert transformed[0].table["roll"].tolist() == [0.0, 0.0, 0.0]
    assert transformed[1].table["roll"].tolist() == [0.0, 0.4]
    assert first_episode.table["dx"].tolist() == [0.0, 1.0, 2.0]
    metadata = transform.metadata_payload()
    assert metadata["column_groups"]["translation"]["threshold"] == 2.5
    assert metadata["column_groups"]["roll"]["threshold"] == pytest.approx(0.325)


@pytest.mark.integration
def test_percentile_denoising_zeroes_every_component_in_selected_rows(
    episode_factory: Callable[[str, Dict[str, List[float]]], Episode],
) -> None:
    episode = episode_factory(
        "episode_a",
        {
            "dx": [0.1, 3.0],
            "dy": [0.1, 4.0],
        },
    )
    transform = PercentileDenoiseColumns(
        column_groups={
            "translation": PercentileDenoisingColumnGroup(
                columns=["dx", "dy"],
                percentile=50.0,
            )
        }
    )

    transformed = transform.apply(episodes=[episode])

    assert transformed[0].table[["dx", "dy"]].to_numpy().tolist() == [
        [0.0, 0.0],
        [3.0, 4.0],
    ]


@pytest.mark.integration
def test_percentile_denoising_rejects_group_without_nonzero_movement(
    episode_factory: Callable[[str, Dict[str, List[float]]], Episode],
) -> None:
    transform = PercentileDenoiseColumns(
        column_groups={
            "roll": PercentileDenoisingColumnGroup(
                columns=["roll"],
                percentile=20.0,
            )
        }
    )
    episode = episode_factory("episode_a", {"roll": [0.0, 0.0]})
    expected_message = (
        "Cannot compute denoising threshold for group 'roll': "
        "the dataset contains no nonzero movement"
    )

    with pytest.raises(ValueError, match=re.escape(expected_message)):
        transform.apply(episodes=[episode])


@pytest.mark.unit
def test_percentile_denoising_metadata_requires_applied_transform() -> None:
    transform = PercentileDenoiseColumns(
        column_groups={
            "roll": PercentileDenoisingColumnGroup(
                columns=["roll"],
                percentile=20.0,
            )
        }
    )
    expected_message = "Percentile denoising metadata is unavailable before apply()"

    with pytest.raises(RuntimeError, match=re.escape(expected_message)):
        transform.metadata_payload()
