"""Tests for tso_sensorium.recording.episode_names module."""

from unittest.mock import patch

import pytest

from tso_sensorium.recording.episode_names import (
    build_episode_name,
    generate_timestamp,
)

DATETIME_PATH = "tso_sensorium.recording.episode_names.datetime.datetime"
GENERATE_TIMESTAMP_PATH = "tso_sensorium.recording.episode_names.generate_timestamp"
TIMESTAMP = "20260827_142355_123456"


@pytest.mark.unit
def test_generate_timestamp_formats_current_time() -> None:
    with patch(DATETIME_PATH) as datetime_mock:
        datetime_mock.now.return_value.strftime.return_value = TIMESTAMP
        result = generate_timestamp()

    datetime_mock.now.assert_called_once_with()
    datetime_mock.now.return_value.strftime.assert_called_once_with("%Y%m%d_%H%M%S_%f")
    assert result == TIMESTAMP


@pytest.mark.unit
@pytest.mark.parametrize(
    "custom_name, expected_name",
    [
        (None, TIMESTAMP),
        ("", TIMESTAMP),
        ("bowel_retraction", f"bowel_retraction_{TIMESTAMP}"),
    ],
)
def test_build_episode_name_always_includes_timestamp(
    custom_name: str | None, expected_name: str
) -> None:
    with patch(
        GENERATE_TIMESTAMP_PATH, return_value=TIMESTAMP
    ) as generate_timestamp_mock:
        result = build_episode_name(custom_name=custom_name)

    generate_timestamp_mock.assert_called_once_with()
    assert result == expected_name
