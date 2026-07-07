"""Tests for tso_sensorium.export.csv_writer module."""

import re
from unittest.mock import patch

import pandas as pd
import pytest

from tso_sensorium.episodes.schema import DatasetSchema, Episode
from tso_sensorium.export.csv_writer import CsvDatasetWriter


@pytest.mark.unit
def test_add_episode_before_open_raises(tmp_path, episode_table_factory):
    writer = CsvDatasetWriter(output_root=tmp_path)
    episode = Episode(name="episode_000", table=episode_table_factory())
    with pytest.raises(
        RuntimeError,
        match=re.escape("Writer must be opened with a schema before adding episodes"),
    ):
        writer.add_episode(episode=episode)


@pytest.mark.unit
def test_add_episode_validates_table_against_schema(
    tmp_path, schema_factory, episode_table_factory
):
    writer = CsvDatasetWriter(output_root=tmp_path)
    writer.open(schema=schema_factory())
    table = episode_table_factory()
    with patch.object(DatasetSchema, "validate_episode_table") as validate:
        writer.add_episode(episode=Episode(name="episode_000", table=table))
    assert validate.call_args.kwargs["table"] is table


@pytest.mark.integration
def test_round_trip_writes_one_folder_per_episode(
    tmp_path, schema_factory, episode_table_factory
):
    writer = CsvDatasetWriter(output_root=tmp_path / "dataset")
    writer.open(schema=schema_factory())
    first_table = episode_table_factory(length=2)
    second_table = episode_table_factory(length=3)
    writer.add_episode(episode=Episode(name="episode_000", table=first_table))
    writer.add_episode(episode=Episode(name="episode_001", table=second_table))
    writer.finalize()

    first_read = pd.read_csv(tmp_path / "dataset" / "episode_000" / "episode.csv")
    second_read = pd.read_csv(tmp_path / "dataset" / "episode_001" / "episode.csv")
    pd.testing.assert_frame_equal(first_read, first_table)
    pd.testing.assert_frame_equal(second_read, second_table)
