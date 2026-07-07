"""Tests for tso_sensorium.episodes.dataset_builder module."""

import re
from unittest.mock import MagicMock

import pytest

from tso_sensorium.episodes.dataset_builder import (
    DatasetBuilder,
    discover_episode_directories,
)
from tso_sensorium.episodes.schema import Episode


@pytest.fixture
def recordings_root_factory(tmp_path):
    def factory(directory_names, file_names=()):
        for directory_name in directory_names:
            (tmp_path / directory_name).mkdir()
        for file_name in file_names:
            (tmp_path / file_name).touch()
        return tmp_path

    return factory


class TestDiscoverEpisodeDirectories:
    @pytest.mark.unit
    def test_returns_sorted_directories_ignoring_files(self, recordings_root_factory):
        root = recordings_root_factory(
            directory_names=["episode_b", "episode_a"],
            file_names=["notes.txt"],
        )
        directories = discover_episode_directories(recordings_root=root)
        assert [path.name for path in directories] == ["episode_a", "episode_b"]

    @pytest.mark.unit
    def test_excludes_directories_by_substring(self, recordings_root_factory):
        root = recordings_root_factory(directory_names=["episode_a", "cache.zarr"])
        directories = discover_episode_directories(
            recordings_root=root, exclude_substrings=[".zarr"]
        )
        assert [path.name for path in directories] == ["episode_a"]

    @pytest.mark.unit
    def test_missing_root_raises(self, tmp_path):
        missing_root = tmp_path / "missing"
        with pytest.raises(
            NotADirectoryError,
            match=re.escape(f"Recordings root not found: {missing_root}"),
        ):
            discover_episode_directories(recordings_root=missing_root)


class TestDatasetBuilder:
    @pytest.mark.unit
    def test_streams_assembled_episodes_to_writer_in_order(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_b", "episode_a"])
        schema = schema_factory()
        writer = MagicMock()

        def assemble(episode_directory):
            return Episode(name=episode_directory.name, table=episode_table_factory())

        builder = DatasetBuilder(
            schema=schema, writer=writer, assemble_episode=assemble, n_jobs=1
        )
        report = builder.build(recordings_root=root)

        writer.open.assert_called_once_with(schema=schema)
        added_names = [
            call.kwargs["episode"].name for call in writer.add_episode.call_args_list
        ]
        assert added_names == ["episode_a", "episode_b"]
        writer.finalize.assert_called_once_with()
        assert report.written == ["episode_a", "episode_b"]
        assert report.failed == {}

    @pytest.mark.unit
    def test_discards_failing_episodes_with_reason(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a", "episode_b"])
        writer = MagicMock()

        def assemble(episode_directory):
            if episode_directory.name == "episode_a":
                raise ValueError("sync gap too large")
            return Episode(name=episode_directory.name, table=episode_table_factory())

        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=assemble,
            n_jobs=1,
        )
        report = builder.build(recordings_root=root)

        assert report.failed == {"episode_a": "sync gap too large"}
        assert report.written == ["episode_b"]
        assert writer.add_episode.call_count == 1

    @pytest.mark.unit
    def test_skips_excluded_directories(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a", "cache.zarr"])
        writer = MagicMock()
        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=lambda directory: Episode(
                name=directory.name, table=episode_table_factory()
            ),
            n_jobs=1,
            exclude_substrings=[".zarr"],
        )
        report = builder.build(recordings_root=root)
        assert report.written == ["episode_a"]
