"""Tests for tso_sensorium.episodes.dataset_builder module."""

import re
from unittest.mock import MagicMock

import pytest

from tso_sensorium.episodes.dataset_builder import (
    BuildCancellationToken,
    BuildPhase,
    BuildProgress,
    DatasetBuilder,
    discover_episode_directories,
)
from tso_sensorium.episodes.schema import (
    CoordinateFrameFeatureMetadata,
    Episode,
    FrameTemporality,
)
from tso_sensorium.export.base import DatasetWriterOperation


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
    def test_writes_schema_coordinate_frame_features_into_metadata(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a"])
        schema = schema_factory(
            coordinate_frame_features={
                "position": CoordinateFrameFeatureMetadata(
                    columns=["x", "y"],
                    frame="robot_base",
                    frame_temporality=FrameTemporality.FIXED,
                )
            }
        )
        writer = MagicMock()
        builder = DatasetBuilder(
            schema=schema,
            writer=writer,
            assemble_episode=lambda directory: Episode(
                name=directory.name, table=episode_table_factory()
            ),
            n_jobs=1,
        )

        builder.build(recordings_root=root)

        metadata = writer.write_metadata.call_args.kwargs["metadata"]
        assert metadata["dataset_name"] == "dataset"
        assert metadata["task"] == "retract the bowel"
        assert metadata["coordinate_frame_features"] == {
            "position": {
                "columns": ["x", "y"],
                "frame": "robot_base",
                "frame_temporality": "fixed",
            }
        }

    @pytest.mark.unit
    def test_applies_dataset_transforms_before_writing_with_provenance(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a"])
        original_episode = Episode(
            name="episode_a",
            table=episode_table_factory(),
        )
        transformed_episode = Episode(
            name="episode_a",
            table=episode_table_factory(),
        )
        transform = MagicMock()
        transform.apply.return_value = [transformed_episode]
        transform.metadata_payload.return_value = {
            "type": "test_transform",
            "threshold": 0.5,
        }
        writer = MagicMock()
        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=lambda directory: original_episode,
            n_jobs=1,
            dataset_transforms=[transform],
        )

        builder.build(recordings_root=root)

        transform.apply.assert_called_once_with(episodes=[original_episode])
        writer.add_episode.assert_called_once_with(episode=transformed_episode)
        metadata = writer.write_metadata.call_args.kwargs["metadata"]
        assert metadata["dataset_transforms"] == [
            {"type": "test_transform", "threshold": 0.5}
        ]

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
    def test_records_writer_failure_under_assembled_episode_name(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(
            directory_names=["source_a", "source_b"],
        )
        writer = MagicMock()
        writer.add_episode.side_effect = [ValueError("invalid frame"), None]
        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=lambda directory: Episode(
                name=f"assembled_{directory.name}",
                table=episode_table_factory(),
            ),
            n_jobs=1,
        )

        report = builder.build(recordings_root=root)

        assert report.failed == {"assembled_source_a": "invalid frame"}
        assert report.written == ["assembled_source_b"]

    @pytest.mark.unit
    def test_reports_completed_work_and_cumulative_failures(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a", "episode_b"])
        writer = MagicMock()
        writer.operation = DatasetWriterOperation.UPDATE
        events: list[BuildProgress] = []

        def assemble(episode_directory):
            if episode_directory.name == "episode_a":
                raise ValueError("invalid recording")
            return Episode(
                name=episode_directory.name,
                table=episode_table_factory(),
            )

        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=assemble,
            n_jobs=1,
            progress_callback=events.append,
        )

        report = builder.build(recordings_root=root)

        assembly_events = [
            event for event in events if event.phase == BuildPhase.ASSEMBLING
        ]
        assert [(event.completed, event.total) for event in assembly_events] == [
            (0, 2),
            (1, 2),
            (2, 2),
        ]
        assert assembly_events[1].failed == {"episode_a": "invalid recording"}
        assert assembly_events[2].failed == {"episode_a": "invalid recording"}
        update_events = [
            event for event in events if event.phase == BuildPhase.UPDATING
        ]
        assert [(event.completed, event.total) for event in update_events] == [
            (0, 1),
            (1, 1),
        ]
        assert events[-1].phase == BuildPhase.COMPLETED
        assert report.cancelled is False

    @pytest.mark.unit
    def test_cancellation_after_write_aborts_without_finalizing(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a", "episode_b"])
        writer = MagicMock()
        writer.operation = DatasetWriterOperation.WRITE
        token = BuildCancellationToken()
        events: list[BuildProgress] = []

        def capture_progress(progress: BuildProgress) -> None:
            events.append(progress)
            if progress.phase == BuildPhase.WRITING and progress.completed == 1:
                token.cancel()

        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=lambda directory: Episode(
                name=directory.name,
                table=episode_table_factory(),
            ),
            n_jobs=1,
            progress_callback=capture_progress,
            cancellation_token=token,
        )

        report = builder.build(recordings_root=root)

        assert report.cancelled is True
        assert report.written == ["episode_a"]
        writer.abort.assert_called_once_with()
        writer.write_metadata.assert_not_called()
        writer.finalize.assert_not_called()
        assert events[-1] == BuildProgress(
            phase=BuildPhase.CANCELLED,
            completed=1,
            total=2,
            current_episode="episode_a",
            failed={},
        )

    @pytest.mark.unit
    def test_writer_exception_aborts_and_propagates(
        self, recordings_root_factory, schema_factory, episode_table_factory
    ):
        root = recordings_root_factory(directory_names=["episode_a"])
        writer = MagicMock()
        writer.add_episode.side_effect = RuntimeError("transaction failed")
        builder = DatasetBuilder(
            schema=schema_factory(),
            writer=writer,
            assemble_episode=lambda directory: Episode(
                name=directory.name,
                table=episode_table_factory(),
            ),
            n_jobs=1,
        )

        with pytest.raises(RuntimeError, match="transaction failed"):
            builder.build(recordings_root=root)

        writer.abort.assert_called_once_with()
        writer.finalize.assert_not_called()

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
