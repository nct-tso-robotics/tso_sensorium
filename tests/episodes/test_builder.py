"""Tests for tso_sensorium.episodes.builder module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tso_sensorium.episodes.builder import (
    DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    EpisodeGenerator,
)

STATE_DATA_PATH = "tso_sensorium.episodes.builder.StateData"
VIDEO_DATA_PATH = "tso_sensorium.episodes.builder.VideoData"


@pytest.fixture
def generated_episode_factory():
    def factory(dataframe: pd.DataFrame) -> EpisodeGenerator:
        generator = EpisodeGenerator()
        generator.dataset = dataframe
        return generator

    return factory


class TestSourceRegistration:
    @pytest.mark.unit
    def test_add_state_registers_state_source_with_configuration(self):
        generator = EpisodeGenerator()
        with patch(STATE_DATA_PATH) as state_data_class:
            chained = generator.add_state(
                state_data_path="state.csv",
                sync_col_name="timestamp",
                dataset_cols=["x", "y"],
            )
        state_data_class.assert_called_once_with(
            state_data_path="state.csv",
            sync_col_name="timestamp",
            dataset_cols=["x", "y"],
            max_sync_difference_seconds=DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
        )
        assert generator.data == [state_data_class.return_value]
        assert chained is generator

    @pytest.mark.unit
    def test_add_video_registers_video_source_with_configuration(self):
        generator = EpisodeGenerator()
        preprocess_fn = MagicMock()
        with patch(VIDEO_DATA_PATH) as video_data_class:
            chained = generator.add_video(
                video_path="left.mp4",
                timestamps_path="timestamps.csv",
                sync_col_name="timestamp",
                frames_output_path="frames",
                frame_col_name="left_frame",
                preprocess_fn=preprocess_fn,
                save_frames=True,
            )
        video_data_class.assert_called_once_with(
            video_path="left.mp4",
            timestamps_path="timestamps.csv",
            sync_col_name="timestamp",
            frames_output_path="frames",
            frame_col_name="left_frame",
            preprocess_fn=preprocess_fn,
            save_frames=True,
            max_sync_difference_seconds=DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
        )
        assert generator.data == [video_data_class.return_value]
        assert chained is generator


class TestGenerateDataset:
    @pytest.mark.unit
    def test_trims_reference_timestamps_to_shared_source_range(self):
        reference_sync_series = pd.Series([-10, 0, 10, 20])
        shared_sync_series = pd.Series([0, 10])
        first_source = MagicMock()
        first_source.sync_col_name = "time"
        first_source.get_sync_col_data.return_value = reference_sync_series
        first_source.get_data.return_value = pd.DataFrame({"a": [1, 2]})
        second_source = MagicMock()
        second_source.get_sync_col_data.return_value = shared_sync_series
        second_source.get_data.return_value = pd.DataFrame({"b": [3, 4]})
        generator = EpisodeGenerator()
        generator.data = [first_source, second_source]

        generator.generate_dataset()

        first_source.get_sync_col_data.assert_called_once_with()
        second_source.get_sync_col_data.assert_called_once_with()
        first_sync = first_source.get_data.call_args.kwargs["source_sync_dataframe"]
        second_sync = second_source.get_data.call_args.kwargs["source_sync_dataframe"]
        pd.testing.assert_series_equal(first_sync, shared_sync_series)
        pd.testing.assert_series_equal(second_sync, shared_sync_series)
        pd.testing.assert_frame_equal(
            generator.dataset,
            pd.DataFrame({"time": [0, 10], "a": [1, 2], "b": [3, 4]}),
        )

    @pytest.mark.unit
    def test_rejects_sources_without_an_overlapping_timestamp_range(self):
        first_source = MagicMock()
        first_source.get_sync_col_data.return_value = pd.Series([0, 10])
        second_source = MagicMock()
        second_source.get_sync_col_data.return_value = pd.Series([20, 30])
        generator = EpisodeGenerator()
        generator.data = [first_source, second_source]

        with pytest.raises(
            ValueError,
            match=(
                "Recorded data sources do not have an overlapping timestamp range."
                " Discarding episode."
            ),
        ):
            generator.generate_dataset()

        first_source.get_data.assert_not_called()
        second_source.get_data.assert_not_called()

    @pytest.mark.unit
    def test_rejects_generation_without_data_sources(self):
        with pytest.raises(
            ValueError,
            match="Cannot generate a dataset without data sources",
        ):
            EpisodeGenerator().generate_dataset()

    @pytest.mark.integration
    def test_aligns_staggered_video_and_state_recordings(self, tmp_path):
        camera_timestamps = [
            0,
            100_000_000,
            200_000_000,
            300_000_000,
            400_000_000,
            500_000_000,
            600_000_000,
        ]
        robot_timestamps = list(range(400_000_000, 602_000_000, 2_000_000))
        camera_path = tmp_path / "camera.csv"
        robot_path = tmp_path / "robot.csv"
        pd.DataFrame({"time": camera_timestamps}).to_csv(camera_path, index=False)
        pd.DataFrame({"time": robot_timestamps, "position": robot_timestamps}).to_csv(
            robot_path, index=False
        )
        generator = EpisodeGenerator()
        generator.add_video(
            video_path=tmp_path / "camera.avi",
            timestamps_path=camera_path,
            sync_col_name="time",
            frames_output_path=tmp_path / "frames",
            frame_col_name="frame",
        ).add_state(
            state_data_path=robot_path,
            sync_col_name="time",
            dataset_cols=["position"],
        )

        generator.generate_dataset()

        pd.testing.assert_frame_equal(
            generator.dataset,
            pd.DataFrame(
                {
                    "time": [400_000_000, 500_000_000, 600_000_000],
                    "frame": [
                        tmp_path / "frames" / "4.png",
                        tmp_path / "frames" / "5.png",
                        tmp_path / "frames" / "6.png",
                    ],
                    "position": [400_000_000, 500_000_000, 600_000_000],
                }
            ),
        )


class TestColumnOperations:
    @pytest.mark.unit
    def test_apply_function_single_column(self, generated_episode_factory):
        generator = generated_episode_factory(pd.DataFrame({"a": [1, 2]}))
        generator.apply_function(
            col_name="a", new_col_name="doubled", function=lambda value: 2 * value
        )
        assert generator.dataset["doubled"].tolist() == [2, 4]

    @pytest.mark.unit
    def test_apply_function_multiple_columns(self, generated_episode_factory):
        generator = generated_episode_factory(
            pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        )
        generator.apply_function(
            col_name=["a", "b"],
            new_col_name=["total", "difference"],
            function=lambda a, b: (a + b, a - b),
        )
        assert generator.dataset["total"].tolist() == [4.0, 6.0]
        assert generator.dataset["difference"].tolist() == [-2.0, -2.0]

    @pytest.mark.unit
    def test_apply_function_rejects_mismatched_name_types(
        self, generated_episode_factory
    ):
        generator = generated_episode_factory(pd.DataFrame({"a": [1]}))
        with pytest.raises(
            TypeError,
            match=(
                "`col_name` and `new_col_name` must either both be strings"
                " or both be lists/tuples of the same length."
            ),
        ):
            generator.apply_function(
                col_name=1, new_col_name=2, function=lambda value: value
            )

    @pytest.mark.unit
    def test_drop_add_and_rename_columns(self, generated_episode_factory):
        generator = generated_episode_factory(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        generator.drop_columns(col_names=["b"])
        generator.add_columns(col_names="phase", col_values=[0, 1])
        generator.rename_columns(col_names=["a"], new_col_names=["position"])
        pd.testing.assert_frame_equal(
            generator.dataset,
            pd.DataFrame({"position": [1, 2], "phase": [0, 1]}),
        )

    @pytest.mark.unit
    def test_get_number_of_rows(self, generated_episode_factory):
        generator = generated_episode_factory(pd.DataFrame({"a": [1, 2, 3]}))
        assert generator.get_number_of_rows() == 3

    @pytest.mark.unit
    def test_save_dataset_writes_csv_without_index(self):
        generator = EpisodeGenerator()
        generator.dataset = MagicMock()
        generator.save_dataset(path="episode.csv")
        generator.dataset.to_csv.assert_called_once_with("episode.csv", index=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "operation",
    [
        lambda generator: generator.apply_function(
            col_name="a", new_col_name="b", function=abs
        ),
        lambda generator: generator.drop_columns(col_names=["a"]),
        lambda generator: generator.add_columns(col_names="a", col_values=[1]),
        lambda generator: generator.get_number_of_rows(),
        lambda generator: generator.rename_columns(
            col_names=["a"], new_col_names=["b"]
        ),
        lambda generator: generator.show_dataset(),
        lambda generator: generator.save_dataset(path="episode.csv"),
    ],
)
def test_operations_raise_before_generation(operation):
    with pytest.raises(ValueError, match="Dataset is not generated yet"):
        operation(EpisodeGenerator())
