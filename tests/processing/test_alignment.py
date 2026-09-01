"""Tests for tso_sensorium.processing.alignment module."""

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest

from tso_sensorium.processing.alignment import (
    DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    StateData,
    VideoData,
)

READ_CSV_PATH = "tso_sensorium.processing.alignment.pd.read_csv"
VIDEO_CAPTURE_PATH = "tso_sensorium.processing.alignment.cv2.VideoCapture"
IMWRITE_PATH = "tso_sensorium.processing.alignment.cv2.imwrite"


@pytest.fixture
def state_dataframe_factory():
    def factory(timestamps=(0, 10, 20), values=("a", "b", "c")) -> pd.DataFrame:
        return pd.DataFrame({"timestamp": list(timestamps), "value": list(values)})

    return factory


@pytest.fixture
def video_data_factory():
    def factory(
        frames_output_path="frames",
        preprocess_fn=None,
        save_frames=False,
    ) -> VideoData:
        return VideoData(
            video_path="episode.mp4",
            timestamps_path="timestamps.csv",
            sync_col_name="timestamp",
            frames_output_path=frames_output_path,
            frame_col_name="frame_path",
            preprocess_fn=preprocess_fn,
            save_frames=save_frames,
        )

    return factory


class TestStateData:
    @pytest.mark.unit
    def test_get_sync_col_data_reads_configured_column(self, state_dataframe_factory):
        dataframe = state_dataframe_factory(timestamps=(5, 15), values=("a", "b"))
        state_data = StateData(
            state_data_path="state.csv",
            sync_col_name="timestamp",
            dataset_cols=["value"],
        )
        with patch(READ_CSV_PATH, return_value=dataframe) as read_csv:
            sync_column = state_data.get_sync_col_data()
        read_csv.assert_called_once_with("state.csv")
        pd.testing.assert_series_equal(sync_column, dataframe["timestamp"])

    @pytest.mark.unit
    def test_get_data_aligns_rows_to_closest_timestamps(self, state_dataframe_factory):
        dataframe = state_dataframe_factory(
            timestamps=(0, 10, 20), values=("a", "b", "c")
        )
        state_data = StateData(
            state_data_path="state.csv",
            sync_col_name="timestamp",
            dataset_cols=["value"],
        )
        source_timestamps = pd.Series([1, 19])
        with patch(READ_CSV_PATH, return_value=dataframe):
            aligned = state_data.get_data(source_sync_dataframe=source_timestamps)
        assert list(aligned.columns) == ["value"]
        assert aligned["value"].tolist() == ["a", "c"]

    @pytest.mark.unit
    def test_raises_when_sync_gap_exceeds_threshold(self, state_dataframe_factory):
        dataframe = state_dataframe_factory(timestamps=(0,), values=("a",))
        state_data = StateData(
            state_data_path="state.csv",
            sync_col_name="timestamp",
            dataset_cols=["value"],
        )
        gap = int(0.3 * 1e9)
        expected_message = (
            f"The maximum timestamp difference ({gap / 1e9} seconds) is"
            f" greater than {DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS} seconds"
            f" for data state.csv. Discarding episode."
        )
        with patch(READ_CSV_PATH, return_value=dataframe):
            with pytest.raises(ValueError, match=re.escape(expected_message)):
                state_data.get_data(source_sync_dataframe=pd.Series([gap]))

    @pytest.mark.unit
    def test_custom_tolerance_accepts_larger_gaps(self, state_dataframe_factory):
        dataframe = state_dataframe_factory(timestamps=(0,), values=("a",))
        state_data = StateData(
            state_data_path="state.csv",
            sync_col_name="timestamp",
            dataset_cols=["value"],
            max_sync_difference_seconds=0.5,
        )
        gap = int(0.3 * 1e9)
        with patch(READ_CSV_PATH, return_value=dataframe):
            aligned = state_data.get_data(source_sync_dataframe=pd.Series([gap]))
        assert aligned["value"].tolist() == ["a"]


class TestVideoData:
    @pytest.mark.unit
    def test_frame_count_is_read_once_and_cached(self, video_data_factory):
        video_data = video_data_factory()
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.get.return_value = 3
        with patch(VIDEO_CAPTURE_PATH, return_value=capture) as video_capture:
            first_count = video_data._get_frame_count()
            second_count = video_data._get_frame_count()

        assert first_count == 3
        assert second_count == 3
        video_capture.assert_called_once_with("episode.mp4")
        capture.get.assert_called_once_with(cv2.CAP_PROP_FRAME_COUNT)
        capture.release.assert_called_once_with()

    @pytest.mark.unit
    def test_get_data_returns_aligned_frame_paths(self, video_data_factory):
        video_data = video_data_factory(frames_output_path="frames")
        timestamps = pd.DataFrame({"timestamp": [0, 10, 20]})
        with (
            patch(READ_CSV_PATH, return_value=timestamps),
            patch.object(video_data, "_get_frame_count", return_value=3),
        ):
            aligned = video_data.get_data(source_sync_dataframe=pd.Series([1, 19]))
        assert list(aligned.columns) == ["frame_path"]
        assert aligned["frame_path"].tolist() == [
            Path("frames", "0.png"),
            Path("frames", "2.png"),
        ]

    @pytest.mark.unit
    def test_get_data_saves_frames_when_enabled(self, video_data_factory):
        video_data = video_data_factory(save_frames=True)
        timestamps = pd.DataFrame({"timestamp": [0]})
        with (
            patch.object(video_data, "_save_frames") as save_frames,
            patch.object(video_data, "_get_frame_count", return_value=1),
            patch(READ_CSV_PATH, return_value=timestamps),
        ):
            video_data.get_data(source_sync_dataframe=pd.Series([0]))
        save_frames.assert_called_once_with()

    @pytest.mark.unit
    def test_timestamp_rows_are_bounded_to_decodable_frames(self, video_data_factory):
        video_data = video_data_factory()
        timestamps = pd.DataFrame({"timestamp": [0, 10, 20]})
        with (
            patch(READ_CSV_PATH, return_value=timestamps),
            patch.object(video_data, "_get_frame_count", return_value=2),
        ):
            sync_column = video_data.get_sync_col_data()

        assert sync_column.tolist() == [0, 10]

    @pytest.mark.unit
    def test_save_frames_writes_preprocessed_frames(
        self, video_data_factory, tmp_path, rng
    ):
        frames = [
            rng.integers(0, 255, size=(4, 4, 3), dtype=np.uint8) for _ in range(2)
        ]
        preprocess_fn = MagicMock(side_effect=lambda frame: frame + 1)
        video_data = video_data_factory(
            frames_output_path=tmp_path / "frames",
            preprocess_fn=preprocess_fn,
            save_frames=True,
        )
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.side_effect = [
            (True, frames[0]),
            (True, frames[1]),
            (False, None),
        ]
        with (
            patch(VIDEO_CAPTURE_PATH, return_value=capture) as video_capture,
            patch(IMWRITE_PATH) as imwrite,
        ):
            video_data._save_frames()
        video_capture.assert_called_once_with("episode.mp4")
        assert preprocess_fn.call_count == 2
        assert imwrite.call_count == 2
        for frame_number, (written_call, frame) in enumerate(
            zip(imwrite.call_args_list, frames)
        ):
            written_path, written_frame = written_call.args
            assert written_path == str(tmp_path / "frames" / f"{frame_number}.png")
            np.testing.assert_array_equal(written_frame, frame + 1)
        capture.release.assert_called_once_with()

    @pytest.mark.unit
    def test_save_frames_raises_when_video_cannot_open(
        self, video_data_factory, tmp_path
    ):
        video_data = video_data_factory(frames_output_path=tmp_path / "frames")
        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch(VIDEO_CAPTURE_PATH, return_value=capture):
            with pytest.raises(
                ValueError,
                match=re.escape("Could not open video file episode.mp4"),
            ):
                video_data._save_frames()


@pytest.mark.unit
def test_nan_timestamp_raises_instead_of_silent_corruption(tmp_path):
    state_path = tmp_path / "state.csv"
    pd.DataFrame({"time": [0, np.nan, 2e8], "x": [1.0, 2.0, 3.0]}).to_csv(
        state_path, index=False
    )
    state = StateData(
        state_data_path=state_path, sync_col_name="time", dataset_cols=["x"]
    )
    with pytest.raises(ValueError, match="missing values"):
        state.get_data(source_sync_dataframe=pd.Series([0.0, 1e8, 2e8]))
