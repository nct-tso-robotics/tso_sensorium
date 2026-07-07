"""Tests for tso_sensorium.recording.core module."""

import csv
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import cv2
import numpy as np
import pytest

from tso_sensorium.recording.core import (
    TimestampedCsvRecorder,
    VideoFileWriter,
    image_buffer_to_bgr_frame,
)

CSV_WRITER_PATH = "tso_sensorium.recording.core.csv.writer"
VIDEO_WRITER_PATH = "tso_sensorium.recording.core.cv2.VideoWriter"
FOURCC_PATH = "tso_sensorium.recording.core.cv2.VideoWriter_fourcc"


@pytest.fixture
def frame_factory(rng):
    def factory(height: int = 48, width: int = 64) -> np.ndarray:
        return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)

    return factory


class TestTimestampedCsvRecorder:
    @pytest.mark.unit
    def test_opens_file_and_writes_header_with_time_column(self):
        csv_writer = MagicMock()
        with (
            patch("builtins.open", mock_open()) as open_mock,
            patch(CSV_WRITER_PATH, return_value=csv_writer),
        ):
            TimestampedCsvRecorder(
                output_folder="out", file_name="state", csv_header=["x", "y"]
            )
        open_mock.assert_called_once_with(
            Path("out", "state.csv"), mode="w", newline=""
        )
        csv_writer.writerow.assert_called_once_with(["time", "x", "y"])

    @pytest.mark.unit
    def test_write_row_prepends_timestamp(self):
        csv_writer = MagicMock()
        opened = mock_open()
        opened.return_value.closed = False
        with (
            patch("builtins.open", opened),
            patch(CSV_WRITER_PATH, return_value=csv_writer),
        ):
            recorder = TimestampedCsvRecorder(
                output_folder="out", file_name="state", csv_header=["x"]
            )
            recorder.write_row(timestamp_nanoseconds=123, values=[1.5])
        csv_writer.writerow.assert_called_with([123, 1.5])

    @pytest.mark.unit
    def test_rows_after_close_are_dropped(self, tmp_path):
        recorder = TimestampedCsvRecorder(
            output_folder=tmp_path, file_name="state", csv_header=["x"]
        )
        recorder.close()
        recorder.write_row(timestamp_nanoseconds=123, values=[1.5])
        assert (tmp_path / "state.csv").read_text().strip() == "time,x"

    @pytest.mark.unit
    def test_close_closes_file(self):
        with (
            patch("builtins.open", mock_open()) as open_mock,
            patch(CSV_WRITER_PATH),
        ):
            recorder = TimestampedCsvRecorder(
                output_folder="out", file_name="state", csv_header=["x"]
            )
            recorder.close()
        open_mock.return_value.close.assert_called_once_with()

    @pytest.mark.integration
    def test_round_trip_writes_header_and_rows(self, tmp_path):
        recorder = TimestampedCsvRecorder(
            output_folder=tmp_path, file_name="state", csv_header=["x", "y"]
        )
        recorder.write_row(timestamp_nanoseconds=123, values=[1.5, "a"])
        recorder.close()
        with open(tmp_path / "state.csv", newline="") as csv_file:
            rows = list(csv.reader(csv_file))
        assert rows == [["time", "x", "y"], ["123", "1.5", "a"]]


class TestVideoFileWriter:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "lossless_compression, expected_fourcc, expected_extension",
        [
            (True, "HFYU", "avi"),
            (False, "mp4v", "mp4"),
        ],
    )
    def test_creates_encoder_from_first_frame_size(
        self,
        frame_factory,
        lossless_compression,
        expected_fourcc,
        expected_extension,
    ):
        writer = VideoFileWriter(
            output_folder="out",
            file_name="video",
            frames_per_second=30.0,
            lossless_compression=lossless_compression,
        )
        frame = frame_factory(height=48, width=64)
        with (
            patch(VIDEO_WRITER_PATH) as video_writer_class,
            patch(FOURCC_PATH, return_value="fourcc_code") as fourcc,
        ):
            writer.write_frame(frame=frame)
        fourcc.assert_called_once_with(*expected_fourcc)
        video_writer_class.assert_called_once_with(
            str(Path("out", f"video.{expected_extension}")),
            "fourcc_code",
            30.0,
            (64, 48),
        )
        write_mock = video_writer_class.return_value.write
        assert write_mock.call_count == 1
        assert write_mock.call_args.args[0] is frame

    @pytest.mark.unit
    def test_reuses_encoder_for_subsequent_frames(self, frame_factory):
        writer = VideoFileWriter(
            output_folder="out", file_name="video", frames_per_second=30.0
        )
        with (
            patch(VIDEO_WRITER_PATH) as video_writer_class,
            patch(FOURCC_PATH),
        ):
            writer.write_frame(frame=frame_factory())
            writer.write_frame(frame=frame_factory())
        assert video_writer_class.call_count == 1
        assert video_writer_class.return_value.write.call_count == 2

    @pytest.mark.unit
    def test_close_releases_encoder(self, frame_factory):
        writer = VideoFileWriter(
            output_folder="out", file_name="video", frames_per_second=30.0
        )
        with patch(VIDEO_WRITER_PATH) as video_writer_class, patch(FOURCC_PATH):
            writer.write_frame(frame=frame_factory())
            writer.close()
        video_writer_class.return_value.release.assert_called_once_with()

    @pytest.mark.unit
    def test_close_without_frames_does_not_create_encoder(self):
        writer = VideoFileWriter(
            output_folder="out", file_name="video", frames_per_second=30.0
        )
        with patch(VIDEO_WRITER_PATH) as video_writer_class:
            writer.close()
        video_writer_class.assert_not_called()

    @pytest.mark.integration
    def test_round_trip_writes_readable_video(self, tmp_path, frame_factory):
        writer = VideoFileWriter(
            output_folder=tmp_path, file_name="video", frames_per_second=30.0
        )
        for _ in range(3):
            writer.write_frame(frame=frame_factory(height=48, width=64))
        writer.close()
        capture = cv2.VideoCapture(str(tmp_path / "video.mp4"))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        assert frame_count == 3
        assert (width, height) == (64, 48)


class TestImageBufferToBgrFrame:
    @pytest.mark.unit
    def test_reshapes_three_channel_buffer(self, rng):
        frame = rng.integers(0, 255, size=(4, 6, 3), dtype=np.uint8)
        decoded = image_buffer_to_bgr_frame(
            data=frame.tobytes(), height=4, width=6, encoding="bgr8"
        )
        np.testing.assert_array_equal(decoded, frame)

    @pytest.mark.unit
    def test_converts_bgra_buffer_to_bgr(self, rng):
        frame = rng.integers(0, 255, size=(4, 6, 4), dtype=np.uint8)
        decoded = image_buffer_to_bgr_frame(
            data=frame.tobytes(), height=4, width=6, encoding="bgra8"
        )
        assert decoded.shape == (4, 6, 3)
        np.testing.assert_array_equal(decoded, frame[:, :, :3])


class TestVideoFileWriterClose:
    @pytest.mark.unit
    def test_frames_after_close_are_dropped(self, tmp_path):
        writer = VideoFileWriter(
            output_folder=tmp_path, file_name="camera", frames_per_second=15
        )
        writer.close()
        writer.write_frame(frame=np.zeros((8, 6, 3), dtype=np.uint8))
        assert not (tmp_path / "camera.mp4").exists()


class TestImageEncodings:
    @pytest.mark.unit
    def test_bgr8_passthrough(self):
        frame = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
        result = image_buffer_to_bgr_frame(
            data=frame.tobytes(), height=2, width=2, encoding="bgr8"
        )
        np.testing.assert_array_equal(result, frame)

    @pytest.mark.unit
    def test_rgb8_swaps_red_and_blue(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        rgb[0, 0] = [10, 20, 30]  # R, G, B
        result = image_buffer_to_bgr_frame(
            data=rgb.tobytes(), height=1, width=1, encoding="rgb8"
        )
        assert result[0, 0].tolist() == [30, 20, 10]  # B, G, R

    @pytest.mark.unit
    def test_mono8_expands_to_three_channels(self):
        mono = np.array([[7, 9]], dtype=np.uint8)
        result = image_buffer_to_bgr_frame(
            data=mono.tobytes(), height=1, width=2, encoding="mono8"
        )
        assert result.shape == (1, 2, 3)
        assert result[0, 0].tolist() == [7, 7, 7]

    @pytest.mark.unit
    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError, match="Unsupported image encoding 'yuv422'"):
            image_buffer_to_bgr_frame(
                data=b"\x00" * 6, height=1, width=1, encoding="yuv422"
            )
