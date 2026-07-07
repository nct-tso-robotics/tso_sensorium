"""Framework-free building blocks for sensor recording.

ROS adapters convert incoming messages to plain timestamps, values, and
frames, then delegate the actual file writing to these classes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

TIME_COLUMN = "time"
CSV_EXTENSION = "csv"
LOSSLESS_FOURCC = "HFYU"
LOSSY_FOURCC = "mp4v"
LOSSLESS_EXTENSION = "avi"
LOSSY_EXTENSION = "mp4"
BGRA_ENCODING = "bgra8"
IMAGE_METADATA_HEADER = ["encoding", "height", "width", "is_bigendian", "step"]


def image_buffer_to_bgr_frame(
    data: bytes, height: int, width: int, encoding: str
) -> np.ndarray:
    """Decode a raw image message buffer into a BGR frame.

    Args:
        data: Raw pixel buffer of the image message.
        height: Image height in pixels.
        width: Image width in pixels.
        encoding: Pixel encoding declared by the message.

    Returns:
        BGR frame, (H, W, 3).
    """
    frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))
    if encoding == BGRA_ENCODING:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


class TimestampedCsvRecorder:
    """Writes timestamped rows to a CSV file.

    The file is opened at construction and the header row is written
    immediately, with a time column prepended to the given columns.

    Args:
        output_folder: Directory to save the CSV file.
        file_name: Name of the CSV file, without extension.
        csv_header: Column names written after the time column.
    """

    def __init__(
        self,
        output_folder: Path | str,
        file_name: str,
        csv_header: Sequence[str],
    ):
        self.file_path = Path(output_folder, f"{file_name}.{CSV_EXTENSION}")
        self._csv_file = open(self.file_path, mode="w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([TIME_COLUMN] + list(csv_header))

    def write_row(self, timestamp_nanoseconds: int, values: Sequence[Any]) -> None:
        """Append one row with its timestamp.

        Rows arriving after ``close`` are dropped: subscription callbacks
        can still fire while a recording is being stopped.

        Args:
            timestamp_nanoseconds: Acquisition time of the values.
            values: Row values in header order.
        """
        if self._csv_file.closed:
            return
        self._csv_writer.writerow([timestamp_nanoseconds] + list(values))

    def close(self) -> None:
        """Flush and close the CSV file."""
        self._csv_file.close()


class VideoFileWriter:
    """Lazily initialized video file writer.

    The underlying encoder is created on the first frame, when the frame
    size is known. Lossless compression writes HuffYUV into an ``.avi``
    container, otherwise mp4v into ``.mp4``.

    Args:
        output_folder: Directory to save the video file.
        file_name: Name of the video file, without extension.
        frames_per_second: Playback frame rate of the written video.
        lossless_compression: Whether to encode losslessly. Lossless files
            are considerably larger.
    """

    def __init__(
        self,
        output_folder: Path | str,
        file_name: str,
        frames_per_second: float,
        lossless_compression: bool = False,
    ):
        extension = LOSSLESS_EXTENSION if lossless_compression else LOSSY_EXTENSION
        self.file_path = Path(output_folder, f"{file_name}.{extension}")
        self.frames_per_second = frames_per_second
        self.fourcc = LOSSLESS_FOURCC if lossless_compression else LOSSY_FOURCC
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._closed = False

    def write_frame(self, frame: np.ndarray) -> None:
        """Append a BGR frame, creating the encoder on first use.

        Frames arriving after ``close`` are dropped: subscription
        callbacks can still fire while a recording is being stopped.

        Args:
            frame: BGR image, (H, W, 3). All frames must share one size.
        """
        if self._closed:
            return
        if self._video_writer is None:
            height, width = frame.shape[:2]
            self._video_writer = cv2.VideoWriter(
                str(self.file_path),
                cv2.VideoWriter_fourcc(*self.fourcc),
                self.frames_per_second,
                (width, height),
            )
        self._video_writer.write(frame)

    def close(self) -> None:
        """Finalize the video file if any frame was written."""
        self._closed = True
        if self._video_writer is not None:
            self._video_writer.release()
