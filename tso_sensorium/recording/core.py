"""Framework-free building blocks for sensor recording.

ROS adapters convert incoming messages to plain timestamps, values, and
frames, then delegate the actual file writing to these classes.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import threading
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
BGR_ENCODING = "bgr8"
RGB_ENCODING = "rgb8"
BGRA_ENCODING = "bgra8"
RGBA_ENCODING = "rgba8"
MONO_ENCODING = "mono8"
_ENCODING_CONVERSIONS = {
    RGB_ENCODING: cv2.COLOR_RGB2BGR,
    BGRA_ENCODING: cv2.COLOR_BGRA2BGR,
    RGBA_ENCODING: cv2.COLOR_RGBA2BGR,
    MONO_ENCODING: cv2.COLOR_GRAY2BGR,
}
IMAGE_METADATA_HEADER = ["encoding", "height", "width", "is_bigendian", "step"]
FFMPEG_EXECUTABLE = "ffmpeg"
FFMPEG_LOG_LEVEL = "error"
RETIMED_FILE_INFIX = ".retimed"


def image_buffer_to_bgr_frame(
    data: bytes, height: int, width: int, encoding: str
) -> np.ndarray:
    """Decode a raw image message buffer into a BGR frame.

    Handles the common ``sensor_msgs/Image`` byte encodings; ``bgr8`` is
    already in the target layout, the others are converted to it.

    Args:
        data: Raw pixel buffer of the image message.
        height: Image height in pixels.
        width: Image width in pixels.
        encoding: Pixel encoding declared by the message.

    Returns:
        BGR frame, (H, W, 3).
    """
    frame = np.frombuffer(data, dtype=np.uint8).reshape((height, width, -1))
    if encoding == BGR_ENCODING:
        return frame
    conversion = _ENCODING_CONVERSIONS.get(encoding)
    if conversion is None:
        raise ValueError(
            f"Unsupported image encoding '{encoding}'. Supported encodings:"
            f" {[BGR_ENCODING, *_ENCODING_CONVERSIONS]}."
        )
    return cv2.cvtColor(frame, conversion)


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
        self._lock = threading.Lock()

    def write_row(self, timestamp_nanoseconds: int, values: Sequence[Any]) -> None:
        """Append one row with its timestamp.

        Rows arriving after ``close`` are dropped: subscription callbacks
        can still fire while a recording is being stopped, so the check and
        the write are guarded together against a concurrent ``close``.

        Args:
            timestamp_nanoseconds: Acquisition time of the values.
            values: Row values in header order.
        """
        with self._lock:
            if self._csv_file.closed:
                return
            self._csv_writer.writerow([timestamp_nanoseconds] + list(values))

    def close(self) -> None:
        """Flush and close the CSV file."""
        with self._lock:
            self._csv_file.close()


class VideoFileWriter:
    """Lazily initialized, timestamp-aware video file writer.

    The underlying encoder is created on the first frame, when the frame
    size is known. Lossless compression writes HuffYUV into an ``.avi``
    container, otherwise mp4v into ``.mp4``. On close, the video is retimed
    without re-encoding so its duration follows the acquisition timestamps
    while retaining exactly one encoded frame per input frame.

    Args:
        output_folder: Directory to save the video file.
        file_name: Name of the video file, without extension.
        frames_per_second: Initial encoder rate. Acquisition timestamps
            determine the finalized playback rate.
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
        self._first_timestamp_nanoseconds: Optional[int] = None
        self._last_timestamp_nanoseconds: Optional[int] = None
        self._frame_count = 0
        self._closed = False
        self._lock = threading.Lock()

    def write_frame(self, frame: np.ndarray, timestamp_nanoseconds: int) -> None:
        """Append a BGR frame, creating the encoder on first use.

        Frames arriving after ``close`` are dropped: subscription
        callbacks can still fire while a recording is being stopped, so the
        check and the write are guarded together against a concurrent
        ``close``.

        Args:
            frame: BGR image, (H, W, 3). All frames must share one size.
            timestamp_nanoseconds: Acquisition time of the frame.
        """
        with self._lock:
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
            if self._first_timestamp_nanoseconds is None:
                self._first_timestamp_nanoseconds = timestamp_nanoseconds
            self._last_timestamp_nanoseconds = timestamp_nanoseconds
            self._frame_count += 1

    def _effective_frames_per_second(self) -> float:
        if (
            self._first_timestamp_nanoseconds is None
            or self._last_timestamp_nanoseconds is None
            or self._last_timestamp_nanoseconds <= self._first_timestamp_nanoseconds
        ):
            raise RuntimeError(
                "Cannot retime video because frame timestamps do not increase."
            )
        elapsed_seconds = (
            self._last_timestamp_nanoseconds - self._first_timestamp_nanoseconds
        ) / 1_000_000_000
        return (self._frame_count - 1) / elapsed_seconds

    def _retime_command(
        self,
        ffmpeg_path: str,
        temporary_file_path: Path,
        effective_frames_per_second: float,
    ) -> list[str]:
        command = [
            ffmpeg_path,
            "-y",
            "-loglevel",
            FFMPEG_LOG_LEVEL,
        ]
        if self.file_path.suffix == f".{LOSSLESS_EXTENSION}":
            command.extend(["-i", str(self.file_path)])
        else:
            timestamp_scale = self.frames_per_second / effective_frames_per_second
            command.extend(
                ["-itsscale", str(timestamp_scale), "-i", str(self.file_path)]
            )
        command.extend(["-map", "0:v:0", "-c:v", "copy"])
        if self.file_path.suffix == f".{LOSSLESS_EXTENSION}":
            command.extend(["-r", str(effective_frames_per_second)])
        command.extend(["-an", str(temporary_file_path)])
        return command

    def _retime_video(self) -> None:
        effective_frames_per_second = self._effective_frames_per_second()
        ffmpeg_path = shutil.which(FFMPEG_EXECUTABLE)
        if ffmpeg_path is None:
            raise RuntimeError(
                f"Cannot retime video because {FFMPEG_EXECUTABLE} is not available."
            )
        temporary_file_path = self.file_path.with_name(
            f".{self.file_path.stem}{RETIMED_FILE_INFIX}{self.file_path.suffix}"
        )
        command = self._retime_command(
            ffmpeg_path=ffmpeg_path,
            temporary_file_path=temporary_file_path,
            effective_frames_per_second=effective_frames_per_second,
        )
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            temporary_file_path.unlink(missing_ok=True)
            error_message = result.stderr.strip()
            raise RuntimeError(f"Failed to retime video: {error_message}")
        temporary_file_path.replace(self.file_path)

    def close(self) -> None:
        """Finalize the video file if any frame was written."""
        with self._lock:
            self._closed = True
            if self._video_writer is not None:
                self._video_writer.release()
                if self._frame_count > 1:
                    self._retime_video()
