"""Timestamp alignment of recorded data sources (state CSVs and videos)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd

DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS = 0.1
NANOSECONDS_PER_SECOND = 1e9
FRAME_FILE_EXTENSION = ".png"


class StateData:
    """Aligns a recorded state CSV to a reference timestamp column.

    Args:
        state_data_path: CSV file with one row per recorded message.
        sync_col_name: Timestamp column used for synchronization.
        dataset_cols: Columns to keep in the aligned output.
        max_sync_difference_seconds: Largest tolerated gap between a
            reference timestamp and this source's closest message.
    """

    def __init__(
        self,
        state_data_path: Path | str,
        sync_col_name: str,
        dataset_cols: Optional[list[str]],
        max_sync_difference_seconds: float = DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    ):
        self.state_data_path = state_data_path
        self.sync_col_name = sync_col_name
        self.dataset_cols = dataset_cols
        self.max_sync_difference_seconds = max_sync_difference_seconds

    def _get_aligned_indexes(self, source_sync_dataframe: pd.Series) -> np.ndarray:
        """Find the closest matching timestamps between source and state data.

        Args:
            source_sync_dataframe: Reference timestamps to align to.

        Returns:
            Indices of state data rows that align with source timestamps.
        """
        self_sync_col = self.get_sync_col_data().to_numpy()
        source_sync_col = source_sync_dataframe.to_numpy()
        aligned_sync_col = np.zeros_like(source_sync_col)
        if np.isnan(self_sync_col).any() or np.isnan(source_sync_col).any():
            raise ValueError(
                f"Timestamp column contains missing values in"
                f" {self.state_data_path}. Discarding episode."
            )

        max_diff = 0
        # For each timestamp in source data, find the closest match in state data
        for i, source_time in enumerate(source_sync_col):
            time_diffs = np.abs(self_sync_col - source_time)
            closest_idx = np.argmin(time_diffs)
            diff = time_diffs[closest_idx]
            if diff > max_diff:
                max_diff = diff
            aligned_sync_col[i] = closest_idx
        if max_diff > self.max_sync_difference_seconds * NANOSECONDS_PER_SECOND:
            raise ValueError(
                f"The maximum timestamp difference"
                f" ({max_diff / NANOSECONDS_PER_SECOND} seconds) is greater"
                f" than {self.max_sync_difference_seconds} seconds for data"
                f" {self.state_data_path}. Discarding episode."
            )
        return aligned_sync_col

    def get_data(self, source_sync_dataframe: pd.Series) -> pd.DataFrame:
        """Get aligned state data based on timestamp synchronization.

        Args:
            source_sync_dataframe: Reference timestamps to align to.

        Returns:
            Aligned rows restricted to the configured dataset columns.
        """
        dataframe = pd.read_csv(self.state_data_path)
        aligned_sync_col = self._get_aligned_indexes(
            source_sync_dataframe=source_sync_dataframe
        )
        filtered_dataframe = dataframe.iloc[aligned_sync_col].reset_index(drop=True)
        return filtered_dataframe[self.dataset_cols]

    def get_sync_col_data(self) -> pd.Series:
        """Read the synchronization timestamp column from the CSV."""
        return pd.read_csv(self.state_data_path)[self.sync_col_name]


class VideoData(StateData):
    """Aligns recorded video frames to a reference timestamp column.

    Frames are extracted from the video, optionally preprocessed, saved to
    disk, and referenced by path in the aligned output.

    Args:
        video_path: Video file with one frame per recorded timestamp.
        timestamps_path: CSV file with the per-frame timestamps.
        sync_col_name: Timestamp column used for synchronization.
        frames_output_path: Directory where extracted frames are saved.
        frame_col_name: Column name holding the frame path in the output.
        preprocess_fn: Optional transform applied to each frame before
            saving.
        save_frames: Whether to extract and save frames to disk.
    """

    def __init__(
        self,
        video_path: Path | str,
        timestamps_path: Path | str,
        sync_col_name: str,
        frames_output_path: Path | str,
        frame_col_name: str,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save_frames: bool = False,
        max_sync_difference_seconds: float = DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    ):
        super().__init__(
            state_data_path=timestamps_path,
            sync_col_name=sync_col_name,
            dataset_cols=None,
            max_sync_difference_seconds=max_sync_difference_seconds,
        )
        self.video_path = video_path
        self.frames_output_path = frames_output_path
        self.frame_col_name = frame_col_name
        self.save_frames = save_frames
        self.preprocess_fn = preprocess_fn
        self._frame_count: Optional[int] = None
        self._sync_column: Optional[pd.Series] = None

    def _get_frame_count(self) -> int:
        """Return the number of decodable frames reported by the video."""
        if self._frame_count is not None:
            return self._frame_count
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file {self.video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frame_count <= 0:
            raise ValueError(f"Video file has no frames: {self.video_path}")
        self._frame_count = frame_count
        return frame_count

    def get_sync_col_data(self) -> pd.Series:
        """Read timestamps bounded to frames that exist in the video."""
        if self._sync_column is not None:
            return self._sync_column
        timestamps = super().get_sync_col_data()
        frame_count = self._get_frame_count()
        if len(timestamps) != frame_count:
            logging.warning(
                "Video/timestamp length mismatch for %s: %d frames, %d timestamps; "
                "using the shared prefix",
                self.video_path,
                frame_count,
                len(timestamps),
            )
        self._sync_column = timestamps.iloc[:frame_count].reset_index(drop=True)
        return self._sync_column

    def _get_frame_name(self, frame_number: int) -> str:
        return f"{int(frame_number)}{FRAME_FILE_EXTENSION}"

    def _save_frames(self) -> None:
        """Extract all video frames to the output directory."""
        Path(self.frames_output_path).mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video file {self.video_path}")
        frame_count = 0
        while True:
            frame_available, frame = capture.read()
            if not frame_available:
                break
            if self.preprocess_fn is not None:
                frame = self.preprocess_fn(frame)
            frame_path = Path(
                self.frames_output_path,
                self._get_frame_name(frame_number=frame_count),
            )
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
        capture.release()

    def get_data(self, source_sync_dataframe: pd.Series) -> pd.DataFrame:
        """Get aligned frame paths based on timestamp synchronization.

        Args:
            source_sync_dataframe: Reference timestamps to align to.

        Returns:
            Single-column dataframe with the aligned frame paths.
        """
        if self.save_frames:
            self._save_frames()
        frame_numbers = self._get_aligned_indexes(
            source_sync_dataframe=source_sync_dataframe
        ).tolist()
        frame_paths = [
            Path(
                self.frames_output_path,
                self._get_frame_name(frame_number=frame_number),
            )
            for frame_number in frame_numbers
        ]
        return pd.DataFrame(frame_paths, columns=[self.frame_col_name])
