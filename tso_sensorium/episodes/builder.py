"""Episode generation from recorded data sources."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from tso_sensorium.processing.alignment import (
    DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    StateData,
    VideoData,
)

# Runtime-evaluated alias: typing generics keep it valid on Python 3.8.
ColumnNames = Union[str, List[str], Tuple[str, ...]]


class EpisodeGenerator:
    """Builds an episode table from multiple recorded data sources.

    Data sources are aligned on the first source's timestamps, concatenated
    column-wise, and can be transformed before saving. All methods return
    self so calls can be chained.
    """

    def __init__(self):
        self.data: list[StateData] = []
        self.dataset: Optional[pd.DataFrame] = None

    def add_video(
        self,
        video_path: Path | str,
        timestamps_path: Path | str,
        sync_col_name: str,
        frames_output_path: Path | str,
        frame_col_name: str,
        preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        save_frames: bool = False,
        max_sync_difference_seconds: float = DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    ) -> "EpisodeGenerator":
        """Register a video source for the episode.

        Args:
            video_path: Video file with one frame per recorded timestamp.
            timestamps_path: CSV file with the per-frame timestamps.
            sync_col_name: Timestamp column used for synchronization.
            frames_output_path: Directory where extracted frames are saved.
            frame_col_name: Column name holding the frame path.
            preprocess_fn: Optional transform applied to each frame.
            save_frames: Whether to extract and save frames to disk.
            max_sync_difference_seconds: Largest tolerated timestamp gap.

        Returns:
            Self, for method chaining.
        """
        self.data.append(
            VideoData(
                video_path=video_path,
                timestamps_path=timestamps_path,
                sync_col_name=sync_col_name,
                frames_output_path=frames_output_path,
                frame_col_name=frame_col_name,
                preprocess_fn=preprocess_fn,
                save_frames=save_frames,
                max_sync_difference_seconds=max_sync_difference_seconds,
            )
        )
        return self

    def add_state(
        self,
        state_data_path: Path | str,
        sync_col_name: str,
        dataset_cols: list[str],
        max_sync_difference_seconds: float = DEFAULT_MAX_SYNC_DIFFERENCE_SECONDS,
    ) -> "EpisodeGenerator":
        """Register a state CSV source for the episode.

        Args:
            state_data_path: CSV file with one row per recorded message.
            sync_col_name: Timestamp column used for synchronization.
            dataset_cols: Columns to keep in the aligned output.
            max_sync_difference_seconds: Largest tolerated timestamp gap.

        Returns:
            Self, for method chaining.
        """
        self.data.append(
            StateData(
                state_data_path=state_data_path,
                sync_col_name=sync_col_name,
                dataset_cols=dataset_cols,
                max_sync_difference_seconds=max_sync_difference_seconds,
            )
        )
        return self

    def generate_dataset(self) -> "EpisodeGenerator":
        """Align all data sources into a single episode table.

        The first registered source provides the reference timestamps. Reference
        rows outside the interval covered by every source are discarded before
        nearest-neighbour alignment, since independently started subscriptions
        do not receive their first and last messages at exactly the same time.

        Returns:
            Self, for method chaining.
        """
        if not self.data:
            raise ValueError("Cannot generate a dataset without data sources")

        source_sync_columns = [source.get_sync_col_data() for source in self.data]
        empty_sources = [
            source.state_data_path
            for source, sync_column in zip(self.data, source_sync_columns)
            if sync_column.empty
        ]
        if empty_sources:
            raise ValueError(
                f"Timestamp columns are empty for data sources: {empty_sources}."
                " Discarding episode."
            )

        overlap_start = max(sync_column.min() for sync_column in source_sync_columns)
        overlap_end = min(sync_column.max() for sync_column in source_sync_columns)
        if overlap_start > overlap_end:
            raise ValueError(
                "Recorded data sources do not have an overlapping timestamp range."
                " Discarding episode."
            )

        reference_sync_column = source_sync_columns[0]
        sync_col_data = reference_sync_column[
            reference_sync_column.between(overlap_start, overlap_end)
        ].reset_index(drop=True)
        if sync_col_data.empty:
            raise ValueError(
                "The reference data source has no samples in the shared timestamp"
                " range. Discarding episode."
            )

        dataframe = self.data[0].get_data(source_sync_dataframe=sync_col_data)
        for data in self.data[1:]:
            dataframe = pd.concat(
                [dataframe, data.get_data(source_sync_dataframe=sync_col_data)],
                axis=1,
            )
        sync_column_name = self.data[0].sync_col_name
        if sync_column_name not in dataframe.columns:
            dataframe.insert(0, sync_column_name, sync_col_data.to_numpy())
        self.dataset = dataframe
        return self

    def _check_dataset_is_generated(self) -> None:
        """Ensure the dataset has been generated before operating on it."""
        if self.dataset is None:
            raise ValueError("Dataset is not generated yet")

    def apply_function(
        self,
        col_name: ColumnNames,
        new_col_name: ColumnNames,
        function: Callable,
    ) -> "EpisodeGenerator":
        """Apply a function to dataset columns and create new columns.

        Supports both single column and multiple column operations: with
        string names the function maps one column to one new column, with
        lists the function receives one value per input column and returns
        one value per output column.

        Args:
            col_name: Input column name or names.
            new_col_name: Output column name or names.
            function: Transformation applied row-wise.

        Returns:
            Self, for method chaining.
        """
        self._check_dataset_is_generated()
        if isinstance(col_name, str) and isinstance(new_col_name, str):
            self.dataset[new_col_name] = self.dataset[col_name].apply(function)
        elif isinstance(col_name, (list, tuple)) or isinstance(
            new_col_name, (list, tuple)
        ):
            if isinstance(col_name, str):
                col_name = [col_name]
            if isinstance(new_col_name, str):
                new_col_name = [new_col_name]
            self.dataset[new_col_name] = self.dataset.apply(
                lambda row: function(*(row[c] for c in col_name)),
                axis=1,
                result_type="expand",
            )
        else:
            raise TypeError(
                "`col_name` and `new_col_name` must either both be strings"
                " or both be lists/tuples of the same length."
            )
        return self

    def drop_columns(self, col_names: list[str]) -> "EpisodeGenerator":
        """Remove the given columns from the dataset."""
        self._check_dataset_is_generated()
        self.dataset = self.dataset.drop(columns=col_names)
        return self

    def add_columns(
        self,
        col_names: ColumnNames,
        col_values: Union[pd.Series, np.ndarray, list, float, int, str],
    ) -> "EpisodeGenerator":
        """Add columns with the given values to the dataset."""
        self._check_dataset_is_generated()
        self.dataset[col_names] = col_values
        return self

    def get_number_of_rows(self) -> int:
        """Get the number of rows in the dataset."""
        self._check_dataset_is_generated()
        return len(self.dataset)

    def rename_columns(
        self, col_names: list[str], new_col_names: list[str]
    ) -> "EpisodeGenerator":
        """Rename dataset columns pairwise."""
        self._check_dataset_is_generated()
        self.dataset.rename(columns=dict(zip(col_names, new_col_names)), inplace=True)
        return self

    def show_dataset(self, n: int = 10) -> "EpisodeGenerator":
        """Print the first n rows of the dataset."""
        self._check_dataset_is_generated()
        print(self.dataset.head(n=n))
        return self

    def save_dataset(self, path: Path | str) -> "EpisodeGenerator":
        """Write the dataset to a CSV file."""
        self._check_dataset_is_generated()
        self.dataset.to_csv(path, index=False)
        return self
