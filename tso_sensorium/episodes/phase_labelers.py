"""Automatic phase labeling over recorded episode signals.

Labelers read a recorded state CSV and produce automatic phase segments,
merged into each episode's ``annotations.json`` without touching manual
segments. Requires Python 3.9+.
"""

from __future__ import annotations

import abc
from typing import Annotated, Dict, List, Literal, Optional, Union

import pandas as pd
from pydantic import Field

from tso_sensorium.configuration import ConfigModel

from tso_sensorium.episodes.annotations import (
    ANNOTATIONS_FILE_NAME,
    EpisodeAnnotations,
    PhaseSegment,
)
from tso_sensorium.episodes.dataset_builder import discover_episode_directories


class PhaseLabeler(ConfigModel, abc.ABC):
    """Produces automatic phase segments from an episode's signals."""

    @abc.abstractmethod
    def label(self, table: pd.DataFrame, sync_column: str) -> List[PhaseSegment]:
        """Segment one episode.

        Args:
            table: Recorded state rows, one per message.
            sync_column: Timestamp column of the table.

        Returns:
            Automatic segments covering the labeled spans.
        """


class ColumnThresholdLabeler(PhaseLabeler):
    """Two-phase labeling by thresholding one numeric column.

    Consecutive rows on the same side of the threshold form one segment;
    a segment ends where the next one starts, and the last segment ends
    just after the final timestamp.

    Args:
        column: Numeric column driving the phases.
        threshold: Boundary between the two phases.
        above_phase: Phase label where the column exceeds the threshold.
        below_phase: Phase label elsewhere.
    """

    type: Literal["column_threshold"] = "column_threshold"
    column: str = ""
    threshold: float = 0.0
    above_phase: int = 1
    below_phase: int = 0

    def label(self, table: pd.DataFrame, sync_column: str) -> List[PhaseSegment]:
        if len(table) == 0:
            return []
        values = table[self.column].to_numpy(dtype=float)
        times = table[sync_column].to_numpy()
        phases = [
            self.above_phase if value > self.threshold else self.below_phase
            for value in values
        ]
        segments = []
        span_start_index = 0
        for index in range(1, len(phases) + 1):
            if index < len(phases) and phases[index] == phases[span_start_index]:
                continue
            end = int(times[index]) if index < len(phases) else int(times[-1]) + 1
            segments.append(
                PhaseSegment(
                    start=int(times[span_start_index]),
                    end=end,
                    phase=phases[span_start_index],
                )
            )
            span_start_index = index
        return segments


class PhaseTrigger(ConfigModel, abc.ABC):
    """Condition ending a sequential phase."""

    @abc.abstractmethod
    def first_index(self, table: pd.DataFrame, start_index: int) -> int:
        """Return the first row at or after ``start_index`` firing the trigger.

        Args:
            table: Recorded state rows.
            start_index: Row where the current phase begins.

        Returns:
            Absolute row index where the next phase starts; the table
            length when the trigger never fires.
        """


class ColumnThresholdTrigger(PhaseTrigger):
    """Fires when a numeric column crosses a threshold.

    Args:
        column: Numeric column watched by the trigger.
        threshold: Boundary value.
        above: Whether the trigger fires above the threshold instead of
            at or below it.
    """

    type: Literal["column_threshold"] = "column_threshold"
    column: str = ""
    threshold: float = 0.5
    above: bool = True

    def first_index(self, table: pd.DataFrame, start_index: int) -> int:
        values = table[self.column].to_numpy(dtype=float)[start_index:]
        mask = values > self.threshold if self.above else values <= self.threshold
        matches = mask.nonzero()[0]
        offset = int(matches[0]) if len(matches) else len(values)
        return start_index + offset


class MovementTrigger(PhaseTrigger):
    """Fires on the magnitude of motion between consecutive rows.

    The movement norm is rotation invariant, so robot-frame and
    camera-frame position columns give identical results.

    Args:
        columns: Position columns whose consecutive differences are
            measured.
        epsilon: Movement magnitude threshold.
        above: Whether the trigger fires when movement reaches ``epsilon``
            instead of when it settles below it.
    """

    type: Literal["movement"] = "movement"
    columns: List[str] = Field(default_factory=list)
    epsilon: float = 1e-4
    above: bool = True

    def first_index(self, table: pd.DataFrame, start_index: int) -> int:
        positions = table[self.columns].to_numpy(dtype=float)[start_index:]
        if len(positions) < 2:
            return start_index + max(len(positions) - 1, 0)
        movement = ((positions[1:] - positions[:-1]) ** 2).sum(axis=1) ** 0.5
        mask = movement >= self.epsilon if self.above else movement <= self.epsilon
        matches = mask.nonzero()[0]
        offset = int(matches[0]) if len(matches) else len(movement)
        return start_index + offset


AnyPhaseTrigger = Annotated[
    Union[ColumnThresholdTrigger, MovementTrigger], Field(discriminator="type")
]


class SequentialPhase(ConfigModel):
    """One phase of a sequential labeling plan.

    Args:
        trigger: Condition ending the phase; the final phase runs to the
            end of the episode with no trigger.
        phase: Integer phase label from the dataset's legend.
    """

    phase: int = 0
    trigger: Optional[AnyPhaseTrigger] = None


class SequentialTriggerLabeler(PhaseLabeler):
    """Phases advance in order, each ended by its trigger.

    Mirrors trigger-driven task structures: the episode starts in the
    first phase, each trigger hands over to the next phase at the row
    where it fires, and the final phase covers the remainder.

    Args:
        phases: Ordered phases with their end triggers.
    """

    type: Literal["sequential_trigger"] = "sequential_trigger"
    phases: List[SequentialPhase] = Field(default_factory=list)

    def label(self, table: pd.DataFrame, sync_column: str) -> List[PhaseSegment]:
        if len(table) == 0 or not self.phases:
            return []
        times = table[sync_column].to_numpy()
        segments = []
        start_index = 0
        for sequential_phase in self.phases:
            if start_index >= len(table):
                break
            if sequential_phase.trigger is None:
                end_index = len(table)
            else:
                end_index = sequential_phase.trigger.first_index(
                    table=table, start_index=start_index
                )
            if end_index <= start_index:
                continue
            end = (
                int(times[end_index]) if end_index < len(table) else int(times[-1]) + 1
            )
            segments.append(
                PhaseSegment(
                    start=int(times[start_index]),
                    end=end,
                    phase=sequential_phase.phase,
                )
            )
            start_index = end_index
        return segments


AnyPhaseLabeler = Annotated[
    Union[ColumnThresholdLabeler, SequentialTriggerLabeler],
    Field(discriminator="type"),
]


class PhaseLabelingConfig(ConfigModel):
    """A labeling run over a folder of recorded episodes.

    Args:
        recordings_root: Directory containing one folder per episode.
        state_file: State CSV inside each episode driving the labeler.
        labeler: Labeling strategy.
        sync_column: Timestamp column of the state CSV.
        annotations_file: Annotations file name inside each episode.
        exclude_directory_substrings: Directory names containing any of
            these are not treated as episodes.
    """

    recordings_root: str = ""
    state_file: str = ""
    labeler: AnyPhaseLabeler = Field(default_factory=ColumnThresholdLabeler)
    sync_column: str = "time"
    annotations_file: str = ANNOTATIONS_FILE_NAME
    exclude_directory_substrings: List[str] = Field(default_factory=lambda: [".zarr"])


def run_phase_labeling(config: PhaseLabelingConfig) -> Dict[str, int]:
    """Label all episodes under the root, updating their annotations.

    Manual segments are preserved; previous automatic segments are
    replaced.

    Args:
        config: Labeling run configuration.

    Returns:
        Mapping of episode name to the number of automatic segments
        written.
    """
    if not config.recordings_root:
        raise ValueError("recordings_root is required")
    report = {}
    for episode_directory in discover_episode_directories(
        recordings_root=config.recordings_root,
        exclude_substrings=config.exclude_directory_substrings,
    ):
        table = pd.read_csv(episode_directory / config.state_file)
        segments = config.labeler.label(table=table, sync_column=config.sync_column)
        annotations_path = episode_directory / config.annotations_file
        annotations = EpisodeAnnotations.load(path=annotations_path)
        annotations.replace_auto_segments(segments=segments)
        annotations.save(path=annotations_path)
        report[episode_directory.name] = len(segments)
    return report
