"""Parallel episode assembly streamed into a dataset writer."""

from __future__ import annotations

import datetime
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Optional, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm

from tso_sensorium.episodes.dataset_transforms import DatasetTransform
from tso_sensorium.episodes.legend import DatasetMetadata
from tso_sensorium.episodes.schema import DatasetSchema, Episode
from tso_sensorium.export.base import DatasetWriter, DatasetWriterOperation


class BuildPhase(str, Enum):
    """Phase of a dataset build exposed through progress events."""

    DISCOVERING = "discovering"
    ASSEMBLING = "assembling"
    TRANSFORMING = "transforming"
    WRITING = "writing"
    UPDATING = "updating"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BuildProgress:
    """Snapshot of concrete dataset-build progress.

    Args:
        phase: Current build phase.
        completed: Completed work units in the current phase.
        total: Total work units in the current phase.
        current_episode: Episode most recently processed, when applicable.
        failed: Cumulative episode failures observed so far.
    """

    phase: BuildPhase
    completed: int
    total: int
    current_episode: Optional[str] = None
    failed: dict[str, str] = field(default_factory=dict)


class BuildCancellationToken:
    """Thread-safe cooperative cancellation signal for dataset generation."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request cancellation after the current work unit."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""
        return self._event.is_set()


ProgressCallback = Callable[[BuildProgress], None]


@dataclass
class BuildReport:
    """Outcome of a dataset build.

    Args:
        written: Names of episodes written to the dataset, in write order.
        failed: Mapping of episode directory name to failure reason.
        cancelled: Whether generation stopped through cooperative cancellation.
    """

    written: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    cancelled: bool = False


def discover_episode_directories(
    recordings_root: Path | str,
    exclude_substrings: Optional[list[str]] = None,
) -> list[Path]:
    """List episode recording directories under the root, sorted by name.

    Args:
        recordings_root: Directory containing one subdirectory per episode.
        exclude_substrings: Directory names containing any of these are
            skipped (e.g. cache directories living next to the episodes).

    Returns:
        Sorted episode directories.
    """
    root = Path(recordings_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Recordings root not found: {recordings_root}")
    if exclude_substrings is None:
        exclude_substrings = []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and not any(substring in path.name for substring in exclude_substrings)
    )


def _assemble_safely(
    assemble_episode: Callable[[Path], Episode],
    episode_directory: Path,
) -> Tuple[Optional[Episode], Optional[str]]:
    """Assemble one episode, reporting failures instead of raising.

    Episodes with unsalvageable recordings (synchronization gaps, missing
    or unreadable files) are discarded with their failure reason rather
    than aborting the whole build.
    """
    try:
        return assemble_episode(episode_directory), None
    except (ValueError, OSError) as error:
        return None, str(error)


class DatasetBuilder:
    """Assembles recorded episodes and streams them into a dataset writer.

    Episode assembly (alignment, frame extraction, transformations) runs
    in parallel; writing runs sequentially in directory order because
    writers are not required to be safe for concurrent use.

    Args:
        schema: Dataset schema shared by all episodes.
        writer: Destination format writer.
        assemble_episode: Builds one episode from its recording directory.
        n_jobs: Parallel assembly jobs, with joblib semantics (-1 uses all
            cores).
        exclude_substrings: Directory names containing any of these are
            not treated as episodes.
        dataset_metadata: Dataset name, task, phase legend, and any existing
            coordinate-frame metadata merged into the generated dataset metadata.
        dataset_transforms: Transformations computed across all successfully
            assembled episodes before writing.
        progress_callback: Receives immutable progress snapshots from the
            coordinator thread.
        cancellation_token: Cooperative cancellation signal checked between
            work units.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        writer: DatasetWriter,
        assemble_episode: Callable[[Path], Episode],
        n_jobs: int = -1,
        exclude_substrings: Optional[list[str]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        dataset_transforms: Optional[list[DatasetTransform]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        cancellation_token: Optional[BuildCancellationToken] = None,
    ):
        self.schema = schema
        self.writer = writer
        self.assemble_episode = assemble_episode
        self.n_jobs = n_jobs
        self.exclude_substrings = exclude_substrings
        self.dataset_metadata = dataset_metadata
        self.dataset_transforms = dataset_transforms or []
        self.progress_callback = progress_callback
        self.cancellation_token = cancellation_token

    def _emit_progress(
        self,
        phase: BuildPhase,
        completed: int,
        total: int,
        report: BuildReport,
        current_episode: Optional[str] = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            BuildProgress(
                phase=phase,
                completed=completed,
                total=total,
                current_episode=current_episode,
                failed=dict(report.failed),
            )
        )

    def _cancellation_requested(self) -> bool:
        return (
            self.cancellation_token is not None
            and self.cancellation_token.is_cancelled()
        )

    def _cancel_build(
        self,
        report: BuildReport,
        completed: int,
        total: int,
        current_episode: Optional[str] = None,
    ) -> BuildReport:
        self.writer.abort()
        report.cancelled = True
        self._emit_progress(
            phase=BuildPhase.CANCELLED,
            completed=completed,
            total=total,
            report=report,
            current_episode=current_episode,
        )
        return report

    def build(self, recordings_root: Path | str) -> BuildReport:
        """Assemble all episodes under the root and write the dataset.

        Args:
            recordings_root: Directory containing one subdirectory per
                episode.

        Returns:
            Report with written episode names and per-episode failures.
        """
        report = BuildReport()
        self._emit_progress(
            phase=BuildPhase.DISCOVERING,
            completed=0,
            total=0,
            report=report,
        )
        if self._cancellation_requested():
            return self._cancel_build(report=report, completed=0, total=0)
        episode_directories = discover_episode_directories(
            recordings_root=recordings_root,
            exclude_substrings=self.exclude_substrings,
        )
        episode_count = len(episode_directories)
        self._emit_progress(
            phase=BuildPhase.DISCOVERING,
            completed=episode_count,
            total=episode_count,
            report=report,
        )
        if self._cancellation_requested():
            return self._cancel_build(
                report=report,
                completed=episode_count,
                total=episode_count,
            )
        self._emit_progress(
            phase=BuildPhase.ASSEMBLING,
            completed=0,
            total=episode_count,
            report=report,
        )
        if self._cancellation_requested():
            return self._cancel_build(
                report=report,
                completed=0,
                total=episode_count,
            )
        results: Generator = Parallel(n_jobs=self.n_jobs, return_as="generator")(
            delayed(_assemble_safely)(self.assemble_episode, episode_directory)
            for episode_directory in episode_directories
        )
        episodes = []
        with tqdm(total=episode_count) as progress_bar:
            for completed, (episode_directory, result) in enumerate(
                zip(episode_directories, results),
                start=1,
            ):
                episode, failure_reason = result
                if episode is None:
                    report.failed[episode_directory.name] = failure_reason
                    current_episode = episode_directory.name
                else:
                    episodes.append(episode)
                    current_episode = episode.name
                progress_bar.update()
                self._emit_progress(
                    phase=BuildPhase.ASSEMBLING,
                    completed=completed,
                    total=episode_count,
                    report=report,
                    current_episode=current_episode,
                )
                if self._cancellation_requested():
                    results.close()
                    return self._cancel_build(
                        report=report,
                        completed=completed,
                        total=episode_count,
                        current_episode=current_episode,
                    )

        transform_metadata = []
        transform_count = len(self.dataset_transforms)
        self._emit_progress(
            phase=BuildPhase.TRANSFORMING,
            completed=0,
            total=transform_count,
            report=report,
        )
        for completed, transform in enumerate(self.dataset_transforms, start=1):
            episodes = transform.apply(episodes=episodes)
            transform_metadata.append(transform.metadata_payload())
            self._emit_progress(
                phase=BuildPhase.TRANSFORMING,
                completed=completed,
                total=transform_count,
                report=report,
            )
            if self._cancellation_requested():
                return self._cancel_build(
                    report=report,
                    completed=completed,
                    total=transform_count,
                )

        operation = getattr(self.writer, "operation", DatasetWriterOperation.WRITE)
        write_phase = (
            BuildPhase.UPDATING
            if operation == DatasetWriterOperation.UPDATE
            else BuildPhase.WRITING
        )
        try:
            self.writer.open(schema=self.schema)
            write_count = len(episodes)
            self._emit_progress(
                phase=write_phase,
                completed=0,
                total=write_count,
                report=report,
            )
            for completed, episode in enumerate(episodes, start=1):
                if self._cancellation_requested():
                    return self._cancel_build(
                        report=report,
                        completed=completed - 1,
                        total=write_count,
                        current_episode=episode.name,
                    )
                # A bad episode (missing frame, empty task) discards that
                # episode rather than aborting the whole build unfinalized.
                try:
                    self.writer.add_episode(episode=episode)
                except (ValueError, OSError, KeyError) as error:
                    report.failed[episode.name] = str(error)
                else:
                    report.written.append(episode.name)
                self._emit_progress(
                    phase=write_phase,
                    completed=completed,
                    total=write_count,
                    report=report,
                    current_episode=episode.name,
                )
                if self._cancellation_requested():
                    return self._cancel_build(
                        report=report,
                        completed=completed,
                        total=write_count,
                        current_episode=episode.name,
                    )
            metadata = self.dataset_metadata or DatasetMetadata()
            metadata_payload = metadata.to_payload()
            if not metadata_payload["dataset_name"]:
                metadata_payload["dataset_name"] = self.schema.name
            if not metadata_payload["task"] and self.schema.task is not None:
                metadata_payload["task"] = self.schema.task
            if self.schema.coordinate_frame_features:
                metadata_payload["coordinate_frame_features"] = {
                    name: feature.model_dump(mode="json")
                    for name, feature in sorted(
                        self.schema.coordinate_frame_features.items()
                    )
                }
            if transform_metadata:
                metadata_payload["dataset_transforms"] = transform_metadata
            metadata_payload["generation"] = {
                "written": report.written,
                "failed": report.failed,
                "generated_at": datetime.datetime.now().isoformat(),
            }
            self._emit_progress(
                phase=BuildPhase.FINALIZING,
                completed=0,
                total=1,
                report=report,
            )
            if self._cancellation_requested():
                return self._cancel_build(report=report, completed=0, total=1)
            self.writer.write_metadata(metadata=metadata_payload)
            self.writer.finalize()
        except Exception:
            self.writer.abort()
            raise
        self._emit_progress(
            phase=BuildPhase.FINALIZING,
            completed=1,
            total=1,
            report=report,
        )
        self._emit_progress(
            phase=BuildPhase.COMPLETED,
            completed=1,
            total=1,
            report=report,
        )
        return report
