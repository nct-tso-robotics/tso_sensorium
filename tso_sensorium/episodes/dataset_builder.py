"""Parallel episode assembly streamed into a dataset writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

from joblib import Parallel, delayed
from tqdm import tqdm

import datetime

from tso_sensorium.episodes.legend import DatasetMetadata
from tso_sensorium.episodes.schema import DatasetSchema, Episode
from tso_sensorium.export.base import DatasetWriter


@dataclass
class BuildReport:
    """Outcome of a dataset build.

    Args:
        written: Names of episodes written to the dataset, in write order.
        failed: Mapping of episode directory name to failure reason.
    """

    written: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


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
        dataset_metadata: Dataset name, task, and phase legend written
            into the generated dataset.
    """

    def __init__(
        self,
        schema: DatasetSchema,
        writer: DatasetWriter,
        assemble_episode: Callable[[Path], Episode],
        n_jobs: int = -1,
        exclude_substrings: Optional[list[str]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
    ):
        self.schema = schema
        self.writer = writer
        self.assemble_episode = assemble_episode
        self.n_jobs = n_jobs
        self.exclude_substrings = exclude_substrings
        self.dataset_metadata = dataset_metadata

    def build(self, recordings_root: Path | str) -> BuildReport:
        """Assemble all episodes under the root and write the dataset.

        Args:
            recordings_root: Directory containing one subdirectory per
                episode.

        Returns:
            Report with written episode names and per-episode failures.
        """
        episode_directories = discover_episode_directories(
            recordings_root=recordings_root,
            exclude_substrings=self.exclude_substrings,
        )
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_assemble_safely)(self.assemble_episode, episode_directory)
            for episode_directory in tqdm(episode_directories)
        )
        report = BuildReport()
        self.writer.open(schema=self.schema)
        for episode_directory, (episode, failure_reason) in zip(
            episode_directories, results
        ):
            if episode is None:
                report.failed[episode_directory.name] = failure_reason
                continue
            self.writer.add_episode(episode=episode)
            report.written.append(episode.name)
        metadata_payload = (
            self.dataset_metadata.to_payload()
            if self.dataset_metadata is not None
            else {}
        )
        metadata_payload["generation"] = {
            "written": report.written,
            "failed": report.failed,
            "generated_at": datetime.datetime.now().isoformat(),
        }
        self.writer.write_metadata(metadata=metadata_payload)
        self.writer.finalize()
        return report
