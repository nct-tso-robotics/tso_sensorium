"""Transactional action-only updates for existing LeRobot v3 datasets."""

from __future__ import annotations

import ctypes
import fcntl
import json
import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from tso_sensorium.episodes.schema import DatasetSchema, Episode
from tso_sensorium.export.base import DatasetWriterOperation
from tso_sensorium.export.lerobot_writer import (
    ACTION_FEATURE,
    CAMERA_FEATURE_PREFIX,
    DATASET_METADATA_FILE_NAME,
    STATE_FEATURE,
    VIDEO_DTYPE,
    build_lerobot_features,
    resolve_episode_tasks,
)

LEROBOT_V3_VERSION = "v3.0"
INFO_PATH = Path("meta/info.json")
STATS_PATH = Path("meta/stats.json")
TASKS_PATH = Path("meta/tasks.parquet")
EPISODES_DIRECTORY = Path("meta/episodes")
EPISODES_PATH_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)
LOCAL_VALIDATION_REPOSITORY = "local/action-update-validation"
ACTION_STATS_PREFIX = f"stats/{ACTION_FEATURE}/"
DEFAULT_FEATURES = {
    "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
    "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
    "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
    "index": {"dtype": "int64", "shape": (1,), "names": None},
    "task_index": {"dtype": "int64", "shape": (1,), "names": None},
}
AT_FDCWD = -100
RENAME_EXCHANGE = 2

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, list["JsonValue"], dict[str, "JsonValue"]]
JsonInput = Union[
    JsonScalar,
    np.ndarray,
    np.generic,
    list["JsonInput"],
    tuple["JsonInput", ...],
    dict[str, "JsonInput"],
]


class LeRobotDatasetCompatibilityError(RuntimeError):
    """Raised when a dataset cannot be updated without changing other data."""


@dataclass(frozen=True)
class _FileIdentity:
    size: int
    modified_nanoseconds: int
    inode: int
    mode: int


@dataclass(frozen=True)
class _EpisodeLocation:
    episode_index: int
    name: str
    length: int
    data_path: Path
    dataset_from_index: int
    dataset_to_index: int
    tasks: tuple[str, ...]


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise LeRobotDatasetCompatibilityError(
            f"Expected a JSON object in {path}, got {type(payload).__name__}"
        )
    return payload


def _json_compatible(value: JsonInput) -> JsonValue:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _write_new_json(path: Path, payload: dict, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary_path = Path(temporary_file.name)
    with temporary_file:
        json.dump(_json_compatible(payload), temporary_file, indent=indent)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)


def _capture_manifest(root: Path) -> dict[str, _FileIdentity]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() and not path.is_symlink():
            continue
        metadata = path.lstat()
        manifest[path.relative_to(root).as_posix()] = _FileIdentity(
            size=metadata.st_size,
            modified_nanoseconds=metadata.st_mtime_ns,
            inode=metadata.st_ino,
            mode=metadata.st_mode,
        )
    return manifest


def _atomic_exchange_directories(first: Path, second: Path) -> None:
    if not sys.platform.startswith("linux"):
        raise RuntimeError(
            "Atomic LeRobot action updates require Linux renameat2 support"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "Atomic LeRobot action updates require renameat2(RENAME_EXCHANGE)"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(first),
        AT_FDCWD,
        os.fsencode(second),
        RENAME_EXCHANGE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            "Atomic directory exchange failed: " + os.strerror(error_number),
            f"{first} <-> {second}",
        )


class LeRobotActionUpdateWriter:
    """Update actions and their metadata in an existing LeRobot dataset.

    The writer validates every regenerated episode against the existing
    observations, auxiliary values, tasks, indices, schema, and episode order.
    It then builds a hardlinked sibling staging tree, replaces action-bearing
    files with new inodes, and atomically exchanges the complete directories.

    Args:
        dataset_root: Existing LeRobot v3 dataset root.
        task_column: Optional episode-table column holding per-row task strings.
    """

    operation = DatasetWriterOperation.UPDATE

    def __init__(
        self,
        dataset_root: Path | str,
        task_column: Optional[str] = None,
    ):
        self.dataset_root = Path(dataset_root)
        self.task_column = task_column
        self._schema: Optional[DatasetSchema] = None
        self._info: Optional[dict] = None
        self._source_metadata: Optional[dict] = None
        self._pending_metadata: Optional[dict] = None
        self._locations: list[_EpisodeLocation] = []
        self._task_by_index: dict[int, str] = {}
        self._actions: dict[int, np.ndarray] = {}
        self._action_stats: dict[int, dict] = {}
        self._next_episode_index = 0
        self._manifest: Optional[dict[str, _FileIdentity]] = None
        self._stage_container: Optional[Path] = None
        self._lock_file_descriptor: Optional[int] = None
        self._committed = False
        self._cached_data_path: Optional[Path] = None
        self._cached_data_table: Optional[pa.Table] = None

    def _acquire_lock(self) -> None:
        lock_file_descriptor = os.open(self.dataset_root, os.O_RDONLY)
        try:
            fcntl.flock(
                lock_file_descriptor,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            os.close(lock_file_descriptor)
            raise LeRobotDatasetCompatibilityError(
                f"Another action update is using {self.dataset_root}"
            ) from error
        self._lock_file_descriptor = lock_file_descriptor

    def _release_lock(self) -> None:
        if self._lock_file_descriptor is None:
            return
        fcntl.flock(self._lock_file_descriptor, fcntl.LOCK_UN)
        os.close(self._lock_file_descriptor)
        self._lock_file_descriptor = None

    def _require_file(self, relative_path: Path) -> Path:
        path = self.dataset_root / relative_path
        if not path.is_file():
            raise LeRobotDatasetCompatibilityError(
                f"LeRobot dataset is missing required file {relative_path}"
            )
        return path

    def _validate_features(self, schema: DatasetSchema) -> None:
        if self._info is None:
            raise RuntimeError("LeRobot dataset info is not loaded")
        expected_features = {
            **build_lerobot_features(schema=schema, camera_dtype=VIDEO_DTYPE),
            **DEFAULT_FEATURES,
        }
        actual_features = self._info.get("features")
        if not isinstance(actual_features, dict):
            raise LeRobotDatasetCompatibilityError(
                "LeRobot info.json does not define a feature mapping"
            )
        if set(actual_features) != set(expected_features):
            raise LeRobotDatasetCompatibilityError(
                "LeRobot feature keys do not match the generation schema: "
                f"existing={sorted(actual_features)}, "
                f"generated={sorted(expected_features)}"
            )
        for key, expected in expected_features.items():
            actual = actual_features[key]
            comparable_actual = {
                "dtype": actual.get("dtype"),
                "shape": tuple(actual.get("shape", [])),
                "names": actual.get("names"),
            }
            comparable_expected = {
                "dtype": expected["dtype"],
                "shape": tuple(expected["shape"]),
                "names": expected["names"],
            }
            if comparable_actual != comparable_expected:
                raise LeRobotDatasetCompatibilityError(
                    f"LeRobot feature '{key}' does not match the generation "
                    f"schema: existing={comparable_actual}, "
                    f"generated={comparable_expected}"
                )
        inline_images = [
            key
            for key, feature in actual_features.items()
            if key.startswith(CAMERA_FEATURE_PREFIX) and feature["dtype"] != VIDEO_DTYPE
        ]
        if inline_images:
            raise LeRobotDatasetCompatibilityError(
                "Action-only update does not rewrite inline image parquet data: "
                f"{inline_images}"
            )

    def _load_episode_locations(self) -> list[_EpisodeLocation]:
        if self._info is None or self._source_metadata is None:
            raise RuntimeError("LeRobot dataset metadata is not loaded")
        generation = self._source_metadata.get("generation")
        written = generation.get("written") if isinstance(generation, dict) else None
        if not isinstance(written, list) or not all(
            isinstance(name, str) for name in written
        ):
            raise LeRobotDatasetCompatibilityError(
                "dataset_metadata.json must contain generation.written episode names"
            )
        total_episodes = self._info.get("total_episodes")
        if len(written) != total_episodes:
            raise LeRobotDatasetCompatibilityError(
                "dataset_metadata.json episode count does not match info.json: "
                f"{len(written)} != {total_episodes}"
            )
        episode_files = sorted(
            (self.dataset_root / EPISODES_DIRECTORY).rglob("*.parquet")
        )
        if not episode_files:
            raise LeRobotDatasetCompatibilityError(
                "LeRobot dataset has no episode metadata parquet files"
            )
        locations_by_index = {}
        data_path_template = self._info.get("data_path")
        if not isinstance(data_path_template, str):
            raise LeRobotDatasetCompatibilityError(
                "LeRobot info.json does not define data_path"
            )
        for episode_file in episode_files:
            table = pq.read_table(episode_file)
            required_columns = {
                "episode_index",
                "length",
                "tasks",
                "data/chunk_index",
                "data/file_index",
                "dataset_from_index",
                "dataset_to_index",
                "meta/episodes/chunk_index",
                "meta/episodes/file_index",
            }
            missing_columns = sorted(required_columns - set(table.column_names))
            if missing_columns:
                raise LeRobotDatasetCompatibilityError(
                    f"Episode metadata {episode_file} is missing columns "
                    f"{missing_columns}"
                )
            action_stat_columns = {
                name
                for name in table.column_names
                if name.startswith(ACTION_STATS_PREFIX)
            }
            if not action_stat_columns:
                raise LeRobotDatasetCompatibilityError(
                    f"Episode metadata {episode_file} has no action statistics"
                )
            values = table.select(sorted(required_columns)).to_pydict()
            for row_index, episode_index_value in enumerate(values["episode_index"]):
                episode_index = int(episode_index_value)
                if episode_index in locations_by_index:
                    raise LeRobotDatasetCompatibilityError(
                        f"Duplicate episode_index {episode_index} in episode metadata"
                    )
                if episode_index < 0 or episode_index >= len(written):
                    raise LeRobotDatasetCompatibilityError(
                        f"Episode metadata contains out-of-range index {episode_index}"
                    )
                metadata_chunk = int(values["meta/episodes/chunk_index"][row_index])
                metadata_file = int(values["meta/episodes/file_index"][row_index])
                expected_metadata_path = (
                    self.dataset_root
                    / EPISODES_PATH_TEMPLATE.format(
                        chunk_index=metadata_chunk,
                        file_index=metadata_file,
                    )
                )
                if episode_file != expected_metadata_path:
                    raise LeRobotDatasetCompatibilityError(
                        f"Episode {episode_index} points to {expected_metadata_path}, "
                        f"but is stored in {episode_file}"
                    )
                data_relative_path = Path(
                    data_path_template.format(
                        chunk_index=int(values["data/chunk_index"][row_index]),
                        file_index=int(values["data/file_index"][row_index]),
                    )
                )
                if data_relative_path.is_absolute() or ".." in data_relative_path.parts:
                    raise LeRobotDatasetCompatibilityError(
                        f"Unsafe LeRobot data path {data_relative_path}"
                    )
                self._require_file(relative_path=data_relative_path)
                length = int(values["length"][row_index])
                if length <= 0:
                    raise LeRobotDatasetCompatibilityError(
                        f"Episode {episode_index} has non-positive length {length}"
                    )
                dataset_from_index = int(values["dataset_from_index"][row_index])
                dataset_to_index = int(values["dataset_to_index"][row_index])
                if dataset_to_index - dataset_from_index != length:
                    raise LeRobotDatasetCompatibilityError(
                        f"Episode {episode_index} length does not match its dataset range"
                    )
                locations_by_index[episode_index] = _EpisodeLocation(
                    episode_index=episode_index,
                    name=written[episode_index],
                    length=length,
                    data_path=data_relative_path,
                    dataset_from_index=dataset_from_index,
                    dataset_to_index=dataset_to_index,
                    tasks=tuple(str(task) for task in values["tasks"][row_index]),
                )
        expected_indices = set(range(len(written)))
        if set(locations_by_index) != expected_indices:
            raise LeRobotDatasetCompatibilityError(
                "Episode metadata indices are incomplete: "
                f"existing={sorted(locations_by_index)}, "
                f"expected={sorted(expected_indices)}"
            )
        locations = [locations_by_index[index] for index in range(len(written))]
        expected_from_index = 0
        for location in locations:
            if location.dataset_from_index != expected_from_index:
                raise LeRobotDatasetCompatibilityError(
                    f"Episode {location.episode_index} starts at "
                    f"{location.dataset_from_index}, expected {expected_from_index}"
                )
            expected_from_index = location.dataset_to_index
        if expected_from_index != self._info.get("total_frames"):
            raise LeRobotDatasetCompatibilityError(
                "Episode lengths do not sum to info.json total_frames"
            )
        return locations

    def _load_task_mapping(self) -> dict[int, str]:
        tasks = pd.read_parquet(self._require_file(relative_path=TASKS_PATH))
        if "task_index" not in tasks.columns:
            raise LeRobotDatasetCompatibilityError(
                "LeRobot tasks.parquet is missing task_index"
            )
        mapping = {
            int(task_index): str(task)
            for task, task_index in tasks["task_index"].items()
        }
        if len(mapping) != len(tasks):
            raise LeRobotDatasetCompatibilityError(
                "LeRobot tasks.parquet contains duplicate task indices"
            )
        return mapping

    def open(self, schema: DatasetSchema) -> None:
        """Open and validate the existing LeRobot dataset.

        Args:
            schema: Schema used to regenerate the episode tables.
        """
        if not self.dataset_root.is_dir() or self.dataset_root.is_symlink():
            raise LeRobotDatasetCompatibilityError(
                f"Existing LeRobot dataset root is required: {self.dataset_root}"
            )
        self._acquire_lock()
        try:
            self._schema = schema
            self._info = _read_json(self._require_file(relative_path=INFO_PATH))
            if self._info.get("codebase_version") != LEROBOT_V3_VERSION:
                raise LeRobotDatasetCompatibilityError(
                    "Action-only update requires LeRobot v3.0, got "
                    f"{self._info.get('codebase_version')}"
                )
            if self._info.get("fps") != schema.fps:
                raise LeRobotDatasetCompatibilityError(
                    f"LeRobot fps {self._info.get('fps')} does not match schema "
                    f"fps {schema.fps}"
                )
            self._source_metadata = _read_json(
                self._require_file(relative_path=Path(DATASET_METADATA_FILE_NAME))
            )
            self._require_file(relative_path=STATS_PATH)
            self._validate_features(schema=schema)
            self._task_by_index = self._load_task_mapping()
            self._locations = self._load_episode_locations()
            self._manifest = _capture_manifest(root=self.dataset_root)
        except Exception:
            self.abort()
            raise

    def _load_data_table(self, relative_path: Path) -> pa.Table:
        if self._cached_data_path != relative_path:
            self._cached_data_table = pq.read_table(self.dataset_root / relative_path)
            self._cached_data_path = relative_path
        if self._cached_data_table is None:
            raise RuntimeError(f"Failed to load LeRobot data file {relative_path}")
        return self._cached_data_table

    @staticmethod
    def _numeric_column(table: pa.Table, name: str) -> np.ndarray:
        return np.asarray(table[name].combine_chunks().to_pylist())

    @staticmethod
    def _require_equal(
        episode_name: str,
        feature_name: str,
        existing: np.ndarray,
        regenerated: np.ndarray,
    ) -> None:
        if existing.shape != regenerated.shape or not np.array_equal(
            existing,
            regenerated,
            equal_nan=True,
        ):
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode_name} changed non-action feature '{feature_name}'"
            )

    def add_episode(self, episode: Episode) -> None:
        """Validate one regenerated episode and retain its transformed actions."""
        if self._schema is None or self._info is None:
            raise RuntimeError("Writer must be opened before adding episodes")
        if self._next_episode_index >= len(self._locations):
            raise LeRobotDatasetCompatibilityError(
                f"Generated extra episode {episode.name}"
            )
        location = self._locations[self._next_episode_index]
        if episode.name != location.name:
            raise LeRobotDatasetCompatibilityError(
                f"Episode order mismatch at index {location.episode_index}: "
                f"existing={location.name}, generated={episode.name}"
            )
        self._schema.validate_episode_table(table=episode.table)
        if len(episode.table) != location.length:
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} length changed from {location.length} to "
                f"{len(episode.table)}"
            )
        table = self._load_data_table(relative_path=location.data_path)
        episode_indices = self._numeric_column(table=table, name="episode_index")
        row_indices = np.flatnonzero(episode_indices == location.episode_index)
        if len(row_indices) != location.length or not np.array_equal(
            row_indices,
            np.arange(row_indices[0], row_indices[0] + location.length),
        ):
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} rows are not a contiguous block of the "
                "declared length"
            )
        existing_frame_indices = self._numeric_column(table=table, name="frame_index")[
            row_indices
        ]
        expected_frame_indices = np.arange(location.length, dtype=np.int64)
        self._require_equal(
            episode_name=episode.name,
            feature_name="frame_index",
            existing=existing_frame_indices,
            regenerated=expected_frame_indices,
        )
        existing_global_indices = self._numeric_column(table=table, name="index")[
            row_indices
        ]
        expected_global_indices = np.arange(
            location.dataset_from_index,
            location.dataset_to_index,
            dtype=np.int64,
        )
        self._require_equal(
            episode_name=episode.name,
            feature_name="index",
            existing=existing_global_indices,
            regenerated=expected_global_indices,
        )
        existing_timestamps = self._numeric_column(table=table, name="timestamp")[
            row_indices
        ]
        expected_timestamps = (
            np.arange(location.length, dtype=np.float32) / self._schema.fps
        ).astype(np.float32)
        self._require_equal(
            episode_name=episode.name,
            feature_name="timestamp",
            existing=existing_timestamps,
            regenerated=expected_timestamps,
        )
        existing_states = self._numeric_column(table=table, name=STATE_FEATURE)[
            row_indices
        ].astype(np.float32)
        regenerated_states = episode.table[self._schema.state_columns].to_numpy(
            dtype=np.float32
        )
        self._require_equal(
            episode_name=episode.name,
            feature_name=STATE_FEATURE,
            existing=existing_states,
            regenerated=regenerated_states,
        )
        for name, feature in self._schema.auxiliary_features.items():
            existing_values = self._numeric_column(table=table, name=name)[row_indices]
            regenerated_values = episode.table[feature.columns].to_numpy(
                dtype=feature.dtype
            )
            if existing_values.ndim == 1 and regenerated_values.shape[1] == 1:
                regenerated_values = regenerated_values[:, 0]
            self._require_equal(
                episode_name=episode.name,
                feature_name=name,
                existing=existing_values,
                regenerated=regenerated_values,
            )
        task_indices = self._numeric_column(table=table, name="task_index")[row_indices]
        missing_task_indices = sorted(
            {
                int(index)
                for index in task_indices
                if int(index) not in self._task_by_index
            }
        )
        if missing_task_indices:
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} references unknown task indices "
                f"{missing_task_indices}"
            )
        existing_tasks = [self._task_by_index[int(index)] for index in task_indices]
        regenerated_tasks = resolve_episode_tasks(
            episode=episode,
            schema=self._schema,
            task_column=self.task_column,
        )
        if existing_tasks != regenerated_tasks:
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} changed non-action task strings"
            )
        if set(existing_tasks) != set(location.tasks):
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} task metadata does not match frame tasks"
            )
        actions = episode.table[self._schema.action_columns].to_numpy(dtype=np.float32)
        if actions.shape != (location.length, len(self._schema.action_columns)):
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} action shape is {actions.shape}, expected "
                f"{(location.length, len(self._schema.action_columns))}"
            )
        if not np.isfinite(actions).all():
            raise LeRobotDatasetCompatibilityError(
                f"Episode {episode.name} actions contain non-finite values"
            )
        action_feature = self._info["features"][ACTION_FEATURE]
        episode_stats = compute_episode_stats(
            episode_data={ACTION_FEATURE: actions},
            features={ACTION_FEATURE: action_feature},
        )[ACTION_FEATURE]
        self._actions[location.episode_index] = actions
        self._action_stats[location.episode_index] = episode_stats
        self._next_episode_index += 1

    def write_metadata(self, metadata: dict) -> None:
        """Store generated provenance until the atomic commit."""
        self._pending_metadata = json.loads(json.dumps(metadata))

    def _assert_target_unchanged(self) -> None:
        if self._manifest is None:
            raise RuntimeError("Target manifest is not available")
        current_manifest = _capture_manifest(root=self.dataset_root)
        if current_manifest != self._manifest:
            raise LeRobotDatasetCompatibilityError(
                "LeRobot dataset changed while the action update was being prepared"
            )

    def _stage_dataset(self) -> Path:
        self._stage_container = Path(
            tempfile.mkdtemp(
                prefix=f".{self.dataset_root.name}.action-update-",
                dir=self.dataset_root.parent,
            )
        )
        stage_root = self._stage_container / "dataset"
        shutil.copytree(
            self.dataset_root,
            stage_root,
            copy_function=os.link,
            symlinks=True,
        )
        self._assert_target_unchanged()
        return stage_root

    @staticmethod
    def _rewrite_parquet(
        source_path: Path,
        destination_path: Path,
        transform: Callable[[pa.Table], pa.Table],
    ) -> None:
        source = pq.ParquetFile(source_path)
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            dir=destination_path.parent,
            delete=False,
        )
        temporary_path = Path(temporary_file.name)
        temporary_file.close()
        with pq.ParquetWriter(
            temporary_path,
            schema=source.schema_arrow,
            compression="snappy",
            use_dictionary=True,
        ) as writer:
            for row_group_index in range(source.num_row_groups):
                row_group = source.read_row_group(row_group_index)
                writer.write_table(transform(row_group))
        os.replace(temporary_path, destination_path)

    def _replace_action_column(self, table: pa.Table) -> pa.Table:
        episode_indices = self._numeric_column(table=table, name="episode_index")
        frame_indices = self._numeric_column(table=table, name="frame_index")
        action_dimension = len(self._schema.action_columns)
        actions = np.empty((len(table), action_dimension), dtype=np.float32)
        for row_index, (episode_index, frame_index) in enumerate(
            zip(episode_indices, frame_indices)
        ):
            episode_actions = self._actions.get(int(episode_index))
            if episode_actions is None or int(frame_index) >= len(episode_actions):
                raise LeRobotDatasetCompatibilityError(
                    f"No regenerated action for episode {episode_index}, frame "
                    f"{frame_index}"
                )
            actions[row_index] = episode_actions[int(frame_index)]
        column_index = table.schema.get_field_index(ACTION_FEATURE)
        field = table.schema.field(column_index)
        action_array = pa.array(actions.tolist(), type=field.type)
        return table.set_column(column_index, field, action_array)

    def _replace_episode_action_stats(self, table: pa.Table) -> pa.Table:
        episode_indices = self._numeric_column(table=table, name="episode_index")
        transformed = table
        action_stat_columns = [
            name for name in table.column_names if name.startswith(ACTION_STATS_PREFIX)
        ]
        expected_stat_names = {
            f"{ACTION_STATS_PREFIX}{name}"
            for name in next(iter(self._action_stats.values()))
        }
        if set(action_stat_columns) != expected_stat_names:
            raise LeRobotDatasetCompatibilityError(
                "Stored per-episode action statistic fields do not match LeRobot "
                f"statistics: existing={sorted(action_stat_columns)}, "
                f"expected={sorted(expected_stat_names)}"
            )
        for column_name in action_stat_columns:
            stat_name = column_name.removeprefix(ACTION_STATS_PREFIX)
            values = []
            for episode_index in episode_indices:
                episode_stats = self._action_stats.get(int(episode_index))
                if episode_stats is None:
                    raise LeRobotDatasetCompatibilityError(
                        f"No regenerated statistics for episode {episode_index}"
                    )
                values.append(episode_stats[stat_name].tolist())
            column_index = transformed.schema.get_field_index(column_name)
            field = transformed.schema.field(column_index)
            transformed = transformed.set_column(
                column_index,
                field,
                pa.array(values, type=field.type),
            )
        return transformed

    @staticmethod
    def _assert_non_action_columns_unchanged(
        source_path: Path, staged_path: Path
    ) -> None:
        source = pq.read_table(source_path).combine_chunks()
        staged = pq.read_table(staged_path).combine_chunks()
        if source.schema != staged.schema or source.num_rows != staged.num_rows:
            raise LeRobotDatasetCompatibilityError(
                f"Staged parquet schema changed for {source_path}"
            )
        for column_name in source.column_names:
            if column_name == ACTION_FEATURE or column_name.startswith(
                ACTION_STATS_PREFIX
            ):
                continue
            if not source[column_name].equals(staged[column_name]):
                raise LeRobotDatasetCompatibilityError(
                    f"Staged parquet changed non-action column '{column_name}' in "
                    f"{source_path}"
                )

    def _rewrite_data_files(self, stage_root: Path) -> None:
        data_paths = sorted({location.data_path for location in self._locations})
        for relative_path in data_paths:
            source_path = self.dataset_root / relative_path
            staged_path = stage_root / relative_path
            self._rewrite_parquet(
                source_path=source_path,
                destination_path=staged_path,
                transform=self._replace_action_column,
            )
            self._assert_non_action_columns_unchanged(
                source_path=source_path,
                staged_path=staged_path,
            )

    def _rewrite_episode_stats(self, stage_root: Path) -> None:
        episode_paths = sorted(
            path.relative_to(self.dataset_root)
            for path in (self.dataset_root / EPISODES_DIRECTORY).rglob("*.parquet")
        )
        for relative_path in episode_paths:
            source_path = self.dataset_root / relative_path
            staged_path = stage_root / relative_path
            self._rewrite_parquet(
                source_path=source_path,
                destination_path=staged_path,
                transform=self._replace_episode_action_stats,
            )
            self._assert_non_action_columns_unchanged(
                source_path=source_path,
                staged_path=staged_path,
            )

    def _rewrite_global_stats(self, stage_root: Path) -> dict:
        global_stats = _read_json(self.dataset_root / STATS_PATH)
        action_stats = aggregate_stats(
            [
                {ACTION_FEATURE: self._action_stats[index]}
                for index in range(len(self._locations))
            ]
        )[ACTION_FEATURE]
        if ACTION_FEATURE not in global_stats:
            raise LeRobotDatasetCompatibilityError(
                "LeRobot global statistics do not contain action"
            )
        global_stats[ACTION_FEATURE] = action_stats
        _write_new_json(
            path=stage_root / STATS_PATH,
            payload=global_stats,
            indent=4,
        )
        return action_stats

    def _validate_staged_dataset(
        self,
        stage_root: Path,
        expected_global_action_stats: dict,
    ) -> None:
        for location in self._locations:
            table = pq.read_table(stage_root / location.data_path)
            episode_indices = self._numeric_column(table=table, name="episode_index")
            row_indices = np.flatnonzero(episode_indices == location.episode_index)
            staged_actions = self._numeric_column(table=table, name=ACTION_FEATURE)[
                row_indices
            ].astype(np.float32)
            if not np.array_equal(
                staged_actions, self._actions[location.episode_index]
            ):
                raise LeRobotDatasetCompatibilityError(
                    f"Staged actions do not match episode {location.name}"
                )
        staged_episode_tables = [
            pq.read_table(path)
            for path in sorted((stage_root / EPISODES_DIRECTORY).rglob("*.parquet"))
        ]
        staged_episode_metadata = pa.concat_tables(staged_episode_tables)
        staged_episode_indices = self._numeric_column(
            table=staged_episode_metadata,
            name="episode_index",
        )
        for location in self._locations:
            row_indices = np.flatnonzero(
                staged_episode_indices == location.episode_index
            )
            if len(row_indices) != 1:
                raise LeRobotDatasetCompatibilityError(
                    f"Staged metadata has {len(row_indices)} rows for episode "
                    f"{location.name}"
                )
            row_index = int(row_indices[0])
            for name, expected in self._action_stats[location.episode_index].items():
                column_name = f"{ACTION_STATS_PREFIX}{name}"
                actual = np.asarray(
                    staged_episode_metadata[column_name][row_index].as_py()
                )
                if not np.allclose(actual, expected, rtol=0.0, atol=1e-15):
                    raise LeRobotDatasetCompatibilityError(
                        f"Staged action statistic '{name}' is incorrect for "
                        f"episode {location.name}"
                    )
        staged_stats = _read_json(stage_root / STATS_PATH)[ACTION_FEATURE]
        for name, expected in expected_global_action_stats.items():
            actual = np.asarray(staged_stats[name])
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-15):
                raise LeRobotDatasetCompatibilityError(
                    f"Staged global action statistic '{name}' is incorrect"
                )
        staged_metadata = _read_json(stage_root / DATASET_METADATA_FILE_NAME)
        if staged_metadata != self._pending_metadata:
            raise LeRobotDatasetCompatibilityError(
                "Staged dataset provenance does not match generated metadata"
            )
        dataset = LeRobotDataset(
            repo_id=LOCAL_VALIDATION_REPOSITORY,
            root=stage_root,
            download_videos=False,
        )
        if dataset.num_episodes != len(self._locations) or dataset.num_frames != sum(
            location.length for location in self._locations
        ):
            raise LeRobotDatasetCompatibilityError(
                "Staged LeRobot dataset count validation failed"
            )
        del dataset

    def finalize(self) -> None:
        """Stage, validate, and atomically commit the action-only update."""
        if self._schema is None or self._info is None:
            raise RuntimeError("Writer must be opened before finalizing")
        if self._next_episode_index != len(self._locations):
            raise LeRobotDatasetCompatibilityError(
                f"Generated {self._next_episode_index} episodes, but the existing "
                f"dataset contains {len(self._locations)}"
            )
        if self._pending_metadata is None:
            raise RuntimeError("Dataset metadata must be provided before finalizing")
        generated_names = self._pending_metadata.get("generation", {}).get("written")
        expected_names = [location.name for location in self._locations]
        if generated_names != expected_names:
            raise LeRobotDatasetCompatibilityError(
                "Generated metadata episode order does not match the existing dataset"
            )
        self._assert_target_unchanged()
        stage_root = self._stage_dataset()
        self._rewrite_data_files(stage_root=stage_root)
        self._rewrite_episode_stats(stage_root=stage_root)
        global_action_stats = self._rewrite_global_stats(stage_root=stage_root)
        _write_new_json(
            path=stage_root / DATASET_METADATA_FILE_NAME,
            payload=self._pending_metadata,
            indent=2,
        )
        self._validate_staged_dataset(
            stage_root=stage_root,
            expected_global_action_stats=global_action_stats,
        )
        self._assert_target_unchanged()
        _atomic_exchange_directories(self.dataset_root, stage_root)
        self._committed = True
        try:
            self._discard_stage()
        finally:
            self._release_lock()

    def _discard_stage(self) -> None:
        if self._stage_container is None:
            return
        stage_container = self._stage_container
        try:
            shutil.rmtree(stage_container)
        except OSError as error:
            logging.warning(
                "Could not remove LeRobot action-update staging at %s: %s",
                stage_container,
                error,
            )
            return
        self._stage_container = None

    def abort(self) -> None:
        """Discard staging state and leave the existing dataset untouched."""
        self._cached_data_table = None
        self._cached_data_path = None
        try:
            self._discard_stage()
        finally:
            self._release_lock()
