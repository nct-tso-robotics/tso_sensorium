"""LeRobot dataset writer. Requires the ``lerobot`` optional dependency."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from tso_sensorium.episodes.schema import DatasetSchema, Episode
from tso_sensorium.export.base import DatasetWriterOperation

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
CAMERA_FEATURE_PREFIX = "observation.images."
TASK_KEY = "task"
IMAGE_DIMENSION_NAMES = ["height", "width", "channels"]
FLOAT_DTYPE = "float32"
VIDEO_DTYPE = "video"
IMAGE_DTYPE = "image"
DATASET_METADATA_FILE_NAME = "dataset_metadata.json"


def build_lerobot_features(schema: DatasetSchema, camera_dtype: str) -> dict[str, dict]:
    """Build LeRobot feature declarations from a Sensorium schema.

    Args:
        schema: Sensorium dataset schema.
        camera_dtype: LeRobot camera storage dtype.

    Returns:
        LeRobot feature declarations excluding automatically generated fields.
    """
    features = {
        STATE_FEATURE: {
            "dtype": FLOAT_DTYPE,
            "shape": (len(schema.state_columns),),
            "names": schema.state_columns,
        },
        ACTION_FEATURE: {
            "dtype": FLOAT_DTYPE,
            "shape": (len(schema.action_columns),),
            "names": schema.action_columns,
        },
    }
    for name, feature in schema.auxiliary_features.items():
        features[name] = {
            "dtype": feature.dtype,
            "shape": (len(feature.columns),),
            "names": feature.columns,
        }
    for camera in schema.cameras:
        features[f"{CAMERA_FEATURE_PREFIX}{camera.name}"] = {
            "dtype": camera_dtype,
            "shape": (camera.height, camera.width, 3),
            "names": IMAGE_DIMENSION_NAMES,
        }
    return features


def resolve_episode_tasks(
    episode: Episode,
    schema: DatasetSchema,
    task_column: Optional[str],
) -> list[str]:
    """Resolve one task string for every row of an episode.

    Args:
        episode: Episode whose task values are resolved.
        schema: Dataset schema supplying the default task.
        task_column: Optional per-row task column.

    Returns:
        Task string for each episode row.

    Raises:
        ValueError: If a row needs a default and neither the episode nor schema
            defines one.
    """
    task_values = None
    if task_column is not None and task_column in episode.table:
        task_values = episode.table[task_column].fillna("").astype(str).tolist()
    default_task = episode.task if episode.task is not None else schema.task
    if (task_values is None or any(value == "" for value in task_values)) and (
        default_task is None
    ):
        raise ValueError(
            f"Episode {episode.name} has no task and the schema does not define one"
        )
    if task_values is None:
        return [default_task] * len(episode.table)
    return [default_task if value == "" else value for value in task_values]


class LeRobotDatasetWriter:
    """Writes episodes into a LeRobot dataset.

    Camera frames are loaded from the paths referenced in the episode
    table and stored through LeRobot's image or video pipeline; arm state
    and action columns are concatenated into flat float vectors. Named
    auxiliary features retain their configured numeric dtype.

    Args:
        repo_id: Dataset repository identifier, e.g. "user/dataset".
        output_root: Local directory for dataset storage.
        use_videos: Whether to encode camera streams as videos instead of
            individual images.
        task_column: Episode table column holding per-frame task strings;
            rows with empty values fall back to the episode or schema
            task.
    """

    operation = DatasetWriterOperation.WRITE

    def __init__(
        self,
        repo_id: str,
        output_root: Path | str,
        use_videos: bool = True,
        task_column: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.output_root = Path(output_root)
        self.use_videos = use_videos
        self.task_column = task_column
        self._dataset: Optional[LeRobotDataset] = None
        self._schema: Optional[DatasetSchema] = None
        self._owns_output_root = False

    def _build_features(self, schema: DatasetSchema) -> dict:
        camera_dtype = VIDEO_DTYPE if self.use_videos else IMAGE_DTYPE
        return build_lerobot_features(schema=schema, camera_dtype=camera_dtype)

    def open(self, schema: DatasetSchema) -> None:
        """Create the LeRobot dataset for the given schema."""
        if not schema.arms:
            raise ValueError("LeRobot export requires at least one arm in the schema")
        output_root_existed = self.output_root.exists()
        self._owns_output_root = not output_root_existed
        self._schema = schema
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=schema.fps,
            features=self._build_features(schema=schema),
            root=self.output_root,
            use_videos=self.use_videos,
        )

    @staticmethod
    def _load_frame(frame_path: Path | str) -> np.ndarray:
        image = cv2.imread(str(frame_path))
        if image is None:
            raise FileNotFoundError(f"Could not read frame {frame_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def add_episode(self, episode: Episode) -> None:
        """Append one episode, loading camera frames from disk."""
        if self._dataset is None or self._schema is None:
            raise RuntimeError(
                "Writer must be opened with a schema before adding episodes"
            )
        schema = self._schema
        schema.validate_episode_table(table=episode.table)
        task_values = resolve_episode_tasks(
            episode=episode,
            schema=schema,
            task_column=self.task_column,
        )
        state_values = episode.table[schema.state_columns].to_numpy(dtype=np.float32)
        action_values = episode.table[schema.action_columns].to_numpy(dtype=np.float32)
        auxiliary_values = {
            name: episode.table[feature.columns].to_numpy(dtype=feature.dtype)
            for name, feature in schema.auxiliary_features.items()
        }
        for row_index in range(len(episode.table)):
            frame = {
                TASK_KEY: task_values[row_index],
                STATE_FEATURE: state_values[row_index],
                ACTION_FEATURE: action_values[row_index],
            }
            for name, values in auxiliary_values.items():
                frame[name] = values[row_index]
            for camera in schema.cameras:
                frame_path = episode.table[camera.frame_column].iloc[row_index]
                frame[f"{CAMERA_FEATURE_PREFIX}{camera.name}"] = self._load_frame(
                    frame_path=frame_path
                )
            self._dataset.add_frame(frame)
        self._dataset.save_episode()

    def write_metadata(self, metadata: dict) -> None:
        """Write the dataset metadata JSON next to the LeRobot dataset."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / DATASET_METADATA_FILE_NAME).write_text(
            json.dumps(metadata, indent=2)
        )

    def finalize(self) -> None:
        """Finalize the LeRobot dataset metadata."""
        if self._dataset is not None:
            self._dataset.finalize()
        self._owns_output_root = False

    def abort(self) -> None:
        """Discard incomplete output only when this writer created its root."""
        if self._dataset is not None and self._dataset.writer is not None:
            self._dataset.writer.cancel_pending_videos()
            self._dataset.writer.close_writer()
            self._dataset.writer._finalized = True
            self._dataset._is_finalized = True
        if self._owns_output_root and self.output_root.is_dir():
            shutil.rmtree(self.output_root)
        self._owns_output_root = False
        self._dataset = None
