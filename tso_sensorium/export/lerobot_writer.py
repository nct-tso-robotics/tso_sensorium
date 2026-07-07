"""LeRobot dataset writer. Requires the ``lerobot`` optional dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from tso_sensorium.episodes.schema import DatasetSchema, Episode

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
CAMERA_FEATURE_PREFIX = "observation.images."
TASK_KEY = "task"
IMAGE_DIMENSION_NAMES = ["height", "width", "channels"]
FLOAT_DTYPE = "float32"
VIDEO_DTYPE = "video"
IMAGE_DTYPE = "image"


class LeRobotDatasetWriter:
    """Writes episodes into a LeRobot dataset.

    Camera frames are loaded from the paths referenced in the episode
    table and stored through LeRobot's image or video pipeline; arm state
    and action columns are concatenated into flat float vectors.

    Args:
        repo_id: Dataset repository identifier, e.g. "user/dataset".
        output_root: Local directory for dataset storage.
        use_videos: Whether to encode camera streams as videos instead of
            individual images.
        task_column: Episode table column holding per-frame task strings;
            rows with empty values fall back to the episode or schema
            task.
    """

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

    def _build_features(self, schema: DatasetSchema) -> dict:
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
        camera_dtype = VIDEO_DTYPE if self.use_videos else IMAGE_DTYPE
        for camera in schema.cameras:
            features[f"{CAMERA_FEATURE_PREFIX}{camera.name}"] = {
                "dtype": camera_dtype,
                "shape": (camera.height, camera.width, 3),
                "names": IMAGE_DIMENSION_NAMES,
            }
        return features

    def open(self, schema: DatasetSchema) -> None:
        """Create the LeRobot dataset for the given schema."""
        if not schema.arms:
            raise ValueError("LeRobot export requires at least one arm in the schema")
        self._schema = schema
        self._dataset = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=schema.fps,
            features=self._build_features(schema=schema),
            root=self.output_root,
            use_videos=self.use_videos,
        )

    def _resolve_task(self, episode: Episode) -> str:
        task = episode.task if episode.task is not None else self._schema.task
        if task is None:
            raise ValueError(
                f"Episode {episode.name} has no task and the schema does not define one"
            )
        return task

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
        task_values = None
        if self.task_column is not None and self.task_column in episode.table:
            task_values = (
                episode.table[self.task_column].fillna("").astype(str).tolist()
            )
        default_task = None
        if task_values is None or any(value == "" for value in task_values):
            default_task = self._resolve_task(episode=episode)
        state_values = episode.table[schema.state_columns].to_numpy(dtype=np.float32)
        action_values = episode.table[schema.action_columns].to_numpy(dtype=np.float32)
        for row_index in range(len(episode.table)):
            task = default_task
            if task_values is not None and task_values[row_index] != "":
                task = task_values[row_index]
            frame = {
                TASK_KEY: task,
                STATE_FEATURE: state_values[row_index],
                ACTION_FEATURE: action_values[row_index],
            }
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
        (self.output_root / "dataset_metadata.json").write_text(
            json.dumps(metadata, indent=2)
        )

    def finalize(self) -> None:
        """Finalize the LeRobot dataset metadata."""
        if self._dataset is not None:
            self._dataset.finalize()
