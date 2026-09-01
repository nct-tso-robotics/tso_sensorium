"""Tests for tso_sensorium.export.lerobot_writer module."""

import json
import re
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lerobot")

from tso_sensorium.episodes.schema import AuxiliaryFeature, Episode  # noqa: E402
from tso_sensorium.export.lerobot_writer import (  # noqa: E402
    LeRobotDatasetWriter,
)

LEROBOT_DATASET_PATH = "tso_sensorium.export.lerobot_writer.LeRobotDataset"
IMREAD_PATH = "tso_sensorium.export.lerobot_writer.cv2.imread"


@pytest.fixture
def writer_factory(tmp_path):
    def factory(use_videos=True) -> LeRobotDatasetWriter:
        return LeRobotDatasetWriter(
            repo_id="tso/test_dataset",
            output_root=tmp_path / "dataset",
            use_videos=use_videos,
        )

    return factory


class TestOpen:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "use_videos, expected_camera_dtype",
        [
            (True, "video"),
            (False, "image"),
        ],
    )
    def test_creates_dataset_with_schema_features(
        self,
        writer_factory,
        schema_factory,
        tmp_path,
        use_videos,
        expected_camera_dtype,
    ):
        writer = writer_factory(use_videos=use_videos)
        schema = schema_factory(
            fps=30,
            auxiliary_features={
                "task_phase": AuxiliaryFeature(columns=["task_phase"], dtype="int64")
            },
        )
        with patch(LEROBOT_DATASET_PATH) as dataset_class:
            writer.open(schema=schema)
        dataset_class.create.assert_called_once_with(
            repo_id="tso/test_dataset",
            fps=30,
            features={
                "observation.state": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": ["x", "y"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (2,),
                    "names": ["dx", "dy"],
                },
                "task_phase": {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": ["task_phase"],
                },
                "observation.images.left": {
                    "dtype": expected_camera_dtype,
                    "shape": (8, 8, 3),
                    "names": ["height", "width", "channels"],
                },
            },
            root=tmp_path / "dataset",
            use_videos=use_videos,
        )

    @pytest.mark.unit
    def test_rejects_schema_without_arms(self, writer_factory, schema_factory):
        writer = writer_factory()
        with pytest.raises(
            ValueError,
            match=re.escape("LeRobot export requires at least one arm in the schema"),
        ):
            writer.open(schema=schema_factory(arms=[]))


class TestAddEpisode:
    @pytest.mark.unit
    def test_before_open_raises(self, writer_factory, episode_table_factory):
        writer = writer_factory()
        with pytest.raises(
            RuntimeError,
            match=re.escape(
                "Writer must be opened with a schema before adding episodes"
            ),
        ):
            writer.add_episode(
                episode=Episode(name="episode_000", table=episode_table_factory())
            )

    @pytest.mark.unit
    def test_streams_frames_and_saves_episode(
        self, writer_factory, schema_factory, episode_table_factory
    ):
        writer = writer_factory()
        bgr_image = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255
        table = episode_table_factory(length=2)
        table["task_phase"] = [2, 3]
        schema = schema_factory(
            auxiliary_features={
                "task_phase": AuxiliaryFeature(columns=["task_phase"], dtype="int64")
            }
        )
        with patch(LEROBOT_DATASET_PATH) as dataset_class:
            writer.open(schema=schema)
            dataset = dataset_class.create.return_value
            with patch(IMREAD_PATH, return_value=bgr_image):
                writer.add_episode(episode=Episode(name="episode_000", table=table))
        assert dataset.add_frame.call_count == 2
        first_frame = dataset.add_frame.call_args_list[0].args[0]
        assert first_frame["task"] == "retract the bowel"
        np.testing.assert_array_equal(
            first_frame["observation.state"],
            np.array([0.0, 0.5 * 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            first_frame["action"], np.array([0.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            first_frame["task_phase"], np.array([2], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            first_frame["observation.images.left"],
            cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB),
        )
        dataset.save_episode.assert_called_once_with()

    @pytest.mark.unit
    def test_episode_task_overrides_schema_task(
        self, writer_factory, schema_factory, episode_table_factory
    ):
        writer = writer_factory()
        with patch(LEROBOT_DATASET_PATH) as dataset_class:
            writer.open(schema=schema_factory(task="schema task"))
            dataset = dataset_class.create.return_value
            with patch(IMREAD_PATH, return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
                writer.add_episode(
                    episode=Episode(
                        name="episode_000",
                        table=episode_table_factory(length=1),
                        task="episode task",
                    )
                )
        frame = dataset.add_frame.call_args.args[0]
        assert frame["task"] == "episode task"

    @pytest.mark.unit
    def test_raises_without_any_task(
        self, writer_factory, schema_factory, episode_table_factory
    ):
        writer = writer_factory()
        with patch(LEROBOT_DATASET_PATH):
            writer.open(schema=schema_factory(task=None))
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "Episode episode_000 has no task and the schema does not define one"
                ),
            ):
                writer.add_episode(
                    episode=Episode(
                        name="episode_000", table=episode_table_factory(length=1)
                    )
                )

    @pytest.mark.unit
    def test_raises_for_unreadable_frame(
        self, writer_factory, schema_factory, episode_table_factory
    ):
        writer = writer_factory()
        with patch(LEROBOT_DATASET_PATH):
            writer.open(schema=schema_factory())
            with patch(IMREAD_PATH, return_value=None):
                with pytest.raises(
                    FileNotFoundError,
                    match=re.escape("Could not read frame frames/0.png"),
                ):
                    writer.add_episode(
                        episode=Episode(
                            name="episode_000",
                            table=episode_table_factory(length=1),
                        )
                    )


class TestFinalize:
    @pytest.mark.unit
    def test_finalizes_dataset(self, writer_factory, schema_factory):
        writer = writer_factory()
        with patch(LEROBOT_DATASET_PATH) as dataset_class:
            writer.open(schema=schema_factory())
            writer.finalize()
        dataset_class.create.return_value.finalize.assert_called_once_with()

    @pytest.mark.unit
    def test_before_open_is_noop(self, writer_factory):
        writer = writer_factory()
        writer.finalize()


class TestAbort:
    @pytest.mark.unit
    def test_removes_new_writer_owned_output(
        self, writer_factory, schema_factory, tmp_path
    ):
        writer = writer_factory()
        with patch(LEROBOT_DATASET_PATH):
            writer.open(schema=schema_factory())
            output_root = tmp_path / "dataset"
            output_root.mkdir()
            (output_root / "partial").write_text("incomplete")

            writer.abort()

        assert not output_root.exists()

    @pytest.mark.unit
    def test_never_removes_preexisting_output(
        self, writer_factory, schema_factory, tmp_path
    ):
        output_root = tmp_path / "dataset"
        output_root.mkdir()
        sentinel = output_root / "sentinel"
        sentinel.write_text("preserve")
        writer = writer_factory()
        with patch(LEROBOT_DATASET_PATH):
            writer.open(schema=schema_factory())
            writer.abort()

        assert sentinel.read_text() == "preserve"


@pytest.mark.integration
def test_round_trip_writes_v30_dataset(
    tmp_path, schema_factory, episode_table_factory, rng
):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    frame_paths = []
    for index in range(3):
        frame_path = frames_dir / f"{index}.png"
        cv2.imwrite(
            str(frame_path),
            rng.integers(0, 255, size=(8, 8, 3), dtype=np.uint8),
        )
        frame_paths.append(str(frame_path))

    writer = LeRobotDatasetWriter(
        repo_id="tso/test_dataset",
        output_root=tmp_path / "dataset",
        use_videos=False,
    )
    schema = schema_factory(
        fps=10,
        auxiliary_features={
            "task_phase": AuxiliaryFeature(columns=["task_phase"], dtype="int64")
        },
    )
    writer.open(schema=schema)
    table = episode_table_factory(length=3, frame_paths=frame_paths)
    table["task_phase"] = [0, 1, 1]
    writer.add_episode(
        episode=Episode(
            name="episode_000",
            table=table,
        )
    )
    writer.finalize()

    info = json.loads((tmp_path / "dataset" / "meta" / "info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_frames"] == 3
    assert info["total_episodes"] == 1
    assert "observation.images.left" in info["features"]
    assert info["features"]["task_phase"] == {
        "dtype": "int64",
        "shape": [1],
        "names": ["task_phase"],
    }
    data_files = list((tmp_path / "dataset" / "data").rglob("*.parquet"))
    assert len(data_files) == 1
    assert pd.read_parquet(data_files[0])["task_phase"].tolist() == [0, 1, 1]
