"""Tests for tso_sensorium.export.lerobot_action_update_writer module."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lerobot")

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from lerobot.datasets.compute_stats import (  # noqa: E402
    aggregate_stats,
    compute_episode_stats,
)

from tso_sensorium.episodes.dataset_builder import (  # noqa: E402
    BuildCancellationToken,
    BuildPhase,
    BuildProgress,
    DatasetBuilder,
)
from tso_sensorium.episodes.legend import DatasetMetadata  # noqa: E402
from tso_sensorium.episodes.schema import (  # noqa: E402
    AuxiliaryFeature,
    DatasetSchema,
    Episode,
)
from tso_sensorium.export.lerobot_action_update_writer import (  # noqa: E402
    ACTION_STATS_PREFIX,
    LeRobotActionUpdateWriter,
    LeRobotDatasetCompatibilityError,
)
from tso_sensorium.export.lerobot_writer import (  # noqa: E402
    ACTION_FEATURE,
    LeRobotDatasetWriter,
)

EXCHANGE_PATH = (
    "tso_sensorium.export.lerobot_action_update_writer._atomic_exchange_directories"
)
REMOVE_TREE_PATH = "tso_sensorium.export.lerobot_action_update_writer.shutil.rmtree"


@dataclass(frozen=True)
class ActionUpdateDataset:
    root: Path
    schema: DatasetSchema
    original_episodes: list[Episode]
    updated_episodes: list[Episode]
    updated_metadata: dict


def _episode(
    name: str,
    state_offset: float,
    phase: int,
    task: str,
    actions: list[tuple[float, float]],
) -> Episode:
    length = len(actions)
    return Episode(
        name=name,
        table=pd.DataFrame(
            {
                "x": [state_offset + index for index in range(length)],
                "y": [state_offset - 0.5 * index for index in range(length)],
                "dx": [action[0] for action in actions],
                "dy": [action[1] for action in actions],
                "phase": [phase] * length,
                "language": [task] * length,
            }
        ),
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path=path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _non_action_table(table: pa.Table) -> pa.Table:
    return table.select(
        [
            name
            for name in table.column_names
            if name != ACTION_FEATURE and not name.startswith(ACTION_STATS_PREFIX)
        ]
    ).combine_chunks()


@pytest.fixture
def action_update_dataset_factory(tmp_path, schema_factory):
    def factory() -> ActionUpdateDataset:
        root = tmp_path / "dataset"
        schema = schema_factory(
            cameras=[],
            fps=5,
            auxiliary_features={
                "phase": AuxiliaryFeature(columns=["phase"], dtype="int64")
            },
        )
        original_episodes = [
            _episode(
                name="episode_a",
                state_offset=0.0,
                phase=4,
                task="task beta",
                actions=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
            ),
            _episode(
                name="episode_b",
                state_offset=10.0,
                phase=2,
                task="task alpha",
                actions=[(7.0, 8.0), (9.0, 10.0)],
            ),
        ]
        writer = LeRobotDatasetWriter(
            repo_id="local/action-update-test",
            output_root=root,
            use_videos=True,
            task_column="language",
        )
        writer.open(schema=schema)
        for episode in original_episodes:
            writer.add_episode(episode=episode)
        writer.write_metadata(
            metadata={
                "dataset_name": "dataset",
                "task": "retract the bowel",
                "phase_legend": {},
                "coordinate_frame_features": {},
                "generation": {
                    "written": [episode.name for episode in original_episodes],
                    "failed": {},
                    "generated_at": "before",
                },
            }
        )
        writer.finalize()
        tasks = pd.read_parquet(root / "meta/tasks.parquet")
        tasks.iloc[::-1].to_parquet(root / "meta/tasks.parquet")
        sentinel_path = root / "videos/sentinel.bin"
        sentinel_path.parent.mkdir()
        sentinel_path.write_bytes(b"video asset must remain untouched")
        updated_episodes = [
            _episode(
                name="episode_a",
                state_offset=0.0,
                phase=4,
                task="task beta",
                actions=[(0.0, 0.0), (3.0, 4.0), (0.0, 0.0)],
            ),
            _episode(
                name="episode_b",
                state_offset=10.0,
                phase=2,
                task="task alpha",
                actions=[(0.0, 0.0), (9.0, 10.0)],
            ),
        ]
        updated_metadata = {
            "dataset_name": "dataset",
            "task": "retract the bowel",
            "phase_legend": {},
            "coordinate_frame_features": {},
            "dataset_transforms": [
                {
                    "type": "percentile_denoise_columns",
                    "column_groups": {
                        "translation": {
                            "columns": ["dx", "dy"],
                            "percentile": 30.0,
                            "threshold": 1.5,
                        }
                    },
                }
            ],
            "generation": {
                "written": [episode.name for episode in updated_episodes],
                "failed": {},
                "generated_at": "after",
            },
        }
        return ActionUpdateDataset(
            root=root,
            schema=schema,
            original_episodes=original_episodes,
            updated_episodes=updated_episodes,
            updated_metadata=updated_metadata,
        )

    return factory


def _run_update(dataset: ActionUpdateDataset) -> None:
    writer = LeRobotActionUpdateWriter(
        dataset_root=dataset.root,
        task_column="language",
    )
    writer.open(schema=dataset.schema)
    for episode in dataset.updated_episodes:
        writer.add_episode(episode=episode)
    writer.write_metadata(metadata=dataset.updated_metadata)
    writer.finalize()


@pytest.mark.integration
def test_updates_only_actions_stats_and_provenance(action_update_dataset_factory):
    dataset = action_update_dataset_factory()
    data_path = next((dataset.root / "data").rglob("*.parquet"))
    episodes_path = next((dataset.root / "meta/episodes").rglob("*.parquet"))
    original_data = pq.read_table(data_path)
    original_episode_metadata = pq.read_table(episodes_path)
    original_stats = json.loads((dataset.root / "meta/stats.json").read_text())
    original_info_hash = _file_hash(path=dataset.root / "meta/info.json")
    original_tasks_hash = _file_hash(path=dataset.root / "meta/tasks.parquet")
    sentinel_path = dataset.root / "videos/sentinel.bin"
    original_sentinel_hash = _file_hash(path=sentinel_path)
    original_sentinel_inode = sentinel_path.stat().st_ino

    _run_update(dataset=dataset)

    updated_data = pq.read_table(data_path)
    expected_actions = np.concatenate(
        [
            episode.table[["dx", "dy"]].to_numpy(dtype=np.float32)
            for episode in dataset.updated_episodes
        ]
    )
    np.testing.assert_array_equal(
        np.asarray(updated_data[ACTION_FEATURE].combine_chunks().to_pylist()),
        expected_actions,
    )
    assert _non_action_table(updated_data).equals(_non_action_table(original_data))
    updated_episode_metadata = pq.read_table(episodes_path)
    assert _non_action_table(updated_episode_metadata).equals(
        _non_action_table(original_episode_metadata)
    )
    action_feature = json.loads((dataset.root / "meta/info.json").read_text())[
        "features"
    ][ACTION_FEATURE]
    episode_stats = [
        compute_episode_stats(
            episode_data={
                ACTION_FEATURE: episode.table[["dx", "dy"]].to_numpy(dtype=np.float32)
            },
            features={ACTION_FEATURE: action_feature},
        )[ACTION_FEATURE]
        for episode in dataset.updated_episodes
    ]
    for episode_index, expected_stats in enumerate(episode_stats):
        metadata_frame = pd.read_parquet(episodes_path)
        row = metadata_frame.loc[metadata_frame["episode_index"] == episode_index].iloc[
            0
        ]
        for stat_name, expected_value in expected_stats.items():
            np.testing.assert_allclose(
                row[f"{ACTION_STATS_PREFIX}{stat_name}"],
                expected_value,
                rtol=0.0,
                atol=1e-15,
            )
    expected_global_stats = aggregate_stats(
        [{ACTION_FEATURE: stats} for stats in episode_stats]
    )[ACTION_FEATURE]
    updated_stats = json.loads((dataset.root / "meta/stats.json").read_text())
    for name, expected_value in expected_global_stats.items():
        np.testing.assert_allclose(
            updated_stats[ACTION_FEATURE][name],
            expected_value,
            rtol=0.0,
            atol=1e-15,
        )
    assert {
        key: value for key, value in updated_stats.items() if key != ACTION_FEATURE
    } == {key: value for key, value in original_stats.items() if key != ACTION_FEATURE}
    assert (
        json.loads((dataset.root / "dataset_metadata.json").read_text())
        == dataset.updated_metadata
    )
    assert _file_hash(path=dataset.root / "meta/info.json") == original_info_hash
    assert _file_hash(path=dataset.root / "meta/tasks.parquet") == original_tasks_hash
    assert _file_hash(path=sentinel_path) == original_sentinel_hash
    assert sentinel_path.stat().st_ino == original_sentinel_inode
    assert not list(dataset.root.parent.glob(f".{dataset.root.name}.action-update-*"))


@pytest.mark.integration
@pytest.mark.parametrize(
    "mismatch",
    ["name", "length", "state", "auxiliary", "task"],
)
def test_rejects_non_action_episode_mismatch_without_writes(
    action_update_dataset_factory,
    mismatch: str,
):
    dataset = action_update_dataset_factory()
    hashes_before = _tree_hashes(root=dataset.root)
    episode = dataset.updated_episodes[0]
    table = episode.table.copy()
    name = episode.name
    if mismatch == "name":
        name = "wrong_episode"
    elif mismatch == "length":
        table = table.iloc[:-1].reset_index(drop=True)
    elif mismatch == "state":
        table.loc[0, "x"] += 1.0
    elif mismatch == "auxiliary":
        table.loc[0, "phase"] += 1
    elif mismatch == "task":
        table.loc[0, "language"] = "different task"
    mismatched_episode = Episode(name=name, table=table)
    writer = LeRobotActionUpdateWriter(
        dataset_root=dataset.root,
        task_column="language",
    )
    writer.open(schema=dataset.schema)

    with pytest.raises(LeRobotDatasetCompatibilityError):
        writer.add_episode(episode=mismatched_episode)
    writer.abort()

    assert _tree_hashes(root=dataset.root) == hashes_before


@pytest.mark.integration
def test_rejects_schema_mismatch_without_writes(action_update_dataset_factory):
    dataset = action_update_dataset_factory()
    hashes_before = _tree_hashes(root=dataset.root)
    mismatched_schema = dataset.schema.model_copy(update={"fps": 10})
    writer = LeRobotActionUpdateWriter(dataset_root=dataset.root)

    with pytest.raises(
        LeRobotDatasetCompatibilityError,
        match="LeRobot fps 5 does not match schema fps 10",
    ):
        writer.open(schema=mismatched_schema)

    assert _tree_hashes(root=dataset.root) == hashes_before


@pytest.mark.integration
def test_precommit_failure_leaves_target_byte_identical(
    action_update_dataset_factory,
):
    dataset = action_update_dataset_factory()
    hashes_before = _tree_hashes(root=dataset.root)
    writer = LeRobotActionUpdateWriter(
        dataset_root=dataset.root,
        task_column="language",
    )
    writer.open(schema=dataset.schema)
    for episode in dataset.updated_episodes:
        writer.add_episode(episode=episode)
    writer.write_metadata(metadata=dataset.updated_metadata)

    with patch(EXCHANGE_PATH, side_effect=RuntimeError("exchange unavailable")):
        with pytest.raises(RuntimeError, match="exchange unavailable"):
            writer.finalize()
    writer.abort()

    assert _tree_hashes(root=dataset.root) == hashes_before
    assert not list(dataset.root.parent.glob(f".{dataset.root.name}.action-update-*"))


@pytest.mark.integration
def test_postcommit_cleanup_failure_keeps_successful_update_and_releases_lock(
    action_update_dataset_factory,
    caplog,
):
    dataset = action_update_dataset_factory()
    writer = LeRobotActionUpdateWriter(
        dataset_root=dataset.root,
        task_column="language",
    )
    writer.open(schema=dataset.schema)
    for episode in dataset.updated_episodes:
        writer.add_episode(episode=episode)
    writer.write_metadata(metadata=dataset.updated_metadata)

    cleanup_error = OSError("staging cleanup failed")
    with patch.object(
        writer,
        "_release_lock",
        wraps=writer._release_lock,
    ) as release_lock:
        with patch(REMOVE_TREE_PATH, side_effect=cleanup_error):
            writer.finalize()

    updated_data = pq.read_table(next((dataset.root / "data").rglob("*.parquet")))
    expected_actions = np.concatenate(
        [
            episode.table[["dx", "dy"]].to_numpy(dtype=np.float32)
            for episode in dataset.updated_episodes
        ]
    )
    np.testing.assert_array_equal(
        np.asarray(updated_data[ACTION_FEATURE].combine_chunks().to_pylist()),
        expected_actions,
    )
    release_lock.assert_called_once_with()
    assert "staging cleanup failed" in caplog.text

    writer.abort()
    assert not list(dataset.root.parent.glob(f".{dataset.root.name}.action-update-*"))


@pytest.mark.integration
def test_builder_cancellation_aborts_update_without_writes(
    action_update_dataset_factory,
    tmp_path,
):
    dataset = action_update_dataset_factory()
    hashes_before = _tree_hashes(root=dataset.root)
    recordings_root = tmp_path / "recordings"
    episodes_by_name = {episode.name: episode for episode in dataset.updated_episodes}
    for episode_name in episodes_by_name:
        (recordings_root / episode_name).mkdir(parents=True)
    token = BuildCancellationToken()
    progress_events = []

    def capture_progress(progress: BuildProgress) -> None:
        progress_events.append(progress)
        if progress.phase == BuildPhase.UPDATING and progress.completed == 1:
            token.cancel()

    writer = LeRobotActionUpdateWriter(
        dataset_root=dataset.root,
        task_column="language",
    )
    builder = DatasetBuilder(
        schema=dataset.schema,
        writer=writer,
        assemble_episode=lambda directory: episodes_by_name[directory.name],
        n_jobs=1,
        dataset_metadata=DatasetMetadata(dataset_name="dataset"),
        progress_callback=capture_progress,
        cancellation_token=token,
    )

    report = builder.build(recordings_root=recordings_root)

    assert report.cancelled is True
    assert progress_events[-1].phase == BuildPhase.CANCELLED
    assert _tree_hashes(root=dataset.root) == hashes_before
