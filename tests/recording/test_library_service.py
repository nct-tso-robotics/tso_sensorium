"""Tests for tso_sensorium.recording.library_service module."""

import time
from threading import Event
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

flask = pytest.importorskip("flask")

import pandas as pd  # noqa: E402

from tso_sensorium.configuration import load_config  # noqa: E402
from tso_sensorium.episodes.dataset_builder import (  # noqa: E402
    BuildCancellationToken,
    BuildPhase,
    BuildProgress,
    BuildReport,
)
from tso_sensorium.episodes.dataset_transforms import (  # noqa: E402
    PercentileDenoiseColumns,
    PercentileDenoisingColumnGroup,
)
from tso_sensorium.episodes.generation_config import (  # noqa: E402
    AnnotationsConfig,
    CsvWriterConfig,
    DatasetGenerationConfig,
    LeRobotActionUpdateWriterConfig,
    LeRobotWriterConfig,
    StateSourceConfig,
)
from tso_sensorium.recording.denoising_preview import (  # noqa: E402
    DenoisingPreviewData,
)
from tso_sensorium.recording.library_service import (  # noqa: E402
    GenerationMode,
    LibraryService,
    create_library_app,
)
from tso_sensorium.recording.config import LibraryAppConfig  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENDOSCOPE_GUIDANCE_ANNOTATION_CONFIG = (
    REPOSITORY_ROOT / "configs" / "annotation" / "endoscope_guidance.yaml"
)


@pytest.fixture
def library_client_factory(tmp_path):
    def factory(with_generation=False):
        generation = None
        if with_generation:
            generation = DatasetGenerationConfig(
                states=[StateSourceConfig(state_file="state.csv", columns=["x"])],
                writer=CsvWriterConfig(),
                n_jobs=1,
            )
        root = tmp_path / "episodes"
        root.mkdir(exist_ok=True)
        library = LibraryService(recordings_root=root, generation=generation)
        return create_library_app(library=library).test_client(), root

    return factory


@pytest.mark.unit
def test_status_reports_library_without_recording(library_client_factory):
    client, root = library_client_factory()
    status = client.get("/api/status").get_json()
    assert status["recording_available"] is False
    assert status["state"] == "idle"
    assert status["sensors"] == []
    assert status["output_folder"] == str(root)


@pytest.mark.unit
def test_metadata_round_trip(library_client_factory):
    client, root = library_client_factory()
    saved = client.put(
        "/api/library/metadata",
        json={
            "dataset_name": "bowel_retraction",
            "task": "retract the bowel",
            "phase_legend": {
                "0": {"name": "wait", "instructions": ["hold still", ""]},
                "1": {"name": "retract", "instructions": ["pull back"]},
            },
        },
    )
    assert saved.status_code == 200
    assert (root / "dataset_metadata.json").is_file()
    fetched = client.get("/api/library/metadata").get_json()
    assert fetched["dataset_name"] == "bowel_retraction"
    assert fetched["phase_legend"]["0"]["name"] == "wait"
    assert fetched["phase_legend"]["0"]["instructions"] == ["hold still"]


@pytest.mark.unit
def test_annotations_round_trip_without_ros(library_client_factory):
    client, root = library_client_factory()
    (root / "ep0").mkdir()
    saved = client.put(
        "/api/episodes/ep0/annotations",
        json={"segments": [{"start": 0, "end": 10, "phase": 0}]},
    )
    assert saved.status_code == 200
    fetched = client.get("/api/episodes/ep0/annotations").get_json()
    assert fetched["segments"][0]["phase"] == 0
    assert fetched["segments"][0]["source"] == "manual"


@pytest.mark.unit
def test_root_switching(library_client_factory, tmp_path):
    client, root = library_client_factory()
    other = tmp_path / "other"
    other.mkdir()
    switched = client.post("/api/library/root", json={"path": str(other)})
    assert switched.status_code == 200
    assert client.get("/api/status").get_json()["output_folder"] == str(other)
    missing = client.post("/api/library/root", json={"path": str(other / "nope")})
    assert missing.status_code == 400


@pytest.mark.integration
def test_generation_without_ros(library_client_factory):
    client, root = library_client_factory(with_generation=True)
    episode = root / "ep0"
    episode.mkdir()
    pd.DataFrame({"time": [0, 1], "x": [0.5, 0.6]}).to_csv(
        episode / "state.csv", index=False
    )
    started = client.post("/api/generation/start", json={})
    assert started.status_code == 200
    for _ in range(50):
        generation = client.get("/api/status").get_json()["generation"]
        if generation["state"] != "running":
            break
        time.sleep(0.2)
    assert generation["state"] == "done"
    assert generation["written"] == ["ep0"]
    assert (episode / "episode.csv").is_file()


@pytest.mark.unit
def test_actions_only_start_builds_typed_writer_and_keeps_denoising_override(
    tmp_path,
):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    dataset_root = tmp_path / "existing_lerobot"
    dataset_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        dataset_transforms=[
            PercentileDenoiseColumns(
                column_groups={
                    "movement": PercentileDenoisingColumnGroup(
                        columns=["delta"],
                        percentile=20.0,
                    )
                }
            )
        ],
        annotations=AnnotationsConfig(language_column="phase_language"),
        writer=CsvWriterConfig(output_root=None),
        save_frames=True,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()
    override_path = "dataset_transforms.0.column_groups.movement.percentile"

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post(
            "/api/generation/start",
            json={
                "mode": GenerationMode.ACTIONS_ONLY.value,
                "dataset_root": str(dataset_root),
                "overrides": {override_path: 37.5, "save_frames": True},
            },
        )

    assert response.status_code == 200
    assert response.get_json()["mode"] == GenerationMode.ACTIONS_ONLY.value
    thread_factory.return_value.start.assert_called_once_with()
    generation_config = thread_factory.call_args.kwargs["kwargs"]["generation_config"]
    assert isinstance(generation_config.writer, LeRobotActionUpdateWriterConfig)
    assert generation_config.writer.dataset_root == str(dataset_root.resolve())
    assert generation_config.writer.task_column == "phase_language"
    assert generation_config.save_frames is False
    denoising_group = generation_config.dataset_transforms[0].column_groups["movement"]
    assert denoising_group.percentile == 37.5


@pytest.mark.unit
@pytest.mark.parametrize("dataset_root", [None, "missing_lerobot"])
def test_actions_only_start_rejects_missing_dataset_root(tmp_path, dataset_root):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()
    payload = {"mode": GenerationMode.ACTIONS_ONLY.value}
    if dataset_root is not None:
        payload["dataset_root"] = str(tmp_path / dataset_root)

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post("/api/generation/start", json=payload)

    assert response.status_code == 400
    assert "error" in response.get_json()
    thread_factory.assert_not_called()


@pytest.mark.unit
def test_actions_only_start_preserves_symlink_root_rejection(tmp_path):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    real_dataset_root = tmp_path / "real_lerobot"
    real_dataset_root.mkdir()
    linked_dataset_root = tmp_path / "linked_lerobot"
    linked_dataset_root.symlink_to(real_dataset_root, target_is_directory=True)
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post(
            "/api/generation/start",
            json={
                "mode": GenerationMode.ACTIONS_ONLY.value,
                "dataset_root": str(linked_dataset_root),
            },
        )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": (
            "Actions-only update requires an existing LeRobot dataset root:"
            f" {linked_dataset_root}"
        )
    }
    thread_factory.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("relationship", ["same", "child", "ancestor"])
def test_actions_only_start_rejects_recordings_root_overlap(tmp_path, relationship):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    if relationship == "same":
        dataset_root = recordings_root
    elif relationship == "child":
        dataset_root = recordings_root / "lerobot"
        dataset_root.mkdir()
    else:
        dataset_root = tmp_path
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post(
            "/api/generation/start",
            json={
                "mode": GenerationMode.ACTIONS_ONLY.value,
                "dataset_root": str(dataset_root),
            },
        )

    assert response.status_code == 400
    assert "must not overlap" in response.get_json()["error"]
    thread_factory.assert_not_called()


@pytest.mark.unit
def test_full_mode_rejects_action_update_writer_override(tmp_path):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    dataset_root = tmp_path / "existing_lerobot"
    dataset_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post(
            "/api/generation/start",
            json={
                "mode": GenerationMode.FULL.value,
                "overrides": {
                    "writer": {
                        "type": "lerobot_action_update",
                        "dataset_root": str(dataset_root),
                    }
                },
            },
        )

    assert response.status_code == 400
    assert response.get_json() == {
        "error": ("LeRobot action updates require mode='actions_only' and dataset_root")
    }
    thread_factory.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("output_kind", ["existing", "recordings_child"])
def test_full_lerobot_start_requires_fresh_nonoverlapping_output(
    tmp_path,
    output_kind,
):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    if output_kind == "existing":
        output_root = tmp_path / "existing_output"
        output_root.mkdir()
    else:
        output_root = recordings_root / "new_output"
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post(
            "/api/generation/start",
            json={
                "mode": GenerationMode.FULL.value,
                "overrides": {
                    "writer": {
                        "type": "lerobot",
                        "output_root": str(output_root),
                        "repo_id": None,
                        "use_videos": True,
                    }
                },
            },
        )

    assert response.status_code == 400
    assert "error" in response.get_json()
    thread_factory.assert_not_called()


@pytest.mark.integration
def test_full_lerobot_start_uses_all_endoscope_phase_language_tasks(tmp_path):
    app_config = load_config(
        config_class=LibraryAppConfig,
        config_path=ENDOSCOPE_GUIDANCE_ANNOTATION_CONFIG,
    )
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    library = LibraryService(
        recordings_root=recordings_root,
        generation=app_config.generation,
    )
    client = create_library_app(library=library).test_client()
    output_root = tmp_path / "new_lerobot"

    with patch(
        "tso_sensorium.recording.library_service.threading.Thread"
    ) as thread_factory:
        response = client.post(
            "/api/generation/start",
            json={
                "mode": GenerationMode.FULL.value,
                "overrides": {
                    "writer": {
                        "type": "lerobot",
                        "output_root": str(output_root),
                        "repo_id": None,
                        "use_videos": True,
                    }
                },
            },
        )

    assert response.status_code == 200
    generation_config = thread_factory.call_args.kwargs["kwargs"]["generation_config"]
    assert isinstance(generation_config.writer, LeRobotWriterConfig)
    assert generation_config.writer.task_column == (
        generation_config.annotations.language_column
    )
    assert generation_config.writer.task_column == "language"
    assert generation_config.writer.use_videos is True
    phase_legend = generation_config.annotations.legend.phase_legend
    assert len(phase_legend) == 6
    assert all(
        len(definition.instructions) == 1 for definition in phase_legend.values()
    )


@pytest.mark.unit
def test_generation_progress_snapshots_are_exposed_through_status(tmp_path):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root=str(recordings_root),
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    cancellation = BuildCancellationToken()
    observed_status = {}

    def run_generation(config, progress_callback, cancellation_token):
        progress_callback(
            BuildProgress(
                phase=BuildPhase.ASSEMBLING,
                completed=2,
                total=4,
                current_episode="episode_2",
                failed={"episode_1": "missing state"},
            )
        )
        observed_status.update(library.generation_status())
        progress_callback(
            BuildProgress(
                phase=BuildPhase.COMPLETED,
                completed=1,
                total=1,
                failed={"episode_1": "missing state"},
            )
        )
        return BuildReport(
            written=["episode_2"],
            failed={"episode_1": "missing state"},
            cancelled=False,
        )

    with patch(
        "tso_sensorium.recording.library_service.generate_dataset",
        side_effect=run_generation,
    ) as generate:
        library._run_generation(
            generation_config=generation,
            cancellation=cancellation,
        )

    generate.assert_called_once_with(
        config=generation,
        progress_callback=library._record_generation_progress,
        cancellation_token=cancellation,
    )
    assert observed_status["phase"] == BuildPhase.ASSEMBLING.value
    assert observed_status["completed"] == 2
    assert observed_status["total"] == 4
    assert observed_status["percentage"] == 50.0
    assert observed_status["current_episode"] == "episode_2"
    assert observed_status["failed"] == {"episode_1": "missing state"}
    final_status = library.generation_status()
    assert final_status["state"] == "done"
    assert final_status["written"] == ["episode_2"]


@pytest.mark.integration
def test_cancel_endpoint_persists_cancelling_and_cancelled_states(tmp_path):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()
    generation_started = Event()
    release_generation = Event()

    def wait_for_cancellation(config, progress_callback, cancellation_token):
        generation_started.set()
        release_generation.wait(timeout=5)
        progress_callback(
            BuildProgress(
                phase=BuildPhase.CANCELLED,
                completed=1,
                total=3,
                current_episode="episode_1",
                failed={},
            )
        )
        return BuildReport(cancelled=cancellation_token.is_cancelled())

    with patch(
        "tso_sensorium.recording.library_service.generate_dataset",
        side_effect=wait_for_cancellation,
    ):
        started = client.post(
            "/api/generation/start",
            json={"mode": GenerationMode.FULL.value},
        )
        assert started.status_code == 200
        assert generation_started.wait(timeout=5)

        cancelling = client.post("/api/generation/cancel", json={})
        assert cancelling.status_code == 200
        assert cancelling.get_json()["state"] == "cancelling"
        assert client.get("/api/status").get_json()["generation"]["state"] == (
            "cancelling"
        )

        release_generation.set()
        for _ in range(50):
            final_status = client.get("/api/status").get_json()["generation"]
            if final_status["state"] == "cancelled":
                break
            time.sleep(0.02)

    assert final_status["state"] == "cancelled"
    assert final_status["phase"] == BuildPhase.CANCELLED.value
    assert final_status["completed"] == 1
    assert final_status["total"] == 3
    assert final_status["percentage"] == pytest.approx(100 / 3)
    assert final_status["written"] == []


@pytest.mark.unit
def test_cancel_endpoint_rejects_finalization_point_of_no_return(tmp_path):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    cancellation = BuildCancellationToken()
    library._generation_thread = MagicMock()
    library._generation_thread.is_alive.return_value = True
    library._generation_cancellation = cancellation
    library._generation_status = {
        "state": "running",
        "phase": BuildPhase.FINALIZING.value,
    }
    client = create_library_app(library=library).test_client()

    response = client.post("/api/generation/cancel", json={})

    assert response.status_code == 409
    assert response.get_json() == {
        "error": ("Dataset generation is finalizing and can no longer be cancelled")
    }
    assert cancellation.is_cancelled() is False


@pytest.mark.unit
def test_generation_start_rejects_active_denoising_preview(tmp_path):
    recordings_root = tmp_path / "recordings"
    recordings_root.mkdir()
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()
    library._denoising_preview_lock.acquire()

    try:
        response = client.post(
            "/api/generation/start",
            json={"mode": GenerationMode.FULL.value},
        )
    finally:
        library._denoising_preview_lock.release()

    assert response.status_code == 409
    assert response.get_json() == {
        "error": "Dataset generation cannot start during denoising preview"
    }


@pytest.mark.integration
def test_generation_blocks_annotation_and_metadata_mutations(tmp_path):
    recordings_root = tmp_path / "recordings"
    episode_root = recordings_root / "episode_0"
    episode_root.mkdir(parents=True)
    annotation_path = episode_root / "annotations.json"
    annotation_path.write_text('{"segments": []}')
    metadata_path = recordings_root / "dataset_metadata.json"
    metadata_path.write_text(
        '{"dataset_name": "original", "task": "task", "phase_legend": {}}'
    )
    generation = DatasetGenerationConfig(
        recordings_root="unused",
        writer=CsvWriterConfig(output_root=None),
        save_frames=False,
        n_jobs=1,
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    library._generation_thread = MagicMock()
    library._generation_thread.is_alive.return_value = True
    client = create_library_app(library=library).test_client()
    original_annotations = annotation_path.read_bytes()
    original_metadata = metadata_path.read_bytes()

    annotation_response = client.put(
        "/api/episodes/episode_0/annotations",
        json={"segments": [{"start": 0, "end": 10, "phase": 1}]},
    )
    metadata_response = client.put(
        "/api/library/metadata",
        json={
            "dataset_name": "changed",
            "task": "changed",
            "phase_legend": {},
        },
    )

    assert annotation_response.status_code == 409
    assert metadata_response.status_code == 409
    assert annotation_path.read_bytes() == original_annotations
    assert metadata_path.read_bytes() == original_metadata


@pytest.mark.unit
def test_directory_listing(library_client_factory, tmp_path):
    client, root = library_client_factory()
    (root / "episode_a").mkdir()
    (root / "episode_b").mkdir()
    (root / ".hidden").mkdir()

    listing = client.get(f"/api/library/directories?path={root}").get_json()
    assert listing["path"] == str(root)
    assert listing["parent"] == str(root.parent)
    assert listing["directories"] == ["episode_a", "episode_b"]

    missing = client.get(f"/api/library/directories?path={root / 'nope'}")
    assert missing.status_code == 400


@pytest.mark.unit
def test_directory_listing_reports_filesystem_error(library_client_factory):
    client, root = library_client_factory()
    error_message = f"Permission denied: {root}"

    with patch.object(Path, "iterdir", side_effect=PermissionError(error_message)):
        response = client.get(f"/api/library/directories?path={root}")

    assert response.status_code == 400
    assert response.get_json() == {"error": error_message}


@pytest.mark.unit
def test_denoising_preview_reuses_assembled_data_for_percentile_changes(tmp_path):
    recordings_root = tmp_path / "episodes"
    recordings_root.mkdir()
    generation = DatasetGenerationConfig(
        dataset_transforms=[
            PercentileDenoiseColumns(
                column_groups={
                    "movement": PercentileDenoisingColumnGroup(
                        columns=["delta"],
                        percentile=20.0,
                    )
                }
            )
        ]
    )
    library = LibraryService(
        recordings_root=recordings_root,
        generation=generation,
    )
    client = create_library_app(library=library).test_client()
    preview_data = MagicMock(spec=DenoisingPreviewData)
    preview_data.to_payload.return_value = {
        "available": True,
        "episode_count": 2,
        "failed": {},
        "groups": [],
    }
    override_path = "dataset_transforms.0.column_groups.movement.percentile"

    with patch(
        "tso_sensorium.recording.library_service.load_denoising_preview_data",
        return_value=preview_data,
    ) as load_preview:
        first = client.post(
            "/api/generation/denoising-preview",
            json={"refresh": True, "percentiles": {override_path: 25.0}},
        )
        second = client.post(
            "/api/generation/denoising-preview",
            json={"percentiles": {override_path: 35.0}},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    load_preview.assert_called_once_with(
        config=generation,
        recordings_root=recordings_root,
    )
    assert preview_data.to_payload.call_args_list == [
        call(percentiles={override_path: 25.0}, phase_names={}),
        call(percentiles={override_path: 35.0}, phase_names={}),
    ]


@pytest.mark.integration
def test_episode_deletion_removes_only_selected_episode(library_client_factory):
    client, root = library_client_factory()
    selected = root / "selected"
    selected.mkdir()
    (selected / "state.csv").write_text("time,state\n1,ready\n")
    retained = root / "retained"
    retained.mkdir()

    response = client.delete("/api/episodes/selected")

    assert response.status_code == 200
    assert response.get_json() == {"deleted": "selected"}
    assert not selected.exists()
    assert retained.is_dir()


@pytest.mark.integration
def test_episode_deletion_rejects_missing_episode(library_client_factory):
    client, _ = library_client_factory()

    response = client.delete("/api/episodes/missing")

    assert response.status_code == 400
    assert response.get_json() == {"error": "Episode not found: missing"}


@pytest.mark.unit
def test_episode_name_traversal_rejected(library_client_factory):
    client, root = library_client_factory()
    (root / "real_ep").mkdir()
    assert client.get("/api/episodes/real_ep/annotations").status_code == 200
    # Whether the router (404) or the guard (400) rejects it, no annotation
    # file is ever written outside the recordings root.
    for evil in ("..", "%2e%2e", "sub%2fnested"):
        put = client.put(f"/api/episodes/{evil}/annotations", json={"segments": []})
        assert put.status_code != 200
        assert not (root.parent / "annotations.json").exists()


@pytest.mark.unit
def test_episode_directory_guard_rejects_escape(tmp_path):
    from tso_sensorium.recording.library_service import LibraryService

    root = tmp_path / "recordings"
    (root / "ep0").mkdir(parents=True)
    library = LibraryService(recordings_root=root)
    assert library.episode_directory(episode_name="ep0") == (root / "ep0").resolve()
    with pytest.raises(ValueError, match="Invalid episode name"):
        library.episode_directory(episode_name="..")
    with pytest.raises(ValueError, match="Invalid episode name"):
        library.episode_directory(episode_name="../../etc")
