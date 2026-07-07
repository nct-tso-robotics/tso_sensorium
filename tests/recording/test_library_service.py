"""Tests for tso_sensorium.recording.library_service module."""

import time

import pytest

flask = pytest.importorskip("flask")

import pandas as pd  # noqa: E402

from tso_sensorium.episodes.generation_config import (  # noqa: E402
    CsvWriterConfig,
    DatasetGenerationConfig,
    StateSourceConfig,
)
from tso_sensorium.recording.library_service import (  # noqa: E402
    LibraryService,
    create_library_app,
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
