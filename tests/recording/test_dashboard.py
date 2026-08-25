"""Tests for tso_sensorium.recording.dashboard module."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Optional, Protocol
from unittest.mock import MagicMock, patch

import pytest

flask = pytest.importorskip("flask")

from tso_sensorium.recording.config import (  # noqa: E402
    RecorderConfig,
    RecordingSessionConfig,
    RecordingUIConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)
from tso_sensorium.recording.dashboard import (  # noqa: E402
    LivenessMonitor,
    RecordingService,
    RecordingSession,
    create_app,
)

STATE_TOPIC = "/test/state"
VIDEO_TOPIC = "/test/video"
OPTIONAL_TOPIC = "/test/optional"
STATE_FILE_NAME = "state"
VIDEO_FILE_NAME = "video"
OPTIONAL_FILE_NAME = "optional"


class RecordingServiceFactory(Protocol):
    """Creates a service with controlled recorder readiness."""

    def __call__(
        self,
        recorders: Optional[list[RecorderConfig]] = None,
        message_ages: Optional[dict[str, Optional[float]]] = None,
        alive_topics: Optional[set[str]] = None,
        output_folder: Optional[str] = None,
        record_rosbag: bool = False,
    ) -> tuple[RecordingService, MagicMock, MagicMock, MagicMock]: ...


@pytest.fixture
def recording_service_factory(
    tmp_path: Path,
) -> Generator[RecordingServiceFactory, None, None]:
    services: list[RecordingService] = []

    def factory(
        recorders: Optional[list[RecorderConfig]] = None,
        message_ages: Optional[dict[str, Optional[float]]] = None,
        alive_topics: Optional[set[str]] = None,
        output_folder: Optional[str] = None,
        record_rosbag: bool = False,
    ) -> tuple[RecordingService, MagicMock, MagicMock, MagicMock]:
        configured_recorders = (
            recorders
            if recorders is not None
            else [
                TopicRecorderConfig(
                    file_name=STATE_FILE_NAME,
                    topic_name=STATE_TOPIC,
                    message_type="std_msgs.msg.String",
                    fields=["data"],
                    required=True,
                )
            ]
        )
        liveness = MagicMock(spec=LivenessMonitor)
        liveness.snapshot.return_value = message_ages or {
            recorder.topic_name: None for recorder in configured_recorders
        }
        live = alive_topics or set()
        liveness.is_alive.side_effect = lambda topic_name: topic_name in live
        session = MagicMock(spec=RecordingSession)
        session.episode_name = "episode"
        session_factory = MagicMock(return_value=session)
        config = RecordingUIConfig(
            session=RecordingSessionConfig(
                output_folder=output_folder or str(tmp_path / "recordings"),
                record_rosbag=record_rosbag,
                recorders=configured_recorders,
            ),
            staleness_seconds=1.0,
        )
        service = RecordingService(
            config=config,
            liveness=liveness,
            camera_feed=None,
            session_factory=session_factory,
        )
        services.append(service)
        return service, liveness, session_factory, session

    yield factory

    for service in services:
        service.close()


@pytest.mark.integration
def test_start_rejects_required_source_that_never_published(
    recording_service_factory,
):
    service, _, session_factory, _ = recording_service_factory()
    client = create_app(service=service).test_client()

    response = client.post(
        "/api/recording/start",
        json={"episode_name": "missing_state"},
    )

    assert response.status_code == 409
    assert response.get_json() == {
        "error": (
            "Required recording sources are unavailable: "
            f"{STATE_FILE_NAME} ({STATE_TOPIC}: no messages received)"
        )
    }
    session_factory.assert_not_called()


@pytest.mark.integration
def test_start_rejects_stale_required_state_source(recording_service_factory):
    stale_age = 4.25
    service, _, session_factory, _ = recording_service_factory(
        message_ages={STATE_TOPIC: stale_age},
        alive_topics=set(),
    )
    client = create_app(service=service).test_client()

    response = client.post("/api/recording/start", json={})

    assert response.status_code == 409
    assert response.get_json() == {
        "error": (
            "Required recording sources are unavailable: "
            f"{STATE_FILE_NAME} ({STATE_TOPIC}: last message {stale_age:.3f} seconds ago)"
        )
    }
    session_factory.assert_not_called()


@pytest.mark.integration
def test_start_accepts_previously_observed_low_rate_video(
    recording_service_factory,
):
    video_recorder = VideoRecorderConfig(
        file_name=VIDEO_FILE_NAME,
        topic_name=VIDEO_TOPIC,
        frames_per_second=30.0,
        required=True,
    )
    service, _, session_factory, session = recording_service_factory(
        recorders=[video_recorder],
        message_ages={VIDEO_TOPIC: 4.25},
        alive_topics=set(),
    )
    client = create_app(service=service).test_client()

    response = client.post(
        "/api/recording/start",
        json={"episode_name": "low_rate_video"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"episode_name": "episode"}
    session_factory.assert_called_once_with(
        episode_name="low_rate_video",
        recorder_names=None,
    )
    session.start.assert_called_once_with()


@pytest.mark.integration
def test_start_rejects_long_stale_required_video(recording_service_factory):
    stale_age = 5.25
    video_recorder = VideoRecorderConfig(
        file_name=VIDEO_FILE_NAME,
        topic_name=VIDEO_TOPIC,
        frames_per_second=30.0,
        required=True,
    )
    service, _, session_factory, _ = recording_service_factory(
        recorders=[video_recorder],
        message_ages={VIDEO_TOPIC: stale_age},
        alive_topics=set(),
    )

    response = (
        create_app(service=service)
        .test_client()
        .post(
            "/api/recording/start",
            json={},
        )
    )

    assert response.status_code == 409
    assert response.get_json() == {
        "error": (
            "Required recording sources are unavailable: "
            f"{VIDEO_FILE_NAME} ({VIDEO_TOPIC}: last message {stale_age:.3f} seconds ago)"
        )
    }
    session_factory.assert_not_called()


@pytest.mark.integration
def test_start_allows_unavailable_optional_source(recording_service_factory):
    optional_recorder = TopicRecorderConfig(
        file_name=OPTIONAL_FILE_NAME,
        topic_name=OPTIONAL_TOPIC,
        message_type="std_msgs.msg.String",
        fields=["data"],
        required=False,
    )
    service, _, session_factory, session = recording_service_factory(
        recorders=[optional_recorder],
        message_ages={OPTIONAL_TOPIC: None},
        alive_topics=set(),
    )

    response = (
        create_app(service=service)
        .test_client()
        .post(
            "/api/recording/start",
            json={},
        )
    )

    assert response.status_code == 200
    session_factory.assert_called_once_with(
        episode_name=None,
        recorder_names=None,
    )
    session.start.assert_called_once_with()


@pytest.mark.integration
def test_start_validates_only_selected_recorders(recording_service_factory):
    state_recorder = TopicRecorderConfig(
        file_name=STATE_FILE_NAME,
        topic_name=STATE_TOPIC,
        message_type="std_msgs.msg.String",
        fields=["data"],
        required=True,
    )
    video_recorder = VideoRecorderConfig(
        file_name=VIDEO_FILE_NAME,
        topic_name=VIDEO_TOPIC,
        frames_per_second=30.0,
        required=True,
    )
    service, _, session_factory, _ = recording_service_factory(
        recorders=[state_recorder, video_recorder],
        message_ages={STATE_TOPIC: None, VIDEO_TOPIC: 2.0},
        alive_topics=set(),
    )

    response = (
        create_app(service=service)
        .test_client()
        .post(
            "/api/recording/start",
            json={"recorders": [VIDEO_FILE_NAME]},
        )
    )

    assert response.status_code == 200
    session_factory.assert_called_once_with(
        episode_name=None,
        recorder_names=[VIDEO_FILE_NAME],
    )


@pytest.mark.parametrize(
    "recorder_names, expected_error",
    [
        ([], "At least one recorder must be selected"),
        (
            [STATE_FILE_NAME, STATE_FILE_NAME],
            f"Duplicate recorder names: ['{STATE_FILE_NAME}']",
        ),
        (["missing"], "Unknown recorder names: ['missing']"),
    ],
)
@pytest.mark.integration
def test_start_rejects_invalid_recorder_selection(
    recording_service_factory,
    recorder_names: list[str],
    expected_error: str,
):
    service, _, session_factory, _ = recording_service_factory(
        message_ages={STATE_TOPIC: 0.1},
        alive_topics={STATE_TOPIC},
    )

    response = (
        create_app(service=service)
        .test_client()
        .post(
            "/api/recording/start",
            json={"recorders": recorder_names},
        )
    )

    assert response.status_code == 400
    assert response.get_json() == {"error": expected_error}
    session_factory.assert_not_called()


@pytest.mark.integration
def test_start_checks_output_folder_writability_before_creating_session(
    recording_service_factory,
):
    service, _, session_factory, _ = recording_service_factory(
        message_ages={STATE_TOPIC: 0.1},
        alive_topics={STATE_TOPIC},
    )
    output_folder = service.config.session.output_folder
    filesystem_error = PermissionError(13, "Permission denied", output_folder)

    with patch(
        "tso_sensorium.recording.dashboard.tempfile.TemporaryFile",
        side_effect=filesystem_error,
    ):
        response = (
            create_app(service=service)
            .test_client()
            .post(
                "/api/recording/start",
                json={},
            )
        )

    assert response.status_code == 400
    assert response.get_json() == {"error": str(filesystem_error)}
    session_factory.assert_not_called()


@pytest.mark.integration
def test_service_normalizes_relative_output_folder(
    recording_service_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(tmp_path)

    service, _, _, _ = recording_service_factory(output_folder="recordings")

    expected_root = (tmp_path / "recordings").resolve()
    assert service.config.session.output_folder == str(expected_root)
    assert service.library.recordings_root == expected_root
    assert expected_root.is_dir()


@pytest.mark.integration
def test_delete_rejects_episode_currently_recording(recording_service_factory):
    service, _, _, session = recording_service_factory(
        message_ages={STATE_TOPIC: 0.1},
        alive_topics={STATE_TOPIC},
    )
    client = create_app(service=service).test_client()
    assert client.post("/api/recording/start", json={}).status_code == 200

    with patch.object(service.library, "delete_episode") as delete_episode:
        response = client.delete(f"/api/episodes/{session.episode_name}")

    assert response.status_code == 409
    assert response.get_json() == {
        "error": "Cannot delete the episode currently recording"
    }
    delete_episode.assert_not_called()


@pytest.mark.integration
def test_output_folder_switch_normalizes_and_checks_relative_path(
    recording_service_factory,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    service, _, _, _ = recording_service_factory()
    alternate_root = tmp_path / "alternate"
    alternate_root.mkdir()
    monkeypatch.chdir(tmp_path)

    result = service.set_output_folder(path="alternate")

    assert result == {"output_folder": str(alternate_root.resolve())}
    assert service.config.session.output_folder == str(alternate_root.resolve())
    assert service.library.recordings_root == alternate_root.resolve()
