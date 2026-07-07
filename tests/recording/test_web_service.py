"""Tests for tso_sensorium.recording.ros1.web_service module."""

import time

import numpy as np
import pytest

rospy = pytest.importorskip("rospy")
flask = pytest.importorskip("flask")

from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from tso_sensorium.episodes.generation_config import (  # noqa: E402
    CsvWriterConfig,
    DatasetGenerationConfig,
    StateSourceConfig,
)
from tso_sensorium.recording.config import (  # noqa: E402
    RecordingServiceConfig,
    RecordingSessionConfig,
    StereoViewConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)
from tso_sensorium.recording.ros1.web_service import (  # noqa: E402
    build_recording_service,
    RecordingService,
    create_app,
)

STATE_TOPIC = "/webtest/state"
CAMERA_TOPIC = "/webtest/camera"


@pytest.fixture
def service_factory(ros_node, tmp_path):
    services = []

    def factory(with_generation=False, stereo=False) -> RecordingService:
        generation = None
        if with_generation:
            generation = DatasetGenerationConfig(
                states=[StateSourceConfig(state_file="state.csv", columns=["data"])],
                writer=CsvWriterConfig(),
                n_jobs=1,
            )
        config = RecordingServiceConfig(
            session=RecordingSessionConfig(
                output_folder=str(tmp_path / "episodes"),
                recorders=[
                    TopicRecorderConfig(
                        file_name="state",
                        topic_name=STATE_TOPIC,
                        message_type="std_msgs.msg.String",
                        fields=["data"],
                    )
                ],
            ),
            staleness_seconds=1.0,
            camera_topic=CAMERA_TOPIC if stereo else "",
            stereo=StereoViewConfig(mode="duplicate") if stereo else None,
            generation=generation,
        )
        service = build_recording_service(config=config)
        services.append(service)
        return service

    yield factory
    for service in services:
        service.close()


@pytest.mark.integration
def test_dashboard_and_idle_status(service_factory):
    client = create_app(service=service_factory()).test_client()
    page = client.get("/")
    assert page.status_code == 200
    assert b"TSO Sensorium" in page.data
    status = client.get("/api/status").get_json()
    assert status["state"] == "idle"
    assert status["elapsed_seconds"] is None
    assert status["sensors"][0]["name"] == "state"
    assert status["sensors"][0]["alive"] is False
    assert client.get("/stream/camera").status_code == 404


@pytest.mark.integration
def test_recording_lifecycle_via_api(service_factory):
    service = service_factory()
    client = create_app(service=service).test_client()
    publisher = rospy.Publisher(STATE_TOPIC, String, queue_size=10)

    start = client.post("/api/recording/start", json={"episode_name": "ep1"})
    assert start.status_code == 200
    assert start.get_json()["episode_name"] == "ep1"
    assert client.post("/api/recording/start", json={}).status_code == 409

    time.sleep(1.0)  # Let the subscribers connect before publishing
    for index in range(3):
        publisher.publish(String(data=f"value_{index}"))
        time.sleep(0.2)

    status = client.get("/api/status").get_json()
    assert status["state"] == "recording"
    assert status["episode_name"] == "ep1"
    assert status["sensors"][0]["alive"] is True

    assert client.post("/api/recording/stop").status_code == 200
    assert client.post("/api/recording/stop").status_code == 409

    episodes = client.get("/api/episodes").get_json()
    assert episodes[0]["name"] == "ep1"
    assert any(file["name"] == "state.csv" for file in episodes[0]["files"])
    served = client.get("/episodes/ep1/state.csv")
    assert served.status_code == 200
    assert served.data.startswith(b"time,data")
    # Rows must actually contain the published values: an earlier bug
    # (AnyMsg liveness subscriptions) produced header-only files.
    assert b"value_0" in served.data


@pytest.mark.integration
def test_unknown_recorder_selection_rejected(service_factory):
    client = create_app(service=service_factory()).test_client()
    response = client.post(
        "/api/recording/start", json={"recorders": ["missing_sensor"]}
    )
    assert response.status_code == 400
    assert "missing_sensor" in response.get_json()["error"]


@pytest.mark.integration
def test_generation_via_api(service_factory, tmp_path):
    service = service_factory(with_generation=True)
    client = create_app(service=service).test_client()
    publisher = rospy.Publisher(STATE_TOPIC, String, queue_size=10)

    client.post("/api/recording/start", json={"episode_name": "gen_ep"})
    time.sleep(1.0)  # Let the subscribers connect before publishing
    for index in range(3):
        publisher.publish(String(data=f"value_{index}"))
        time.sleep(0.2)
    client.post("/api/recording/stop")

    alternate_root = tmp_path / "generated"
    start = client.post(
        "/api/generation/start",
        json={
            "overrides": {"writer": {"type": "csv", "output_root": str(alternate_root)}}
        },
    )
    assert start.status_code == 200
    for _ in range(50):
        generation = client.get("/api/status").get_json()["generation"]
        if generation["state"] != "running":
            break
        time.sleep(0.2)
    assert generation["state"] == "done"
    assert generation["written"] == ["gen_ep"]
    generated = alternate_root / "gen_ep" / "episode.csv"
    assert generated.is_file()
    # Aligned rows, not just the header.
    assert len(generated.read_text().strip().splitlines()) > 1


@pytest.mark.integration
def test_generation_without_config_rejected(service_factory):
    client = create_app(service=service_factory()).test_client()
    response = client.post("/api/generation/start")
    assert response.status_code == 400
    assert "No dataset generation configured" in response.get_json()["error"]


@pytest.mark.integration
def test_video_playback_endpoint_transcodes_to_h264(
    service_factory, ros_node, tmp_path, rng
):
    service = service_factory()
    client = create_app(service=service).test_client()
    image_publisher = rospy.Publisher("/webtest/image", Image, queue_size=10)

    video_config = VideoRecorderConfig(
        file_name="camera", topic_name="/webtest/image", frames_per_second=15
    )
    service.config.session.recorders.append(video_config)
    client.post("/api/recording/start", json={"episode_name": "video_ep"})
    time.sleep(1.0)  # Let the subscribers connect before publishing
    for _ in range(5):
        message = Image()
        message.height = 48
        message.width = 64
        message.encoding = "bgr8"
        message.step = 64 * 3
        message.data = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8).tobytes()
        message.header.stamp = rospy.Time.now()
        image_publisher.publish(message)
        time.sleep(0.1)
    client.post("/api/recording/stop")

    recorded = tmp_path / "episodes" / "video_ep" / "camera.mp4"
    assert recorded.is_file() and recorded.stat().st_size > 0
    playback = client.get("/episodes/video_ep/camera.mp4/playback")
    assert playback.status_code == 200
    assert playback.data[4:8] == b"ftyp"
    cache = tmp_path / "episodes" / "video_ep" / ".playback" / "camera.mp4"
    assert cache.is_file()


@pytest.mark.integration
def test_annotation_endpoints_round_trip(service_factory, tmp_path):
    service = service_factory(with_generation=True)
    client = create_app(service=service).test_client()
    episode_folder = tmp_path / "episodes" / "ann_ep"
    episode_folder.mkdir(parents=True)

    saved = client.put(
        "/api/episodes/ann_ep/annotations",
        json={
            "segments": [
                {"start": 0, "end": 100, "phase": 0},
                {
                    "start": 100,
                    "end": 200,
                    "phase": 1,
                    "source": "auto",
                    "language": "pull back",
                },
            ]
        },
    )
    assert saved.status_code == 200
    assert saved.get_json() == {"saved": 2}
    assert (episode_folder / "annotations.json").is_file()

    fetched = client.get("/api/episodes/ann_ep/annotations").get_json()
    assert [segment["phase"] for segment in fetched["segments"]] == [0, 1]
    assert fetched["segments"][0]["source"] == "manual"
    assert fetched["segments"][1]["language"] == "pull back"
    assert "legend" in fetched

    invalid = client.put(
        "/api/episodes/ann_ep/annotations",
        json={"segments": [{"start": 5, "end": 2, "phase": 0}]},
    )
    assert invalid.status_code == 400


@pytest.mark.integration
def test_stereo_and_anaglyph_streams(service_factory):
    service = service_factory(stereo=True)
    client = create_app(service=service).test_client()

    frame = np.full((8, 6, 3), 90, dtype=np.uint8)
    message = Image(height=8, width=6, encoding="bgr8", step=18, data=frame.tobytes())
    publisher = rospy.Publisher(CAMERA_TOPIC, Image, queue_size=1, latch=True)
    deadline = time.monotonic() + 5.0
    while service.camera_feed.latest_jpeg is None and time.monotonic() < deadline:
        publisher.publish(message)
        time.sleep(0.1)
    publisher.unregister()
    assert service.camera_feed.latest_jpeg is not None

    assert client.get("/api/status").get_json()["stereo_available"] is True
    for route in ("/stream/stereo", "/stream/anaglyph"):
        response = client.get(route)
        chunk = next(response.response)
        assert b"Content-Type: image/jpeg" in chunk
        response.close()

    no_stereo = create_app(service=service_factory()).test_client()
    assert no_stereo.get("/stream/stereo").status_code == 404
    assert no_stereo.get("/api/status").get_json()["stereo_available"] is False
