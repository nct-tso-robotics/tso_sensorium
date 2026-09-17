"""Tests for tso_sensorium.recording.ros2.web_service module."""

import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
flask = pytest.importorskip("flask")

from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from tso_sensorium.recording.config import (  # noqa: E402
    RecordingSessionConfig,
    RecordingUIConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)
from tso_sensorium.recording.ros2.web_service import (  # noqa: E402
    build_recording_service,
    create_app,
)

STATE_TOPIC = "/webtest2/state"
CAMERA_TOPIC = "/webtest2/camera"
QUEUE_DEPTH = 1


@pytest.fixture
def ros_node():
    rclpy.init()
    node = Node("tso_web_service_test")
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.fixture
def service_factory(ros_node, tmp_path):
    services = []

    def factory():
        config = RecordingUIConfig(
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
            camera_topic=CAMERA_TOPIC,
        )
        service = build_recording_service(node=ros_node, config=config)
        services.append(service)
        return service

    yield factory
    for service in services:
        service.close()


def _publish_until(node, publisher, message, condition, timeout_seconds=5.0):
    deadline = time.monotonic() + timeout_seconds
    while not condition() and time.monotonic() < deadline:
        publisher.publish(message)
        rclpy.spin_once(node, timeout_sec=0.05)
    if not condition():
        raise TimeoutError("Condition not met while publishing")


@pytest.mark.integration
def test_recording_lifecycle_over_http(service_factory, ros_node, tmp_path):
    service = service_factory()
    client = create_app(service=service).test_client()

    status = client.get("/api/status").get_json()
    assert status["state"] == "idle"
    assert status["recording_available"] is True
    assert status["sensors"][0]["alive"] is False

    state_publisher = ros_node.create_publisher(String, STATE_TOPIC, QUEUE_DEPTH)
    _publish_until(
        node=ros_node,
        publisher=state_publisher,
        message=String(data="value_0"),
        condition=lambda: service.liveness.is_alive(topic_name=STATE_TOPIC),
    )
    assert client.get("/api/status").get_json()["sensors"][0]["alive"] is True

    started = client.post("/api/recording/start", json={"episode_name": "ep_ros2"})
    assert started.status_code == 200
    episode_name = started.get_json()["episode_name"]
    assert episode_name.startswith("ep_ros2_")
    assert client.get("/api/status").get_json()["state"] == "recording"

    csv_path = tmp_path / "episodes" / episode_name / "state.csv"
    _publish_until(
        node=ros_node,
        publisher=state_publisher,
        message=String(data="value_1"),
        condition=lambda: csv_path.is_file() and csv_path.stat().st_size > 20,
    )
    stopped = client.post("/api/recording/stop", json={})
    assert stopped.get_json() == {"episode_name": episode_name}
    assert client.get("/api/status").get_json()["state"] == "idle"
    assert "value_1" in csv_path.read_text()

    served = client.get(f"/episodes/{episode_name}/state.csv")
    assert served.status_code == 200
    assert b"value_1" in served.data


@pytest.mark.integration
def test_camera_feed_and_stream(service_factory, ros_node):
    service = service_factory()
    client = create_app(service=service).test_client()

    frame = np.full((8, 6, 3), 90, dtype=np.uint8)
    message = Image(height=8, width=6, encoding="bgr8", step=18, data=frame.tobytes())
    camera_publisher = ros_node.create_publisher(Image, CAMERA_TOPIC, QUEUE_DEPTH)
    _publish_until(
        node=ros_node,
        publisher=camera_publisher,
        message=message,
        condition=lambda: service.camera_feed.latest_jpeg is not None,
    )

    response = client.get("/stream/camera")
    chunk = next(response.response)
    assert b"Content-Type: image/jpeg" in chunk
    response.close()


@pytest.mark.integration
def test_video_readiness_uses_camera_info_without_subscribing_to_images(
    ros_node: Node, tmp_path: Path
) -> None:
    image_topic = "/webtest2/idle/image_raw"
    status_topic = "/webtest2/idle/camera_info"
    config = RecordingUIConfig(
        session=RecordingSessionConfig(
            output_folder=str(tmp_path / "idle_video"),
            recorders=[
                VideoRecorderConfig(
                    file_name="camera",
                    topic_name=image_topic,
                    frames_per_second=30.0,
                    liveness_topic=status_topic,
                    liveness_message_type="sensor_msgs.msg.CameraInfo",
                )
            ],
        ),
        camera_topic="",
        staleness_seconds=1.0,
    )
    service = build_recording_service(node=ros_node, config=config)
    camera_info_publisher = ros_node.create_publisher(
        CameraInfo, status_topic, QUEUE_DEPTH
    )
    _publish_until(
        node=ros_node,
        publisher=camera_info_publisher,
        message=CameraInfo(height=48, width=64),
        condition=lambda: service.liveness.is_alive(topic_name=image_topic),
    )

    assert ros_node.count_subscribers(image_topic) == 0
    sensor = service.status()["sensors"][0]
    assert sensor["topic"] == image_topic
    assert sensor["ready"] is True
    service.close()


@pytest.mark.integration
def test_output_folder_switching(service_factory, ros_node, tmp_path):
    service = service_factory()
    client = create_app(service=service).test_client()

    other = tmp_path / "other_recordings"
    other.mkdir()
    switched = client.post("/api/recording/output_folder", json={"path": str(other)})
    assert switched.status_code == 200
    status = client.get("/api/status").get_json()
    assert status["output_folder"] == str(other)

    missing = client.post(
        "/api/recording/output_folder", json={"path": str(other / "nope")}
    )
    assert missing.status_code == 400

    state_publisher = ros_node.create_publisher(String, STATE_TOPIC, QUEUE_DEPTH)
    _publish_until(
        node=ros_node,
        publisher=state_publisher,
        message=String(data="readiness_probe"),
        condition=lambda: service.liveness.is_alive(topic_name=STATE_TOPIC),
    )
    started = client.post("/api/recording/start", json={"episode_name": "ep_locked"})
    assert started.status_code == 200
    episode_name = started.get_json()["episode_name"]
    locked = client.post("/api/recording/output_folder", json={"path": str(other)})
    assert locked.status_code == 409
    client.post("/api/recording/stop", json={})
    assert (other / episode_name).is_dir()
