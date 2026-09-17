"""Tests for tso_sensorium.recording.ros2.liveness module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("rclpy")

from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import CameraInfo  # noqa: E402

from tso_sensorium.recording.liveness import LivenessSource  # noqa: E402
from tso_sensorium.recording.ros2.liveness import TopicLivenessMonitor  # noqa: E402

IMAGE_TOPIC = "/camera/image_raw"
STATUS_TOPIC = "/camera/camera_info"
CLOCK_PATH = "tso_sensorium.recording.ros2.liveness.time.monotonic"


@pytest.mark.unit
def test_status_subscription_tracks_image_freshness_and_releases_resources() -> None:
    node = MagicMock(spec=Node)
    source = LivenessSource(topic_name=STATUS_TOPIC, message_type=CameraInfo)
    with patch(target=CLOCK_PATH, return_value=10.0) as clock:
        monitor = TopicLivenessMonitor(
            node=node,
            topic_sources={IMAGE_TOPIC: source},
            staleness_seconds=1.0,
        )
        callback = node.create_subscription.call_args.kwargs["callback"]
        node.create_subscription.assert_called_once_with(
            msg_type=CameraInfo,
            topic=STATUS_TOPIC,
            callback=callback,
            qos_profile=1,
        )
        assert monitor.snapshot() == {IMAGE_TOPIC: None}
        assert not monitor.is_alive(topic_name=IMAGE_TOPIC)

        callback(CameraInfo())
        clock.return_value = 10.25
        assert monitor.snapshot() == {IMAGE_TOPIC: 0.25}
        assert monitor.is_alive(topic_name=IMAGE_TOPIC)

        clock.return_value = 11.0
        assert not monitor.is_alive(topic_name=IMAGE_TOPIC)
        assert not monitor.is_alive(topic_name="/missing")
        monitor.close()
        monitor.close()
        node.destroy_subscription.assert_called_once_with(
            node.create_subscription.return_value
        )
