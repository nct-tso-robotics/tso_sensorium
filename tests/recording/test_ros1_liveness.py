"""Tests for tso_sensorium.recording.ros1.liveness module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("rospy")

from sensor_msgs.msg import CameraInfo  # noqa: E402

from tso_sensorium.recording.liveness import LivenessSource  # noqa: E402
from tso_sensorium.recording.ros1.liveness import TopicLivenessMonitor  # noqa: E402

IMAGE_TOPIC = "/camera/image_raw"
STATUS_TOPIC = "/camera/camera_info"
SUBSCRIBER_PATH = "tso_sensorium.recording.ros1.liveness.rospy.Subscriber"
CLOCK_PATH = "tso_sensorium.recording.ros1.liveness.time.monotonic"


class TestTopicLivenessMonitor:
    @pytest.mark.unit
    def test_subscribes_to_status_but_tracks_the_recorded_topic(self) -> None:
        source = LivenessSource(topic_name=STATUS_TOPIC, message_type=CameraInfo)
        with (
            patch(target=SUBSCRIBER_PATH) as subscriber,
            patch(target=CLOCK_PATH, return_value=10.0) as clock,
        ):
            monitor = TopicLivenessMonitor(
                topic_sources={IMAGE_TOPIC: source}, staleness_seconds=1.0
            )
            callback = subscriber.call_args.kwargs["callback"]
            subscriber.assert_called_once_with(
                name=STATUS_TOPIC,
                data_class=CameraInfo,
                callback=callback,
                queue_size=1,
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
            subscriber.return_value.unregister.assert_called_once_with()

    @pytest.mark.unit
    def test_status_callbacks_do_not_mark_other_images_alive(self) -> None:
        sources = {
            "/left/image_raw": LivenessSource(
                topic_name="/left/camera_info", message_type=CameraInfo
            ),
            "/right/image_raw": LivenessSource(
                topic_name="/right/camera_info", message_type=CameraInfo
            ),
        }
        with (
            patch(target=SUBSCRIBER_PATH) as subscriber,
            patch(target=CLOCK_PATH, return_value=10.0),
        ):
            monitor = TopicLivenessMonitor(topic_sources=sources, staleness_seconds=1.0)
            subscriber.call_args_list[0].kwargs["callback"](CameraInfo())

            assert monitor.snapshot() == {
                "/left/image_raw": 0.0,
                "/right/image_raw": None,
            }
            monitor.close()
