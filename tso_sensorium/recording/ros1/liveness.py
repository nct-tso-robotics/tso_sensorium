"""Always-on topic liveness tracking and camera feed capture for ROS 1."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image

from tso_sensorium.recording.core import image_buffer_to_bgr_frame
from tso_sensorium.recording.liveness import LivenessSource

JPEG_QUALITY = 80


class TopicLivenessMonitor:
    """Tracks when each topic last published, independent of recording.

    Subscriptions use each topic's real message type: rospy delivers one
    deserialization per topic per node, so subscribing with a mismatched
    type here would corrupt the recorders' callbacks on the same topics.

    Args:
        topic_sources: Status sources keyed by the recorded topic name.
        staleness_seconds: Age after which a topic counts as stale.
    """

    def __init__(
        self,
        topic_sources: dict[str, LivenessSource],
        staleness_seconds: float = 1.0,
    ) -> None:
        self.staleness_seconds = staleness_seconds
        self._last_message_times: dict[str, Optional[float]] = {
            topic_name: None for topic_name in topic_sources
        }
        self._subscribers = [
            rospy.Subscriber(
                name=source.topic_name,
                data_class=source.message_type,
                callback=self._make_callback(topic_name=topic_name),
                queue_size=1,
            )
            for topic_name, source in topic_sources.items()
        ]

    def _make_callback(self, topic_name: str) -> Callable[[Any], None]:
        def callback(message: Any) -> None:
            self._last_message_times[topic_name] = time.monotonic()

        return callback

    def snapshot(self) -> dict[str, Optional[float]]:
        """Return the age in seconds of each topic's last message.

        Topics that never published have age ``None``.
        """
        now = time.monotonic()
        return {
            topic_name: (round(now - last_time, 3) if last_time is not None else None)
            for topic_name, last_time in self._last_message_times.items()
        }

    def is_alive(self, topic_name: str) -> bool:
        """Whether the topic published within the staleness window."""
        last_time = self._last_message_times.get(topic_name)
        return (
            last_time is not None
            and time.monotonic() - last_time < self.staleness_seconds
        )

    def close(self) -> None:
        """Unsubscribe from all topics."""
        for subscriber in self._subscribers:
            subscriber.unregister()
        self._subscribers = []


class CameraFeed:
    """Keeps the latest frame of an image topic as JPEG bytes.

    Args:
        topic_name: Topic publishing ``sensor_msgs/Image`` messages.
    """

    def __init__(self, topic_name: str):
        self.topic_name = topic_name
        self._latest_jpeg: Optional[bytes] = None
        self._subscriber = rospy.Subscriber(
            topic_name, Image, self._callback, queue_size=1
        )

    def _callback(self, message: Image) -> None:
        frame = image_buffer_to_bgr_frame(
            data=message.data,
            height=message.height,
            width=message.width,
            encoding=message.encoding,
        )
        encoded_ok, encoded = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if encoded_ok:
            self._latest_jpeg = np.asarray(encoded).tobytes()

    @property
    def latest_jpeg(self) -> Optional[bytes]:
        """The most recent frame as JPEG bytes, or ``None`` before the first."""
        return self._latest_jpeg

    def close(self) -> None:
        """Unsubscribe from the image topic."""
        self._subscriber.unregister()
