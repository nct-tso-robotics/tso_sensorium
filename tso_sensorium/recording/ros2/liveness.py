"""Always-on topic liveness tracking and camera feed capture for ROS 2."""

from __future__ import annotations

import time
from typing import Dict, Optional

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image

from tso_sensorium.recording.core import image_buffer_to_bgr_frame

JPEG_QUALITY = 80
QUEUE_DEPTH = 1


class TopicLivenessMonitor:
    """Tracks when each topic last published, independent of recording.

    Args:
        node: Node used to create the subscriptions.
        topic_message_types: Mapping of topic name to its message class.
        staleness_seconds: Age after which a topic counts as stale.
    """

    def __init__(
        self,
        node: Node,
        topic_message_types: Dict[str, type],
        staleness_seconds: float = 1.0,
    ):
        self.node = node
        self.staleness_seconds = staleness_seconds
        self._last_message_times: dict[str, Optional[float]] = {
            topic_name: None for topic_name in topic_message_types
        }
        self._subscriptions = [
            node.create_subscription(
                message_type,
                topic_name,
                self._make_callback(topic_name=topic_name),
                QUEUE_DEPTH,
            )
            for topic_name, message_type in topic_message_types.items()
        ]

    def _make_callback(self, topic_name: str):
        def callback(message) -> None:
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
        """Destroy all subscriptions."""
        for subscription in self._subscriptions:
            self.node.destroy_subscription(subscription)
        self._subscriptions = []


class CameraFeed:
    """Keeps the latest frame of an image topic as JPEG bytes.

    Args:
        node: Node used to create the subscription.
        topic_name: Topic publishing ``sensor_msgs/Image`` messages.
    """

    def __init__(self, node: Node, topic_name: str):
        self.node = node
        self.topic_name = topic_name
        self._latest_jpeg: Optional[bytes] = None
        self._subscription = node.create_subscription(
            Image, topic_name, self._callback, QUEUE_DEPTH
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
        """Destroy the image subscription."""
        self.node.destroy_subscription(self._subscription)
