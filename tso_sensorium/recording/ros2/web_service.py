"""ROS 2 wiring for the recording dashboard.

The service and Flask app live in ``tso_sensorium.recording.dashboard``;
this module builds them with the ROS 2 liveness monitor, camera feed, and
episode sessions. The caller owns the node and must spin it (e.g. an
executor in a background thread) for subscriptions to deliver. The server
is unauthenticated and should be exposed on trusted networks only.
"""

from __future__ import annotations

from typing import Optional

from rclpy.node import Node
from sensor_msgs.msg import Image

from tso_sensorium.recording.config import (
    RecordingServiceConfig,
    TopicRecorderConfig,
)
from tso_sensorium.recording.dashboard import RecordingService, create_app
from tso_sensorium.recording.message_fields import resolve_message_type
from tso_sensorium.recording.ros2.liveness import (
    CameraFeed,
    TopicLivenessMonitor,
)
from tso_sensorium.recording.ros2.session import EpisodeSession

__all__ = ["RecordingService", "build_recording_service", "create_app"]


def build_recording_service(
    node: Node, config: RecordingServiceConfig
) -> RecordingService:
    """Assemble the recording service from ROS 2 components.

    Args:
        node: Node used for all subscriptions.
        config: Service configuration.

    Returns:
        The wired recording service.
    """
    topic_message_types = {
        recorder.topic_name: (
            resolve_message_type(dotted_path=recorder.message_type)
            if isinstance(recorder, TopicRecorderConfig)
            else Image
        )
        for recorder in config.session.recorders
    }

    def session_factory(
        episode_name: Optional[str], recorder_names: Optional[list[str]]
    ) -> EpisodeSession:
        return EpisodeSession(
            node=node,
            config=config.session,
            episode_name=episode_name,
            recorder_names=recorder_names,
        )

    return RecordingService(
        config=config,
        liveness=TopicLivenessMonitor(
            node=node,
            topic_message_types=topic_message_types,
            staleness_seconds=config.staleness_seconds,
        ),
        camera_feed=(
            CameraFeed(node=node, topic_name=config.camera_topic)
            if config.camera_topic
            else None
        ),
        session_factory=session_factory,
    )
