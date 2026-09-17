"""ROS 1 wiring for the recording dashboard.

The service and Flask app live in ``tso_sensorium.recording.dashboard``;
this module builds them with the ROS 1 liveness monitor, camera feed, and
episode sessions. The server is unauthenticated and should be exposed on
trusted networks only.
"""

from __future__ import annotations

from typing import Optional

from tso_sensorium.recording.config import RecordingUIConfig
from tso_sensorium.recording.dashboard import RecordingService, create_app
from tso_sensorium.recording.liveness import build_liveness_sources
from tso_sensorium.recording.ros1.liveness import (
    CameraFeed,
    TopicLivenessMonitor,
)
from tso_sensorium.recording.ros1.session import EpisodeSession

__all__ = ["RecordingService", "build_recording_service", "create_app"]


def build_recording_service(config: RecordingUIConfig) -> RecordingService:
    """Assemble the recording service from ROS 1 components.

    Args:
        config: Service configuration.

    Returns:
        The wired recording service.
    """

    def session_factory(
        episode_name: Optional[str], recorder_names: Optional[list[str]]
    ) -> EpisodeSession:
        return EpisodeSession(
            config=config.session,
            episode_name=episode_name,
            recorder_names=recorder_names,
        )

    return RecordingService(
        config=config,
        liveness=TopicLivenessMonitor(
            topic_sources=build_liveness_sources(recorders=config.session.recorders),
            staleness_seconds=config.staleness_seconds,
        ),
        camera_feed=(
            CameraFeed(topic_name=config.camera_topic) if config.camera_topic else None
        ),
        session_factory=session_factory,
    )
