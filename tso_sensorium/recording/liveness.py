"""Lightweight subscription sources for recording readiness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from tso_sensorium.recording.config import (
    RecorderConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)
from tso_sensorium.recording.message_fields import resolve_message_type


@dataclass(frozen=True)
class LivenessSource:
    """A typed subscription that reports freshness for a recorded stream."""

    topic_name: str
    message_type: type


def build_liveness_sources(
    recorders: Sequence[RecorderConfig],
) -> dict[str, LivenessSource]:
    """Choose status subscriptions without receiving video payloads.

    Args:
        recorders: Streams offered by the recording dashboard.

    Returns:
        Status sources keyed by the corresponding recorded topic name.

    Raises:
        ValueError: A video recorder has no explicit status topic.
        TypeError: A recorder configuration is unsupported.
    """
    sources = {}
    for recorder in recorders:
        if isinstance(recorder, VideoRecorderConfig):
            topic_name = recorder.liveness_topic
            if not topic_name or not topic_name.strip():
                raise ValueError(
                    f"Video recorder '{recorder.topic_name}' requires an explicit "
                    "liveness_topic for the recording UI."
                )
            message_type = recorder.liveness_message_type
        elif isinstance(recorder, TopicRecorderConfig):
            topic_name = recorder.topic_name
            message_type = recorder.message_type
        else:
            raise TypeError(
                f"Unsupported recorder config type: {type(recorder).__name__}"
            )
        sources[recorder.topic_name] = LivenessSource(
            topic_name=topic_name,
            message_type=resolve_message_type(dotted_path=message_type),
        )
    return sources
