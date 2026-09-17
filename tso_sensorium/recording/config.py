"""Configuration schema for recording sessions.

Requires Python 3.9+; the recording adapters themselves stay importable
without it.
"""

from __future__ import annotations

import abc
from typing import Annotated, List, Literal, Optional, Union

from pydantic import Field

from tso_sensorium.configuration import ConfigModel
from tso_sensorium.episodes.generation_config import DatasetGenerationConfig

CAMERA_INFO_MESSAGE_TYPE = "sensor_msgs.msg.CameraInfo"


class RecorderConfig(ConfigModel, abc.ABC):
    """One recorded stream: a topic to capture and where to store it.

    Args:
        file_name: Base name of the output files, without extension.
        topic_name: ROS topic to subscribe to.
        queue_size: Pending messages retained when callbacks fall behind.
        required: Whether recording must refuse to start when this source
            is unavailable. Optional sources may be recorded when present.
    """

    file_name: str = ""
    topic_name: str = ""
    queue_size: int = Field(default=1, ge=1)
    required: bool = True


class TopicRecorderConfig(RecorderConfig):
    """A topic recorded to a timestamped CSV file.

    Args:
        message_type: Dotted path of the message class, e.g.
            "testbed_msgs.msg.RobotState".
        fields: Message attribute paths written as columns; nested
            attributes use dots, e.g. "transform.rotation.x".
        csv_header: Column names; defaults to the field paths with dots
            replaced by underscores.
    """

    type: Literal["topic"] = "topic"
    message_type: str = ""
    fields: List[str] = Field(default_factory=list)
    csv_header: Optional[List[str]] = None


class VideoRecorderConfig(RecorderConfig):
    """An image topic recorded to a video file plus a metadata CSV.

    Args:
        frames_per_second: Playback frame rate of the written video.
        lossless_compression: Whether to encode losslessly. Lossless files
            are considerably larger.
        liveness_topic: Lightweight per-frame status topic used by the UI.
            Required when using the recording UI; no topic name is inferred.
            Command-line recording does not use readiness monitoring.
        liveness_message_type: Message class published on the status topic.
            This does not change the image type used for recording.
    """

    type: Literal["video"] = "video"
    frames_per_second: float = 30.0
    lossless_compression: bool = False
    liveness_topic: Optional[str] = None
    liveness_message_type: str = CAMERA_INFO_MESSAGE_TYPE


AnyRecorderConfig = Annotated[
    Union[TopicRecorderConfig, VideoRecorderConfig], Field(discriminator="type")
]


class RecordingSessionConfig(ConfigModel):
    """A full recording session.

    Args:
        output_folder: Folder storing the recordings; the episode name is
            appended to it.
        episode_name: Optional descriptive prefix. The creation timestamp is
            always appended to the saved episode name.
        record_rosbag: Whether to also record a rosbag of
            ``rosbag_topics``.
        rosbag_topics: Topics captured in the rosbag.
        recorders: Streams recorded in this session.
    """

    output_folder: str = ""
    episode_name: Optional[str] = None
    record_rosbag: bool = False
    rosbag_topics: List[str] = Field(default_factory=list)
    recorders: List[AnyRecorderConfig] = Field(default_factory=list)


class LibraryAppConfig(ConfigModel):
    """Standalone annotation and generation app.

    Args:
        recordings_root: Directory containing one folder per episode.
        generation: Dataset generation offered in the dashboard; disabled
            when ``None``.
        host: Interface the HTTP server binds. Defaults to loopback; the
            dashboard browses and serves the filesystem without
            authentication, so set ``0.0.0.0`` only on a trusted network.
        port: TCP port the HTTP server binds.
    """

    recordings_root: str = ""
    generation: Optional[DatasetGenerationConfig] = None
    host: str = "127.0.0.1"
    port: int = 8080


class RecordingUIConfig(ConfigModel):
    """Browser UI for interactive recording.

    Args:
        session: Recording session configuration, including the recorder
            catalog offered in the dashboard.
        host: Interface the HTTP server binds. Defaults to loopback; the
            dashboard browses and serves the filesystem without
            authentication, so set ``0.0.0.0`` only on a trusted network.
        port: TCP port the HTTP server binds.
        camera_topic: Image topic shown as the live feed; empty disables
            the feed.
        staleness_seconds: Age after which a sensor counts as stale.
        generation: Dataset generation run offered in the dashboard; its
            ``recordings_root`` defaults to the session output folder.
    """

    session: RecordingSessionConfig = Field(
        default_factory=lambda: RecordingSessionConfig()
    )
    host: str = "127.0.0.1"
    port: int = 8080
    camera_topic: str = ""
    staleness_seconds: float = 1.0
    generation: Optional[DatasetGenerationConfig] = None
