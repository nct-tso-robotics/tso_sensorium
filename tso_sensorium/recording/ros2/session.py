"""Reusable episode recording sessions for ROS 2.

Builds recorders from configuration and manages the lifecycle of one
recorded episode, so the CLI script and the web service share the same
recording path.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional

from rclpy.node import Node

from tso_sensorium.recording.config import (
    RecorderConfig,
    RecordingSessionConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)
from tso_sensorium.recording.message_fields import (
    MessageFieldExtractor,
    resolve_message_type,
)
from tso_sensorium.recording.ros2.record import (
    Recorder,
    RosTopicRecorder,
    VideoRecorder,
)


def generate_time_based_string() -> str:
    """Generate an episode name from the current time."""
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S_%f")


def build_recorder(
    node: Node, recorder_config: RecorderConfig, output_folder: Path | str
) -> RosTopicRecorder:
    """Instantiate the ROS 2 recorder described by a config entry.

    Args:
        node: Node used to create the subscription.
        recorder_config: Topic or video recorder configuration.
        output_folder: Directory receiving the recorded files.

    Returns:
        The configured recorder.
    """
    if isinstance(recorder_config, VideoRecorderConfig):
        return VideoRecorder(
            node=node,
            output_folder=output_folder,
            file_name=recorder_config.file_name,
            frames_per_second=recorder_config.frames_per_second,
            topic_name=recorder_config.topic_name,
            lossless_compression=recorder_config.lossless_compression,
            queue_size=recorder_config.queue_size,
        )
    if isinstance(recorder_config, TopicRecorderConfig):
        extractor = MessageFieldExtractor(fields=recorder_config.fields)
        csv_header = (
            recorder_config.csv_header
            if recorder_config.csv_header is not None
            else extractor.csv_header
        )
        return RosTopicRecorder(
            node=node,
            output_folder=output_folder,
            file_name=recorder_config.file_name,
            topic_name=recorder_config.topic_name,
            message_type=resolve_message_type(dotted_path=recorder_config.message_type),
            csv_header=csv_header,
            get_cols_from_msg_func=extractor,
            queue_size=recorder_config.queue_size,
        )
    raise TypeError(
        f"Unsupported recorder config type: {type(recorder_config).__name__}"
    )


class EpisodeSession:
    """One recorded episode, from subscription to closed files.

    Args:
        node: Node used to create the subscriptions.
        config: Recording session configuration.
        episode_name: Name of the episode; defaults to a time-based
            string.
        recorder_names: Subset of configured recorders to use, by
            ``file_name``; all when ``None``.
    """

    def __init__(
        self,
        node: Node,
        config: RecordingSessionConfig,
        episode_name: Optional[str] = None,
        recorder_names: Optional[list[str]] = None,
    ):
        self.episode_name = (
            episode_name if episode_name else generate_time_based_string()
        )
        self.output_folder = Path(config.output_folder).expanduser() / self.episode_name
        selected_configs = self._select_recorders(
            config=config, recorder_names=recorder_names
        )
        self.recorder = Recorder(
            node=node,
            output_folder=self.output_folder,
            record_rosbag=config.record_rosbag,
        )
        for recorder_config in selected_configs:
            self.recorder.add_recorder(
                recorder=build_recorder(
                    node=node,
                    recorder_config=recorder_config,
                    output_folder=self.output_folder,
                )
            )
        if config.record_rosbag:
            self.recorder.add_rosbag_topics(topics=config.rosbag_topics)

    @staticmethod
    def _select_recorders(
        config: RecordingSessionConfig,
        recorder_names: Optional[list[str]],
    ) -> list[RecorderConfig]:
        if recorder_names is None:
            return list(config.recorders)
        available = {recorder.file_name: recorder for recorder in config.recorders}
        unknown = [name for name in recorder_names if name not in available]
        if unknown:
            raise ValueError(f"Unknown recorder names: {unknown}")
        return [available[name] for name in recorder_names]

    def start(self) -> None:
        """Subscribe all recorders and start rosbag capture if enabled."""
        self.recorder.start_recording()

    def close(self) -> None:
        """Unsubscribe everything and finalize all files."""
        self.recorder.close()
