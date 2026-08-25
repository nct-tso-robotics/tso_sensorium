"""ROS 2 topic recording adapters.

Subscribes to ROS 2 topics through a caller-provided node and delegates
file writing to the framework-free recorders in
``tso_sensorium.recording.core``. A ``Recorder`` orchestrates multiple
topic recorders and an optional rosbag2 capture.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from rclpy.node import Node
from sensor_msgs.msg import Image

from tso_sensorium.recording.core import (
    IMAGE_METADATA_HEADER,
    TimestampedCsvRecorder,
    VideoFileWriter,
    image_buffer_to_bgr_frame,
)

NANOSECONDS_PER_SECOND = 1_000_000_000
ROSBAG_DIRECTORY_NAME = "recording_bag"


class RosTopicRecorder:
    """Records a ROS 2 topic to a timestamped CSV file.

    Messages are timestamped with their header stamp when available,
    falling back to the node clock's receive time.

    Args:
        node: Node used to create the subscription and read the clock.
        output_folder: Directory to save the CSV file.
        file_name: Name of the CSV file, without extension.
        topic_name: ROS topic to subscribe to.
        message_type: ROS message type class.
        csv_header: Column headers written after the time column.
        get_cols_from_msg_func: Extracts the row values from a message.
        queue_size: Pending messages retained when the callback falls behind.
    """

    def __init__(
        self,
        node: Node,
        output_folder: Path | str,
        file_name: str,
        topic_name: str,
        message_type: type,
        csv_header: Sequence[str],
        get_cols_from_msg_func: Callable[[Any], Sequence[Any]],
        queue_size: int = 1,
    ):
        self.node = node
        self.topic_name = topic_name
        self.message_type = message_type
        self.get_cols_from_msg_func = get_cols_from_msg_func
        self.queue_size = queue_size
        self.subscription = None
        self.csv_recorder = TimestampedCsvRecorder(
            output_folder=output_folder,
            file_name=file_name,
            csv_header=csv_header,
        )

    def subscribe(self) -> None:
        """Start receiving messages from the topic."""
        self.subscription = self.node.create_subscription(
            self.message_type,
            self.topic_name,
            self.callback,
            self.queue_size,
        )

    def _get_message_timestamp(self, msg: Any) -> int:
        if hasattr(msg, "header"):
            stamp = msg.header.stamp
            return stamp.sec * NANOSECONDS_PER_SECOND + stamp.nanosec
        return self.node.get_clock().now().nanoseconds

    def callback(self, msg: Any) -> None:
        """Write one received message to the CSV file."""
        self.csv_recorder.write_row(
            timestamp_nanoseconds=self._get_message_timestamp(msg=msg),
            values=self.get_cols_from_msg_func(msg),
        )

    def close(self) -> None:
        """Stop receiving messages and close the CSV file."""
        if self.subscription is not None:
            self.node.destroy_subscription(self.subscription)
            self.subscription = None
        self.csv_recorder.close()


class VideoRecorder(RosTopicRecorder):
    """Records an image topic to a video file plus a metadata CSV.

    Args:
        node: Node used to create the subscription and read the clock.
        output_folder: Directory to save the video and CSV files.
        file_name: Base name for both files, without extension.
        frames_per_second: Playback frame rate of the written video.
        topic_name: ROS topic publishing ``sensor_msgs/Image`` messages.
        lossless_compression: Whether to encode losslessly. Lossless files
            are considerably larger.
        queue_size: Pending frames retained when the callback falls behind.
    """

    def __init__(
        self,
        node: Node,
        output_folder: Path | str,
        file_name: str,
        frames_per_second: float,
        topic_name: str,
        lossless_compression: bool = False,
        queue_size: int = 1,
    ):
        super().__init__(
            node=node,
            output_folder=output_folder,
            file_name=file_name,
            topic_name=topic_name,
            message_type=Image,
            csv_header=IMAGE_METADATA_HEADER,
            get_cols_from_msg_func=self._get_image_metadata,
            queue_size=queue_size,
        )
        self.video_writer = VideoFileWriter(
            output_folder=output_folder,
            file_name=file_name,
            frames_per_second=frames_per_second,
            lossless_compression=lossless_compression,
        )

    @staticmethod
    def _get_image_metadata(msg: Any) -> list:
        return [msg.encoding, msg.height, msg.width, msg.is_bigendian, msg.step]

    def callback(self, msg: Any) -> None:
        """Write the frame to the video file and its metadata to the CSV."""
        timestamp_nanoseconds = self._get_message_timestamp(msg=msg)
        self.csv_recorder.write_row(
            timestamp_nanoseconds=timestamp_nanoseconds,
            values=self._get_image_metadata(msg=msg),
        )
        frame = image_buffer_to_bgr_frame(
            data=msg.data,
            height=msg.height,
            width=msg.width,
            encoding=msg.encoding,
        )
        self.video_writer.write_frame(
            frame=frame,
            timestamp_nanoseconds=timestamp_nanoseconds,
        )

    def close(self) -> None:
        """Close the CSV file and finalize the video."""
        super().close()
        self.video_writer.close()


class Recorder:
    """Orchestrates multiple topic recorders and optional rosbag2 capture.

    Args:
        node: Node used for logging.
        output_folder: Base directory for all recordings.
        record_rosbag: Whether to also record a rosbag2 of the registered
            rosbag topics.
    """

    def __init__(
        self,
        node: Node,
        output_folder: Path | str,
        record_rosbag: bool = False,
    ):
        self.node = node
        self.output_folder = output_folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        self.recorders: list[RosTopicRecorder] = []
        self.record_rosbag = record_rosbag
        self.rosbag_topics: list[str] = []
        self.rosbag_process: Optional[subprocess.Popen] = None

    def add_recorder(self, recorder: RosTopicRecorder) -> "Recorder":
        """Register a topic recorder.

        Returns:
            Self, for method chaining.
        """
        self.recorders.append(recorder)
        return self

    def add_rosbag_topics(self, topics: list[str]) -> "Recorder":
        """Set the topics captured by the rosbag2 recording.

        Returns:
            Self, for method chaining.
        """
        self.rosbag_topics = topics
        return self

    def start_recording(self) -> None:
        """Subscribe all recorders and start rosbag2 capture if enabled."""
        if self.record_rosbag and self.rosbag_topics:
            rosbag_directory = str(Path(self.output_folder, ROSBAG_DIRECTORY_NAME))
            command = ["ros2", "bag", "record", "-o", rosbag_directory]
            self.rosbag_process = subprocess.Popen(command + self.rosbag_topics)
            self.node.get_logger().info(f"Started rosbag recording: {rosbag_directory}")
        for recorder in self.recorders:
            recorder.subscribe()

    def close(self) -> None:
        """Close all recorders and stop the rosbag2 capture."""
        for recorder in self.recorders:
            recorder.close()
        if self.rosbag_process is not None:
            self.rosbag_process.terminate()
            self.rosbag_process.wait()
