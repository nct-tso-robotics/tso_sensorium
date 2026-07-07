"""Tests for tso_sensorium.recording.ros2.record module."""

import csv

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from tso_sensorium.recording.ros2 import (  # noqa: E402
    Recorder,
    RosTopicRecorder,
    VideoRecorder,
)

MESSAGE_COUNT = 5
FRAME_HEIGHT = 48
FRAME_WIDTH = 64


@pytest.fixture
def ros_node():
    rclpy.init()
    node = Node("tso_recording_test")
    yield node
    node.destroy_node()
    rclpy.shutdown()


@pytest.mark.integration
def test_recorders_capture_published_topics(ros_node, tmp_path, rng):
    recorder = Recorder(node=ros_node, output_folder=tmp_path)
    recorder.add_recorder(
        recorder=RosTopicRecorder(
            node=ros_node,
            output_folder=tmp_path,
            file_name="state",
            topic_name="/test/state",
            message_type=String,
            csv_header=["data"],
            get_cols_from_msg_func=lambda msg: [msg.data],
        )
    ).add_recorder(
        recorder=VideoRecorder(
            node=ros_node,
            output_folder=tmp_path,
            file_name="camera",
            frames_per_second=30.0,
            topic_name="/test/image",
        )
    )
    recorder.start_recording()

    state_publisher = ros_node.create_publisher(String, "/test/state", 10)
    image_publisher = ros_node.create_publisher(Image, "/test/image", 10)
    for index in range(MESSAGE_COUNT):
        state_message = String()
        state_message.data = f"value_{index}"
        state_publisher.publish(state_message)

        image_message = Image()
        image_message.height = FRAME_HEIGHT
        image_message.width = FRAME_WIDTH
        image_message.encoding = "bgr8"
        image_message.step = FRAME_WIDTH * 3
        image_message.data = rng.integers(
            0, 255, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
        ).tobytes()
        image_message.header.stamp = ros_node.get_clock().now().to_msg()
        image_publisher.publish(image_message)
        for _ in range(10):
            rclpy.spin_once(ros_node, timeout_sec=0.05)
    recorder.close()

    with open(tmp_path / "state.csv", newline="") as state_file:
        state_rows = list(csv.reader(state_file))
    with open(tmp_path / "camera.csv", newline="") as camera_file:
        camera_rows = list(csv.reader(camera_file))
    assert state_rows[0] == ["time", "data"]
    assert [row[1] for row in state_rows[1:]] == [
        f"value_{index}" for index in range(MESSAGE_COUNT)
    ]
    assert camera_rows[0] == [
        "time",
        "encoding",
        "height",
        "width",
        "is_bigendian",
        "step",
    ]
    assert camera_rows[1][1:] == ["bgr8", "48", "64", "0", "192"]
    assert len(camera_rows) == MESSAGE_COUNT + 1
    assert (tmp_path / "camera.mp4").stat().st_size > 0
