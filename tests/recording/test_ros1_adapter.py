"""Tests for tso_sensorium.recording.ros1.record module."""

import csv
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

rospy = pytest.importorskip("rospy")

from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from tso_sensorium.recording.ros1 import (  # noqa: E402
    Recorder,
    RosTopicRecorder,
    VideoRecorder,
)

MESSAGE_COUNT = 5
FRAME_HEIGHT = 48
FRAME_WIDTH = 64
CSV_RECORDER_PATH = "tso_sensorium.recording.ros1.record.TimestampedCsvRecorder"
SUBSCRIBER_PATH = "tso_sensorium.recording.ros1.record.rospy.Subscriber"


@pytest.mark.unit
def test_topic_recorder_uses_configured_subscription_queue() -> None:
    with patch(CSV_RECORDER_PATH) as csv_recorder_class:
        recorder = RosTopicRecorder(
            output_folder="out",
            file_name="state",
            topic_name="/test/state",
            message_type=String,
            csv_header=["data"],
            get_cols_from_msg_func=MagicMock(),
            queue_size=100,
        )
        with patch(SUBSCRIBER_PATH) as subscriber_class:
            recorder.subscribe()
        recorder.close()

    subscriber_class.assert_called_once_with(
        "/test/state",
        String,
        recorder.callback,
        queue_size=100,
    )
    subscriber_class.return_value.unregister.assert_called_once_with()
    csv_recorder_class.return_value.close.assert_called_once_with()


@pytest.mark.integration
def test_recorders_capture_published_topics(ros_node, tmp_path, rng):
    recorder = Recorder(output_folder=tmp_path)
    recorder.add_recorder(
        recorder=RosTopicRecorder(
            output_folder=tmp_path,
            file_name="state",
            topic_name="/test/state",
            message_type=String,
            csv_header=["data"],
            get_cols_from_msg_func=lambda msg: [msg.data],
        )
    ).add_recorder(
        recorder=VideoRecorder(
            output_folder=tmp_path,
            file_name="camera",
            frames_per_second=30.0,
            topic_name="/test/image",
        )
    )
    recorder.start_recording()

    state_publisher = rospy.Publisher("/test/state", String, queue_size=10)
    image_publisher = rospy.Publisher("/test/image", Image, queue_size=10)
    time.sleep(1.0)  # Let the subscribers connect before publishing

    for index in range(MESSAGE_COUNT):
        state_publisher.publish(String(data=f"value_{index}"))

        image_message = Image()
        image_message.height = FRAME_HEIGHT
        image_message.width = FRAME_WIDTH
        image_message.encoding = "bgr8"
        image_message.step = FRAME_WIDTH * 3
        image_message.data = rng.integers(
            0, 255, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
        ).tobytes()
        image_message.header.stamp = rospy.Time.now()
        image_publisher.publish(image_message)
        time.sleep(0.2)

    time.sleep(0.5)
    recorder.close()

    with open(tmp_path / "state.csv", newline="") as state_file:
        state_rows = list(csv.reader(state_file))
    with open(tmp_path / "camera.csv", newline="") as camera_file:
        camera_rows = list(csv.reader(camera_file))
    assert state_rows[0] == ["time", "data"]
    assert [row[1] for row in state_rows[1:]] == [
        f"value_{index}" for index in range(MESSAGE_COUNT)
    ]
    assert camera_rows[1][1:] == ["bgr8", "48", "64", "0", "192"]
    assert len(camera_rows) == MESSAGE_COUNT + 1
    assert (tmp_path / "camera.mp4").stat().st_size > 0
