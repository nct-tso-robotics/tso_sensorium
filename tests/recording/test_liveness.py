"""Tests for tso_sensorium.recording.liveness module."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

import pytest

from tso_sensorium.recording.config import (
    RecorderConfig,
    TopicRecorderConfig,
    VideoRecorderConfig,
)
from tso_sensorium.recording.liveness import (
    LivenessSource,
    build_liveness_sources,
)

RESOLVER_PATH = "tso_sensorium.recording.liveness.resolve_message_type"
CAMERA_INFO_MESSAGE_TYPE = "sensor_msgs.msg.CameraInfo"
HEARTBEAT_MESSAGE_TYPE = "std_msgs.msg.Header"
STATE_MESSAGE_TYPE = "std_msgs.msg.String"


class TestBuildLivenessSources:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "image_topic, status_topic, message_type",
        [
            (
                "/camera/left/image_raw",
                "/camera/left/camera_info",
                CAMERA_INFO_MESSAGE_TYPE,
            ),
            ("image_raw", "camera_status", CAMERA_INFO_MESSAGE_TYPE),
            (
                "/camera/image_raw",
                "/camera/frame_status",
                HEARTBEAT_MESSAGE_TYPE,
            ),
        ],
    )
    def test_video_readiness_uses_status_source_not_image_payload(
        self,
        image_topic: str,
        status_topic: str,
        message_type: str,
    ) -> None:
        recorder = MagicMock(spec=VideoRecorderConfig)
        recorder.topic_name = image_topic
        recorder.liveness_topic = status_topic
        recorder.liveness_message_type = message_type

        with patch(target=RESOLVER_PATH, return_value=int) as resolver:
            sources = build_liveness_sources(recorders=[recorder])

        assert sources == {
            image_topic: LivenessSource(topic_name=status_topic, message_type=int)
        }
        resolver.assert_called_once_with(dotted_path=message_type)

    @pytest.mark.unit
    @pytest.mark.parametrize("status_topic", [None, "", "  "])
    def test_video_without_explicit_status_topic_is_rejected(
        self, status_topic: str | None
    ) -> None:
        image_topic = "/camera/image_raw"
        recorder = MagicMock(spec=VideoRecorderConfig)
        recorder.topic_name = image_topic
        recorder.liveness_topic = status_topic
        recorder.liveness_message_type = CAMERA_INFO_MESSAGE_TYPE

        with patch(target=RESOLVER_PATH) as resolver:
            with pytest.raises(
                ValueError,
                match=re.escape(
                    f"Video recorder '{image_topic}' requires an explicit "
                    "liveness_topic for the recording UI."
                ),
            ):
                build_liveness_sources(recorders=[recorder])

        resolver.assert_not_called()

    @pytest.mark.unit
    def test_non_video_source_preserves_its_actual_message_type(self) -> None:
        recorder = MagicMock(spec=TopicRecorderConfig)
        recorder.topic_name = "/state"
        recorder.message_type = STATE_MESSAGE_TYPE

        with patch(target=RESOLVER_PATH, return_value=str) as resolver:
            sources = build_liveness_sources(recorders=[recorder])

        assert sources == {
            "/state": LivenessSource(topic_name="/state", message_type=str)
        }
        resolver.assert_called_once_with(dotted_path=STATE_MESSAGE_TYPE)

    @pytest.mark.unit
    def test_empty_catalog_does_not_resolve_any_messages(self) -> None:
        with patch(target=RESOLVER_PATH) as resolver:
            sources = build_liveness_sources(recorders=[])

        assert sources == {}
        resolver.assert_not_called()

    @pytest.mark.unit
    def test_unsupported_recorder_is_rejected(self) -> None:
        recorder = RecorderConfig(topic_name="/unsupported", file_name="unsupported")

        with pytest.raises(
            TypeError,
            match=re.escape("Unsupported recorder config type: RecorderConfig"),
        ):
            build_liveness_sources(recorders=[recorder])
