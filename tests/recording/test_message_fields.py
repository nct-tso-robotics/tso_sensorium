"""Tests for tso_sensorium.recording.message_fields module."""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from tso_sensorium.recording.message_fields import (
    MessageFieldExtractor,
    resolve_message_type,
)


class TestMessageFieldExtractor:
    @pytest.mark.unit
    def test_extracts_nested_fields_in_order(self):
        message = SimpleNamespace(
            open=True,
            transform=SimpleNamespace(rotation=SimpleNamespace(x=0.5, w=1.0)),
        )
        extractor = MessageFieldExtractor(
            fields=["transform.rotation.x", "transform.rotation.w", "open"]
        )
        assert extractor(message) == [0.5, 1.0, True]

    @pytest.mark.unit
    def test_header_replaces_dots_with_underscores(self):
        extractor = MessageFieldExtractor(fields=["transform.rotation.x", "open"])
        assert extractor.csv_header == ["transform_rotation_x", "open"]

    @pytest.mark.unit
    def test_rejects_empty_fields(self):
        with pytest.raises(
            ValueError,
            match=re.escape("MessageFieldExtractor requires at least one field"),
        ):
            MessageFieldExtractor(fields=[])


class TestResolveMessageType:
    @pytest.mark.unit
    def test_resolves_class_from_dotted_path(self):
        assert resolve_message_type(dotted_path="pathlib.Path") is Path

    @pytest.mark.unit
    def test_rejects_path_without_module(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Message type must be a dotted path like"
                " 'std_msgs.msg.String', got 'RobotState'"
            ),
        ):
            resolve_message_type(dotted_path="RobotState")

    @pytest.mark.unit
    def test_rejects_missing_class(self):
        with pytest.raises(
            AttributeError,
            match=re.escape("Module pathlib has no message class Missing"),
        ):
            resolve_message_type(dotted_path="pathlib.Missing")
