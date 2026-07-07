"""Field extraction from ROS messages by attribute path."""

from __future__ import annotations

import importlib
from operator import attrgetter
from typing import Any, Sequence

FIELD_SEPARATOR = "."


def resolve_message_type(dotted_path: str) -> type:
    """Import a message class from its dotted path.

    Args:
        dotted_path: Full path, e.g. "testbed_msgs.msg.RobotState".

    Returns:
        The message class.
    """
    module_path, _, class_name = dotted_path.rpartition(FIELD_SEPARATOR)
    if not module_path:
        raise ValueError(
            f"Message type must be a dotted path like"
            f" 'std_msgs.msg.String', got '{dotted_path}'"
        )
    module = importlib.import_module(module_path)
    if not hasattr(module, class_name):
        raise AttributeError(f"Module {module_path} has no message class {class_name}")
    return getattr(module, class_name)


class MessageFieldExtractor:
    """Extracts message attributes as a row of values.

    Nested attributes use dotted paths, e.g. "transform.rotation.x".

    Args:
        fields: Attribute paths extracted from each message, in order.
    """

    def __init__(self, fields: Sequence[str]):
        if not fields:
            raise ValueError("MessageFieldExtractor requires at least one field")
        self.fields = list(fields)
        self._getters = [attrgetter(field) for field in self.fields]

    @property
    def csv_header(self) -> list[str]:
        """Column names derived from the field paths."""
        return [field.replace(FIELD_SEPARATOR, "_") for field in self.fields]

    def __call__(self, message: Any) -> list:
        """Extract the configured fields from one message."""
        return [getter(message) for getter in self._getters]
