"""Episode naming utilities shared by recording adapters."""

from __future__ import annotations

import datetime
from typing import Optional

EPISODE_NAME_SEPARATOR = "_"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S_%f"


def generate_timestamp() -> str:
    """Generate the current time as an episode-name component."""
    return datetime.datetime.now().strftime(TIMESTAMP_FORMAT)


def build_episode_name(custom_name: Optional[str] = None) -> str:
    """Build an episode name ending in its creation timestamp.

    Args:
        custom_name: Optional descriptive prefix.

    Returns:
        Timestamp alone or the custom prefix followed by the timestamp.
    """
    timestamp = generate_timestamp()
    if not custom_name:
        return timestamp
    return EPISODE_NAME_SEPARATOR.join((custom_name, timestamp))
