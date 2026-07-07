"""Dataset writer interface for exporting episodes to storage formats."""

from __future__ import annotations

from typing import Protocol

from tso_sensorium.episodes.schema import DatasetSchema, Episode


class DatasetWriter(Protocol):
    """Writes aligned episodes into a concrete dataset format.

    Writers are opened once with the dataset schema, receive episodes one
    at a time, and finalize the dataset after the last episode. New export
    formats implement this protocol; format-specific dependencies stay in
    the writer's own module.
    """

    def open(self, schema: DatasetSchema) -> None:
        """Prepare the output dataset for the given schema."""
        ...

    def add_episode(self, episode: Episode) -> None:
        """Append one aligned episode to the dataset."""
        ...

    def write_metadata(self, metadata: dict) -> None:
        """Store dataset-level metadata (name, task, phase legend)."""
        ...

    def finalize(self) -> None:
        """Complete the dataset after the last episode."""
        ...
