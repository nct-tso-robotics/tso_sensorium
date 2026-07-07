"""CSV dataset writer producing one folder per episode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from tso_sensorium.episodes.schema import DatasetSchema, Episode

EPISODE_FILE_NAME = "episode.csv"
METADATA_FILE_NAME = "dataset_metadata.json"


class CsvDatasetWriter:
    """Writes each episode to ``<output_root>/<episode_name>/episode.csv``.

    Matches the historical per-episode CSV layout, with camera frames
    referenced by path from the table.

    Args:
        output_root: Directory receiving one subdirectory per episode.
    """

    def __init__(self, output_root: Path | str):
        self.output_root = Path(output_root)
        self._schema: Optional[DatasetSchema] = None

    def open(self, schema: DatasetSchema) -> None:
        """Create the output root and store the schema for validation."""
        self._schema = schema
        self.output_root.mkdir(parents=True, exist_ok=True)

    def add_episode(self, episode: Episode) -> None:
        """Write one episode table to its own subdirectory."""
        if self._schema is None:
            raise RuntimeError(
                "Writer must be opened with a schema before adding episodes"
            )
        self._schema.validate_episode_table(table=episode.table)
        episode_folder = self.output_root / episode.name
        episode_folder.mkdir(parents=True, exist_ok=True)
        episode.table.to_csv(episode_folder / EPISODE_FILE_NAME, index=False)

    def write_metadata(self, metadata: dict) -> None:
        """Write the dataset metadata JSON at the output root."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / METADATA_FILE_NAME).write_text(
            json.dumps(metadata, indent=2)
        )

    def finalize(self) -> None:
        """Nothing to finalize; episodes are self-contained CSV files."""
