"""Dataset-level metadata: identity, phase legend, and coordinate frames.

The metadata lives in ``dataset_metadata.json`` at the recordings root,
is edited from the dashboard or seeded from generation configs, and is
written into every generated dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union

from pydantic import Field

from tso_sensorium.configuration import ConfigModel
from tso_sensorium.episodes.schema import CoordinateFrameFeatureMetadata

DATASET_METADATA_FILE_NAME = "dataset_metadata.json"


class PhaseDefinition(ConfigModel):
    """One entry of the phase legend.

    Args:
        name: Human-readable phase name.
        instructions: Language instruction variants; one is sampled per
            episode when there are several.
    """

    name: str = ""
    instructions: List[str] = Field(default_factory=list)


class DatasetMetadata(ConfigModel):
    """Global information describing a dataset.

    Args:
        dataset_name: Name of the dataset.
        task: Overall task description.
        phase_legend: Mapping of integer phase label to its definition.
        coordinate_frame_features: Vector component groups and their coordinate-frame
            temporality.
    """

    dataset_name: str = ""
    task: str = ""
    phase_legend: Dict[int, PhaseDefinition] = Field(default_factory=dict)
    coordinate_frame_features: Dict[str, CoordinateFrameFeatureMetadata] = Field(
        default_factory=dict
    )

    @classmethod
    def load(cls, path: Union[Path, str]) -> "DatasetMetadata":
        """Load metadata from a JSON file; missing files are empty.

        Args:
            path: Metadata file, typically at the recordings root.

        Returns:
            The stored metadata, or empty metadata when the file does not
            exist.
        """
        metadata_path = Path(path)
        if not metadata_path.is_file():
            return cls()
        payload = json.loads(metadata_path.read_text())
        return cls(
            dataset_name=payload.get("dataset_name", ""),
            task=payload.get("task", ""),
            phase_legend={
                int(label): PhaseDefinition(
                    name=entry.get("name", ""),
                    instructions=list(entry.get("instructions", [])),
                )
                for label, entry in payload.get("phase_legend", {}).items()
            },
            coordinate_frame_features={
                name: CoordinateFrameFeatureMetadata.model_validate(feature)
                for name, feature in payload.get(
                    "coordinate_frame_features", {}
                ).items()
            },
        )

    def save(self, path: Union[Path, str]) -> None:
        """Write the metadata to a JSON file.

        Args:
            path: Metadata file, typically at the recordings root.
        """
        Path(path).write_text(json.dumps(self.to_payload(), indent=2))

    def to_payload(self) -> dict:
        """Serialize for the HTTP API and dataset metadata files."""
        return {
            "dataset_name": self.dataset_name,
            "task": self.task,
            "phase_legend": {
                str(label): {
                    "name": definition.name,
                    "instructions": definition.instructions,
                }
                for label, definition in sorted(self.phase_legend.items())
            },
            "coordinate_frame_features": {
                name: feature.model_dump(mode="json")
                for name, feature in sorted(self.coordinate_frame_features.items())
            },
        }
