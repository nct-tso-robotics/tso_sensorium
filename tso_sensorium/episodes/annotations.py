"""Phase and language annotations layered over recorded episodes.

Annotations live in a sidecar JSON file inside each episode folder, so
raw recordings stay immutable and datasets can be regenerated from
recordings plus annotations at any time. Automatic and manual segments
coexist: re-running a labeler replaces only the automatic ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Union

ANNOTATIONS_FILE_NAME = "annotations.json"
AUTO_SOURCE = "auto"
MANUAL_SOURCE = "manual"
SEGMENT_SOURCES = (AUTO_SOURCE, MANUAL_SOURCE)


@dataclass(frozen=True)
class PhaseSegment:
    """One labeled span of an episode timeline.

    Args:
        start: Segment start, in the episode's timestamp units, inclusive.
        end: Segment end, exclusive.
        phase: Integer phase label from the dataset's legend.
        source: Who produced the label, "auto" or "manual".
        language: Free-form instruction overriding the legend mapping.
    """

    start: int
    end: int
    phase: int
    source: str = AUTO_SOURCE
    language: Optional[str] = None

    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError(
                f"Segment end ({self.end}) must be after start ({self.start})"
            )
        if self.source not in SEGMENT_SOURCES:
            raise ValueError(
                f"source must be one of {SEGMENT_SOURCES}, got '{self.source}'"
            )

    def contains(self, timestamp: Union[int, float]) -> bool:
        """Whether the timestamp falls inside this segment."""
        return self.start <= timestamp < self.end


@dataclass
class EpisodeAnnotations:
    """All annotation segments of one episode, sorted by start time.

    Args:
        segments: Labeled spans; may overlap only through operator error,
            in which case the earliest matching segment wins.
    """

    segments: List[PhaseSegment] = field(default_factory=list)

    @classmethod
    def load(cls, path: Union[Path, str]) -> "EpisodeAnnotations":
        """Load annotations from a JSON file; missing files are empty.

        Args:
            path: Annotations file inside an episode folder.

        Returns:
            The stored annotations, or empty annotations when the file
            does not exist.
        """
        annotations_path = Path(path)
        if not annotations_path.is_file():
            return cls()
        payload = json.loads(annotations_path.read_text())
        segments = [
            PhaseSegment(
                start=entry["start"],
                end=entry["end"],
                phase=int(entry["phase"]),
                source=entry.get("source", AUTO_SOURCE),
                language=entry.get("language"),
            )
            for entry in payload.get("segments", [])
        ]
        return cls(segments=sorted(segments, key=lambda segment: segment.start))

    def save(self, path: Union[Path, str]) -> None:
        """Write the annotations to a JSON file.

        Args:
            path: Annotations file inside an episode folder.
        """
        payload = {
            "segments": [
                asdict(segment)
                for segment in sorted(self.segments, key=lambda segment: segment.start)
            ]
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    def segment_at(self, timestamp: Union[int, float]) -> Optional[PhaseSegment]:
        """Return the segment covering a timestamp, if any."""
        for segment in self.segments:
            if segment.contains(timestamp=timestamp):
                return segment
        return None

    def replace_auto_segments(
        self, segments: List[PhaseSegment]
    ) -> "EpisodeAnnotations":
        """Replace automatic segments, keeping the manual ones.

        Args:
            segments: New automatically produced segments.

        Returns:
            Self, for method chaining.
        """
        manual = [
            segment for segment in self.segments if segment.source == MANUAL_SOURCE
        ]
        self.segments = sorted(
            manual + list(segments), key=lambda segment: segment.start
        )
        return self
