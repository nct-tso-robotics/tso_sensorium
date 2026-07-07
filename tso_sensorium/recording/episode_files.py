"""Listing of recorded episode folders for browsing and replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

BYTES_PER_MEGABYTE = 1024 * 1024
PLAYABLE_EXTENSIONS = (".mp4", ".avi")


@dataclass(frozen=True)
class EpisodeFile:
    """One file inside a recorded episode folder.

    Args:
        name: File name relative to the episode folder.
        size_megabytes: File size in megabytes.
        playable: Whether the playback endpoint can serve the file as
            browser-compatible video.
    """

    name: str
    size_megabytes: float
    playable: bool


@dataclass(frozen=True)
class EpisodeListing:
    """One recorded episode folder.

    Args:
        name: Episode folder name.
        modified_timestamp: Last modification time, seconds since epoch.
        files: Files directly inside the folder, sorted by name.
    """

    name: str
    modified_timestamp: float
    files: list[EpisodeFile] = field(default_factory=list)


def list_episodes(output_folder: Path | str) -> list[EpisodeListing]:
    """List recorded episode folders, newest first.

    Args:
        output_folder: Folder containing one subdirectory per episode.

    Returns:
        Episode listings with their top-level files.
    """
    root = Path(output_folder)
    if not root.is_dir():
        return []
    listings = []
    for episode_directory in root.iterdir():
        if not episode_directory.is_dir():
            continue
        files = [
            EpisodeFile(
                name=file_path.name,
                size_megabytes=round(file_path.stat().st_size / BYTES_PER_MEGABYTE, 2),
                playable=file_path.suffix in PLAYABLE_EXTENSIONS,
            )
            for file_path in sorted(episode_directory.iterdir())
            if file_path.is_file()
        ]
        listings.append(
            EpisodeListing(
                name=episode_directory.name,
                modified_timestamp=episode_directory.stat().st_mtime,
                files=files,
            )
        )
    return sorted(
        listings, key=lambda listing: listing.modified_timestamp, reverse=True
    )
