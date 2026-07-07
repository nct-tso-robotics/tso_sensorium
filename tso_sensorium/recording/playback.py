"""Browser-compatible playback copies of recorded videos.

Recorded videos use codecs browsers cannot decode (mp4v, HuffYUV), so
playback transcodes them to H.264 with ffmpeg, cached next to the source
in a ``.playback`` directory.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PLAYBACK_DIRECTORY = ".playback"
H264_ENCODERS = ["libx264", "libopenh264"]
VIDEO_EXTENSIONS = (".mp4", ".avi")
FFMPEG_BINARY = "ffmpeg"


def ensure_playback_copy(source: Path) -> Path:
    """Return a browser-playable H.264 copy of a recorded video.

    The copy is created on first use and reused until the source changes.

    Args:
        source: Recorded video file.

    Returns:
        Path to the cached playback copy.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Video not found: {source}")
    if source.suffix not in VIDEO_EXTENSIONS:
        raise ValueError(f"Not a video file: {source.name}")
    if shutil.which(FFMPEG_BINARY) is None:
        raise RuntimeError("ffmpeg is required for browser playback")
    cache = source.parent / PLAYBACK_DIRECTORY / f"{source.stem}.mp4"
    if cache.is_file() and cache.stat().st_mtime >= source.stat().st_mtime:
        return cache
    cache.parent.mkdir(exist_ok=True)
    errors = []
    for encoder in H264_ENCODERS:
        result = subprocess.run(
            [
                FFMPEG_BINARY,
                "-y",
                "-i",
                str(source),
                "-c:v",
                encoder,
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(cache),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return cache
        stderr_lines = result.stderr.strip().splitlines()
        detail = stderr_lines[-1] if stderr_lines else f"exit code {result.returncode}"
        errors.append(f"{encoder}: {detail}")
    raise RuntimeError(f"Transcoding failed ({'; '.join(errors)})")
