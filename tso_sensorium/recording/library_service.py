"""Episode library: browsing, playback, annotation, and generation.

Framework-free service behind the dashboard's Library section. It has no
ROS dependency, so the same routes power both the robot-side recording
dashboard and the standalone annotation app on any machine.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Optional

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from tso_sensorium.episodes.annotations import (
    ANNOTATIONS_FILE_NAME,
    EpisodeAnnotations,
    PhaseSegment,
)
from tso_sensorium.episodes.generation import generate_dataset
from tso_sensorium.episodes.generation_config import (
    DatasetGenerationConfig,
    apply_generation_overrides,
)
from tso_sensorium.episodes.legend import (
    DATASET_METADATA_FILE_NAME,
    DatasetMetadata,
    PhaseDefinition,
)
from tso_sensorium.recording.episode_files import list_episodes
from tso_sensorium.recording.playback import ensure_playback_copy
from tso_sensorium.resources import ASSETS_DIR

DASHBOARD_ASSET = "recording_dashboard.html"


class LibraryService:
    """Manages recorded episodes, annotations, and dataset generation.

    Args:
        recordings_root: Directory containing one folder per episode.
        generation: Dataset generation offered in the dashboard; its
            ``recordings_root`` defaults to the current root.
    """

    def __init__(
        self,
        recordings_root: Path | str,
        generation: Optional[DatasetGenerationConfig] = None,
    ):
        self.recordings_root = Path(recordings_root)
        self.generation = generation
        self._generation_thread: Optional[threading.Thread] = None
        self._generation_status: dict = {"state": "idle"}
        self._lock = threading.Lock()

    def set_root(self, recordings_root: Path | str) -> None:
        """Point the library at a different recordings folder."""
        root = Path(recordings_root)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {recordings_root}")
        self.recordings_root = root

    def annotations_path(self, episode_name: str) -> Path:
        """Annotations file of one episode."""
        annotations_file = ANNOTATIONS_FILE_NAME
        if self.generation is not None and self.generation.annotations is not None:
            annotations_file = self.generation.annotations.file_name
        return self.recordings_root / episode_name / annotations_file

    def metadata_path(self) -> Path:
        """Dataset metadata file at the recordings root."""
        metadata_file = DATASET_METADATA_FILE_NAME
        if self.generation is not None and self.generation.annotations is not None:
            metadata_file = self.generation.annotations.metadata_file
        return self.recordings_root / metadata_file

    def dataset_metadata(self) -> DatasetMetadata:
        """Load the dataset metadata, seeded from the generation config.

        The file at the recordings root wins; when absent, an inline
        legend from the generation config is offered as the starting
        point.
        """
        metadata = DatasetMetadata.load(path=self.metadata_path())
        if (
            not metadata.phase_legend
            and self.generation is not None
            and self.generation.annotations is not None
            and self.generation.annotations.legend is not None
        ):
            return self.generation.annotations.legend
        return metadata

    def generation_running(self) -> bool:
        """Whether a generation thread is currently active."""
        return (
            self._generation_thread is not None and self._generation_thread.is_alive()
        )

    def generation_status(self) -> dict:
        """Current generation state for the status endpoint."""
        return dict(self._generation_status)

    def start_generation(self, overrides: Optional[dict] = None) -> dict:
        """Run the configured dataset generation in the background.

        Args:
            overrides: Dot-path overrides applied onto the configured
                generation.

        Returns:
            The generation status at submission time.
        """
        if self.generation is None:
            raise ValueError("No dataset generation configured")
        with self._lock:
            if self.generation_running():
                raise RuntimeError("Dataset generation already running")
            generation_config = self.generation
            if overrides:
                generation_config = apply_generation_overrides(
                    config=generation_config, overrides=overrides
                )
            if not generation_config.recordings_root:
                generation_config = generation_config.model_copy(
                    update={"recordings_root": str(self.recordings_root)}
                )
            self._generation_status = {"state": "running"}
            self._generation_thread = threading.Thread(
                target=self._run_generation,
                kwargs={"generation_config": generation_config},
                daemon=True,
            )
            self._generation_thread.start()
        return dict(self._generation_status)

    def _run_generation(self, generation_config: DatasetGenerationConfig) -> None:
        # The thread would die silently otherwise; surface failures in the
        # status instead.
        try:
            report = generate_dataset(config=generation_config)
            self._generation_status = {
                "state": "done",
                "written": report.written,
                "failed": report.failed,
            }
        except (ValueError, OSError, RuntimeError, ImportError) as error:
            self._generation_status = {"state": "failed", "error": str(error)}

    def library_status(self) -> dict:
        """Library-level entries of the status payload."""
        disk_root = (
            self.recordings_root if self.recordings_root.is_dir() else Path.home()
        )
        disk = shutil.disk_usage(disk_root)
        return {
            "output_folder": str(self.recordings_root),
            "disk_free_gb": round(disk.free / 1024**3, 1),
            "generation": self.generation_status(),
            "generation_available": self.generation is not None,
        }


def register_library_routes(app: Flask, library: LibraryService) -> None:
    """Attach the library endpoints to a Flask app.

    Args:
        app: Application receiving the routes.
        library: Service backing them.
    """

    @app.get("/api/episodes")
    def episodes():
        listings = list_episodes(output_folder=library.recordings_root)
        return jsonify(
            [
                {
                    "name": listing.name,
                    "modified_timestamp": listing.modified_timestamp,
                    "files": [
                        {
                            "name": episode_file.name,
                            "size_megabytes": episode_file.size_megabytes,
                            "playable": episode_file.playable,
                        }
                        for episode_file in listing.files
                    ],
                }
                for listing in listings
            ]
        )

    @app.get("/episodes/<episode_name>/<file_name>")
    def episode_file(episode_name: str, file_name: str):
        return send_from_directory(
            library.recordings_root / episode_name, file_name, conditional=True
        )

    @app.get("/episodes/<episode_name>/<file_name>/playback")
    def episode_playback(episode_name: str, file_name: str):
        playback_copy = ensure_playback_copy(
            source=library.recordings_root / episode_name / file_name
        )
        return send_from_directory(
            playback_copy.parent, playback_copy.name, conditional=True
        )

    @app.get("/api/episodes/<episode_name>/annotations")
    def episode_annotations(episode_name: str):
        annotations = EpisodeAnnotations.load(
            path=library.annotations_path(episode_name=episode_name)
        )
        timeline = []
        generation = library.generation
        if generation is not None and generation.videos:
            timestamps_path = (
                library.recordings_root
                / episode_name
                / generation.videos[0].timestamps_file
            )
            if timestamps_path.is_file():
                timeline = (
                    pd.read_csv(timestamps_path)[generation.sync_column]
                    .astype(int)
                    .tolist()
                )
        return jsonify(
            {
                "segments": [
                    {
                        "start": segment.start,
                        "end": segment.end,
                        "phase": segment.phase,
                        "source": segment.source,
                        "language": segment.language,
                    }
                    for segment in annotations.segments
                ],
                "timeline": timeline,
                "legend": library.dataset_metadata().to_payload(),
            }
        )

    @app.put("/api/episodes/<episode_name>/annotations")
    def save_episode_annotations(episode_name: str):
        payload = request.get_json(silent=True) or {}
        segments = []
        for entry in payload.get("segments", []):
            for required_key in ("start", "end", "phase"):
                if required_key not in entry:
                    raise ValueError(f"Segment is missing '{required_key}'")
            segments.append(
                PhaseSegment(
                    start=int(entry["start"]),
                    end=int(entry["end"]),
                    phase=int(entry["phase"]),
                    source=entry.get("source", "manual"),
                    language=entry.get("language") or None,
                )
            )
        EpisodeAnnotations(segments=segments).save(
            path=library.annotations_path(episode_name=episode_name)
        )
        return jsonify({"saved": len(segments)})

    @app.get("/api/library/metadata")
    def library_metadata():
        return jsonify(library.dataset_metadata().to_payload())

    @app.put("/api/library/metadata")
    def save_library_metadata():
        payload = request.get_json(silent=True) or {}
        metadata = DatasetMetadata(
            dataset_name=str(payload.get("dataset_name", "")),
            task=str(payload.get("task", "")),
            phase_legend={
                int(label): PhaseDefinition(
                    name=str(entry.get("name", "")),
                    instructions=[
                        str(instruction)
                        for instruction in entry.get("instructions", [])
                        if str(instruction).strip()
                    ],
                )
                for label, entry in payload.get("phase_legend", {}).items()
            },
        )
        metadata.save(path=library.metadata_path())
        return jsonify(metadata.to_payload())

    @app.post("/api/library/root")
    def set_library_root():
        payload = request.get_json(silent=True) or {}
        library.set_root(recordings_root=str(payload.get("path", "")))
        return jsonify({"output_folder": str(library.recordings_root)})

    @app.post("/api/generation/start")
    def start_generation():
        payload = request.get_json(silent=True) or {}
        return jsonify(library.start_generation(overrides=payload.get("overrides")))


def register_error_handlers(app: Flask) -> None:
    """Map validation and state errors to JSON responses."""

    def handle_bad_request(error: ValueError):
        return jsonify({"error": str(error)}), 400

    def handle_conflict(error: RuntimeError):
        return jsonify({"error": str(error)}), 409

    app.register_error_handler(ValueError, handle_bad_request)
    app.register_error_handler(RuntimeError, handle_conflict)


def register_dashboard_route(app: Flask) -> None:
    """Serve the dashboard page at the root URL."""

    @app.get("/")
    def dashboard() -> Response:
        page = ASSETS_DIR.joinpath(DASHBOARD_ASSET).read_text()
        return Response(page, mimetype="text/html")


def create_library_app(library: LibraryService) -> Flask:
    """Build the standalone annotation and generation app (no ROS).

    Args:
        library: Service backing the app.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)
    register_error_handlers(app=app)
    register_dashboard_route(app=app)
    register_library_routes(app=app, library=library)

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "state": "idle",
                "recording_available": False,
                "episode_name": None,
                "elapsed_seconds": None,
                "camera_topic": "",
                "sensors": [],
                **library.library_status(),
            }
        )

    return app
