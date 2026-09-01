"""Episode library: browsing, playback, annotation, and generation.

Framework-free service behind the dashboard's Library section. It has no
ROS dependency, so the same routes power both the robot-side recording
dashboard and the standalone annotation app on any machine.
"""

from __future__ import annotations

import shutil
import threading
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Callable, Generator, Optional

import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory

from tso_sensorium.configuration import OverrideValue
from tso_sensorium.episodes.annotations import (
    ANNOTATIONS_FILE_NAME,
    EpisodeAnnotations,
    PhaseSegment,
)
from tso_sensorium.episodes.dataset_builder import (
    BuildCancellationToken,
    BuildPhase,
    BuildProgress,
)
from tso_sensorium.episodes.generation import generate_dataset
from tso_sensorium.episodes.generation_config import (
    DatasetGenerationConfig,
    LeRobotActionUpdateWriterConfig,
    LeRobotWriterConfig,
    apply_generation_overrides,
)
from tso_sensorium.episodes.legend import (
    DATASET_METADATA_FILE_NAME,
    DatasetMetadata,
    PhaseDefinition,
)
from tso_sensorium.recording.denoising_preview import (
    DenoisingPreviewData,
    configured_denoising_groups,
    load_denoising_preview_data,
)
from tso_sensorium.recording.episode_files import list_episodes
from tso_sensorium.recording.playback import ensure_playback_copy
from tso_sensorium.resources import ASSETS_DIR

DASHBOARD_ASSET = "recording_dashboard.html"


class GenerationMode(str, Enum):
    """Dataset operation exposed by the recording dashboard."""

    FULL = "full"
    ACTIONS_ONLY = "actions_only"


class GenerationState(str, Enum):
    """Lifecycle state of one dashboard generation request."""

    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


def _paths_overlap(first: Path, second: Path) -> bool:
    """Return whether either path contains the other."""
    return first == second or first in second.parents or second in first.parents


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
        self._generation_cancellation: Optional[BuildCancellationToken] = None
        self._generation_sequence = 0
        self._generation_status: dict = {"state": GenerationState.IDLE.value}
        self._denoising_preview_data: Optional[DenoisingPreviewData] = None
        self._denoising_preview_lock = threading.Lock()
        self._lock = threading.Lock()

    def set_root(self, recordings_root: Path | str) -> None:
        """Point the library at a different recordings folder."""
        root = Path(recordings_root)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {recordings_root}")
        with self.library_mutation():
            self.recordings_root = root
            self.invalidate_denoising_preview()

    @contextmanager
    def library_mutation(self) -> Generator[None, None, None]:
        """Guard a recordings or annotation mutation against active work.

        Yields:
            Control while the library lifecycle lock is held.

        Raises:
            RuntimeError: If generation or denoising preview is active.
        """
        with self._lock:
            if self.generation_running():
                raise RuntimeError("Cannot modify recordings during generation")
            if self._denoising_preview_lock.locked():
                raise RuntimeError("Cannot modify recordings during denoising preview")
            yield

    def invalidate_denoising_preview(self) -> None:
        """Discard action magnitudes cached for the current recordings root."""
        self._denoising_preview_data = None

    def denoising_preview_available(self) -> bool:
        """Whether the active generation config includes percentile denoising."""
        return self.generation is not None and bool(
            configured_denoising_groups(config=self.generation)
        )

    def denoising_preview(
        self,
        percentiles: Optional[dict[str, float]] = None,
        refresh: bool = False,
    ) -> dict:
        """Preview configured denoising thresholds without writing a dataset.

        Args:
            percentiles: Selected values keyed by generation override path.
            refresh: Whether to reassemble episodes instead of using cached
                magnitudes.

        Returns:
            Magnitude distributions, thresholds, and zeroing counts.

        Raises:
            ValueError: If no percentile denoising transform is configured.
        """
        if self.generation is None or not self.denoising_preview_available():
            raise ValueError("No percentile denoising transform configured")
        with self._denoising_preview_lock:
            with self._lock:
                if self.generation_running():
                    raise RuntimeError(
                        "Denoising preview is unavailable during generation"
                    )
            if refresh:
                self.invalidate_denoising_preview()
            if self._denoising_preview_data is None:
                self._denoising_preview_data = load_denoising_preview_data(
                    config=self.generation,
                    recordings_root=self.recordings_root,
                )
            phase_names = {
                label: definition.name
                for label, definition in self.dataset_metadata().phase_legend.items()
            }
            return self._denoising_preview_data.to_payload(
                percentiles=percentiles,
                phase_names=phase_names,
            )

    def episode_directory(self, episode_name: str) -> Path:
        """Resolve an episode folder, rejecting names that escape the root.

        Args:
            episode_name: Episode folder name from a request.

        Returns:
            The resolved episode directory, always a direct child of the
            recordings root.
        """
        root = self.recordings_root.resolve()
        candidate = (root / episode_name).resolve()
        if candidate.parent != root:
            raise ValueError(f"Invalid episode name: {episode_name}")
        return candidate

    def annotations_path(self, episode_name: str) -> Path:
        """Annotations file of one episode."""
        annotations_file = ANNOTATIONS_FILE_NAME
        if self.generation is not None and self.generation.annotations is not None:
            annotations_file = self.generation.annotations.file_name
        return self.episode_directory(episode_name=episode_name) / annotations_file

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
        if self.generation is not None and self.generation.annotations is not None:
            return self.generation.annotations.resolve_legend(
                recordings_root=self.recordings_root
            )
        return DatasetMetadata.load(path=self.metadata_path())

    def generation_running(self) -> bool:
        """Whether a generation thread is currently active."""
        return (
            self._generation_thread is not None and self._generation_thread.is_alive()
        )

    def generation_status(self) -> dict:
        """Current generation state for the status endpoint."""
        with self._lock:
            status = dict(self._generation_status)
            if "failed" in status:
                status["failed"] = dict(status["failed"])
            if "written" in status:
                status["written"] = list(status["written"])
            return status

    def _prepare_generation_config(
        self,
        mode: GenerationMode,
        overrides: Optional[dict[str, OverrideValue]],
        dataset_root: Optional[Path | str],
    ) -> DatasetGenerationConfig:
        """Resolve one UI request into a fully validated generation config."""
        if self.generation is None:
            raise ValueError("No dataset generation configured")
        generation_config = self.generation
        if overrides:
            generation_config = apply_generation_overrides(
                config=generation_config,
                overrides=overrides,
            )
        encoded_config = generation_config.model_dump(by_alias=True)
        encoded_config["recordings_root"] = str(self.recordings_root)
        if mode == GenerationMode.ACTIONS_ONLY:
            if dataset_root is None or not str(dataset_root).strip():
                raise ValueError("dataset_root is required for an actions-only update")
            expanded_dataset_root = Path(dataset_root).expanduser()
            if not expanded_dataset_root.is_dir() or expanded_dataset_root.is_symlink():
                raise ValueError(
                    "Actions-only update requires an existing LeRobot dataset"
                    f" root: {expanded_dataset_root}"
                )
            canonical_dataset_root = expanded_dataset_root.resolve()
            canonical_recordings_root = self.recordings_root.expanduser().resolve()
            if _paths_overlap(
                first=canonical_dataset_root,
                second=canonical_recordings_root,
            ):
                raise ValueError(
                    "Actions-only LeRobot dataset root must not overlap the"
                    f" recordings root: {canonical_dataset_root} and"
                    f" {canonical_recordings_root}"
                )
            absolute_dataset_root = expanded_dataset_root.absolute()
            encoded_config["save_frames"] = False
            encoded_config["writer"] = LeRobotActionUpdateWriterConfig(
                dataset_root=str(absolute_dataset_root),
            ).model_dump()
        elif dataset_root is not None:
            raise ValueError("dataset_root is only valid for an actions-only update")
        validated_config = DatasetGenerationConfig.model_validate(encoded_config)
        if mode == GenerationMode.FULL and isinstance(
            validated_config.writer, LeRobotActionUpdateWriterConfig
        ):
            raise ValueError(
                "LeRobot action updates require mode='actions_only' and dataset_root"
            )
        if mode == GenerationMode.FULL and isinstance(
            validated_config.writer, LeRobotWriterConfig
        ):
            if not validated_config.writer.output_root:
                raise ValueError("Full LeRobot generation requires a new output root")
            output_root = Path(validated_config.writer.output_root).expanduser()
            canonical_output_root = output_root.resolve()
            canonical_recordings_root = self.recordings_root.expanduser().resolve()
            if _paths_overlap(
                first=canonical_output_root,
                second=canonical_recordings_root,
            ):
                raise ValueError(
                    "Full LeRobot output root must not overlap the recordings"
                    f" root: {canonical_output_root} and"
                    f" {canonical_recordings_root}"
                )
            if output_root.exists() or output_root.is_symlink():
                raise ValueError(
                    f"Full LeRobot generation requires a new output root: {output_root}"
                )
        return validated_config

    def start_generation(
        self,
        mode: GenerationMode = GenerationMode.FULL,
        overrides: Optional[dict[str, OverrideValue]] = None,
        dataset_root: Optional[Path | str] = None,
    ) -> dict:
        """Run the configured dataset generation in the background.

        Args:
            mode: Whether to build a full dataset or atomically update only
                actions in an existing LeRobot dataset.
            overrides: Dot-path overrides applied onto the configured
                generation.
            dataset_root: Existing LeRobot v3 root required by actions-only
                mode.

        Returns:
            The generation status at submission time.

        Raises:
            RuntimeError: If another generation is active.
            ValueError: If the generation request is incomplete or invalid.
        """
        with self._lock:
            if self.generation_running():
                raise RuntimeError("Dataset generation already running")
            if self._denoising_preview_lock.locked():
                raise RuntimeError(
                    "Dataset generation cannot start during denoising preview"
                )
            generation_config = self._prepare_generation_config(
                mode=mode,
                overrides=overrides,
                dataset_root=dataset_root,
            )
            cancellation = BuildCancellationToken()
            self._generation_cancellation = cancellation
            self._generation_sequence += 1
            self._generation_status = {
                "state": GenerationState.RUNNING.value,
                "run_id": self._generation_sequence,
                "mode": mode.value,
                "writer_type": generation_config.writer.type,
                "phase": BuildPhase.DISCOVERING.value,
                "completed": 0,
                "total": 0,
                "percentage": 0.0,
                "current_episode": None,
                "failed": {},
                "written": [],
            }
            submission_status = dict(self._generation_status)
            self._generation_thread = threading.Thread(
                target=self._run_generation,
                kwargs={
                    "generation_config": generation_config,
                    "cancellation": cancellation,
                },
                daemon=True,
            )
            self._generation_thread.start()
        return submission_status

    def cancel_generation(self) -> dict:
        """Request cooperative cancellation of the active generation.

        Returns:
            Updated generation status showing cancellation in progress.

        Raises:
            RuntimeError: If no generation is active.
        """
        with self._lock:
            if not self.generation_running() or self._generation_cancellation is None:
                raise RuntimeError("No dataset generation is running")
            if self._generation_status.get("state") == GenerationState.CANCELLING.value:
                return dict(self._generation_status)
            if self._generation_status.get("phase") in {
                BuildPhase.FINALIZING.value,
                BuildPhase.COMPLETED.value,
            }:
                raise RuntimeError(
                    "Dataset generation is finalizing and can no longer be cancelled"
                )
            self._generation_cancellation.cancel()
            self._generation_status["state"] = GenerationState.CANCELLING.value
            return dict(self._generation_status)

    def _record_generation_progress(self, progress: BuildProgress) -> None:
        """Store a coordinator progress snapshot for polling clients."""
        percentage = (
            min(100.0, progress.completed / progress.total * 100.0)
            if progress.total > 0
            else 0.0
        )
        with self._lock:
            state = self._generation_status.get("state", GenerationState.RUNNING.value)
            if state != GenerationState.CANCELLING.value:
                state = GenerationState.RUNNING.value
            self._generation_status.update(
                {
                    "state": state,
                    "phase": progress.phase.value,
                    "completed": progress.completed,
                    "total": progress.total,
                    "percentage": percentage,
                    "current_episode": progress.current_episode,
                    "failed": dict(progress.failed),
                }
            )

    def _run_generation(
        self,
        generation_config: DatasetGenerationConfig,
        cancellation: BuildCancellationToken,
    ) -> None:
        """Run generation and convert its outcome into persistent UI state."""
        try:
            report = generate_dataset(
                config=generation_config,
                progress_callback=self._record_generation_progress,
                cancellation_token=cancellation,
            )
        except Exception as error:
            with self._lock:
                self._generation_status.update(
                    {
                        "state": GenerationState.FAILED.value,
                        "error": str(error),
                    }
                )
                self._generation_cancellation = None
            return
        with self._lock:
            self._generation_status.update(
                {
                    "state": (
                        GenerationState.CANCELLED.value
                        if report.cancelled
                        else GenerationState.DONE.value
                    ),
                    "written": list(report.written),
                    "failed": dict(report.failed),
                }
            )
            self._generation_cancellation = None

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
            "denoising_preview_available": self.denoising_preview_available(),
        }

    def delete_episode(self, episode_name: str) -> dict[str, str]:
        """Delete one recorded episode and all files below it.

        Args:
            episode_name: Direct child directory of the recordings root.

        Returns:
            The deleted episode name.
        """
        with self.library_mutation():
            episode_directory = self.episode_directory(episode_name=episode_name)
            if not episode_directory.is_dir():
                raise ValueError(f"Episode not found: {episode_name}")
            shutil.rmtree(episode_directory)
            self.invalidate_denoising_preview()
        return {"deleted": episode_name}


def register_library_routes(
    app: Flask,
    library: LibraryService,
    episode_deleter: Optional[Callable[[str], dict[str, str]]] = None,
) -> None:
    """Attach the library endpoints to a Flask app.

    Args:
        app: Application receiving the routes.
        library: Service backing them.
        episode_deleter: Optional recording-aware deletion operation.
    """
    delete_episode = episode_deleter or library.delete_episode

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

    @app.delete("/api/episodes/<episode_name>")
    def remove_episode(episode_name: str):
        return jsonify(delete_episode(episode_name))

    @app.get("/episodes/<episode_name>/<file_name>")
    def episode_file(episode_name: str, file_name: str):
        return send_from_directory(
            library.episode_directory(episode_name=episode_name),
            file_name,
            conditional=True,
        )

    @app.get("/episodes/<episode_name>/<file_name>/playback")
    def episode_playback(episode_name: str, file_name: str):
        playback_copy = ensure_playback_copy(
            source=library.episode_directory(episode_name=episode_name) / file_name
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
                library.episode_directory(episode_name=episode_name)
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
        with library.library_mutation():
            EpisodeAnnotations(segments=segments).save(
                path=library.annotations_path(episode_name=episode_name)
            )
            library.invalidate_denoising_preview()
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
        with library.library_mutation():
            metadata.save(path=library.metadata_path())
            library.invalidate_denoising_preview()
        return jsonify(metadata.to_payload())

    @app.get("/api/library/directories")
    def list_directories():
        requested = request.args.get("path") or str(library.recordings_root)
        root = Path(requested).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {requested}")
        directories = sorted(
            entry.name
            for entry in root.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )
        return jsonify(
            {
                "path": str(root),
                "parent": str(root.parent) if root.parent != root else None,
                "directories": directories,
            }
        )

    @app.post("/api/library/root")
    def set_library_root():
        payload = request.get_json(silent=True) or {}
        library.set_root(recordings_root=str(payload.get("path", "")))
        return jsonify({"output_folder": str(library.recordings_root)})

    @app.post("/api/generation/start")
    def start_generation():
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            raise ValueError("Generation request must be a mapping")
        raw_mode = payload.get("mode", GenerationMode.FULL.value)
        if not isinstance(raw_mode, str):
            raise ValueError("Generation mode must be a string")
        mode = GenerationMode(raw_mode)
        overrides = payload.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            raise ValueError("Generation overrides must be a mapping")
        dataset_root = payload.get("dataset_root")
        if dataset_root is not None and not isinstance(dataset_root, str):
            raise ValueError("Generation dataset_root must be a string")
        return jsonify(
            library.start_generation(
                mode=mode,
                overrides=overrides,
                dataset_root=dataset_root,
            )
        )

    @app.post("/api/generation/cancel")
    def cancel_generation():
        return jsonify(library.cancel_generation())

    @app.post("/api/generation/denoising-preview")
    def denoising_preview():
        payload = request.get_json(silent=True) or {}
        percentiles = payload.get("percentiles")
        if percentiles is not None and not isinstance(percentiles, dict):
            raise ValueError("Denoising preview percentiles must be a mapping")
        refresh = payload.get("refresh", False)
        if not isinstance(refresh, bool):
            raise ValueError("Denoising preview refresh must be a boolean")
        return jsonify(
            library.denoising_preview(
                percentiles=percentiles,
                refresh=refresh,
            )
        )


def register_error_handlers(app: Flask) -> None:
    """Map validation and state errors to JSON responses."""

    def handle_bad_request(error: ValueError | OSError):
        return jsonify({"error": str(error)}), 400

    def handle_conflict(error: RuntimeError):
        return jsonify({"error": str(error)}), 409

    app.register_error_handler(ValueError, handle_bad_request)
    app.register_error_handler(OSError, handle_bad_request)
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
