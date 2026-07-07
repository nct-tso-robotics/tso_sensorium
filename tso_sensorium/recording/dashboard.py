"""ROS-agnostic recording service and dashboard app.

The recording state machine and the Flask app are shared between the
ROS 1 and ROS 2 adapters: each adapter builds the service with its own
liveness monitor, camera feed, and episode session factory.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

from tso_sensorium.processing.stereo_view import StereoViewProcessor

from tso_sensorium.recording.config import RecordingServiceConfig
from tso_sensorium.recording.library_service import (
    LibraryService,
    register_dashboard_route,
    register_error_handlers,
    register_library_routes,
)

IDLE_STATE = "idle"
RECORDING_STATE = "recording"
MJPEG_BOUNDARY = "frame"
MJPEG_FRAME_INTERVAL_SECONDS = 0.05
MJPEG_KEEPALIVE_SECONDS = 2.0


class LivenessMonitor(Protocol):
    """Tracks when each topic last published."""

    def snapshot(self) -> dict[str, Optional[float]]: ...

    def is_alive(self, topic_name: str) -> bool: ...

    def close(self) -> None: ...


class LatestFrameFeed(Protocol):
    """Keeps the latest camera frame as JPEG bytes."""

    @property
    def latest_jpeg(self) -> Optional[bytes]: ...

    def close(self) -> None: ...


class RecordingSession(Protocol):
    """One recorded episode, from subscription to closed files."""

    episode_name: str

    def start(self) -> None: ...

    def close(self) -> None: ...


SessionFactory = Callable[[Optional[str], Optional[list[str]]], RecordingSession]


class RecordingService:
    """State machine behind the dashboard: idle or recording one episode.

    Args:
        config: Service configuration.
        liveness: Per-topic liveness monitor.
        camera_feed: Live camera capture; ``None`` disables the feed.
        session_factory: Builds one episode session from an episode name
            and an optional recorder subset.
    """

    def __init__(
        self,
        config: RecordingServiceConfig,
        liveness: LivenessMonitor,
        camera_feed: Optional[LatestFrameFeed],
        session_factory: SessionFactory,
    ):
        self.config = config
        self.liveness = liveness
        self.camera_feed = camera_feed
        self.stereo_view = (
            StereoViewProcessor(
                mode=config.stereo.mode,
                left_odd=config.stereo.left_odd,
                calibration_path=config.stereo.calibration_path,
            )
            if config.stereo is not None and camera_feed is not None
            else None
        )
        self.library = LibraryService(
            recordings_root=config.session.output_folder,
            generation=config.generation,
        )
        self._session_factory = session_factory
        self._session: Optional[RecordingSession] = None
        self._session_started_at: Optional[float] = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        """Current state, sensor liveness, and library information."""
        ages = self.liveness.snapshot()
        sensors = [
            {
                "name": recorder.file_name,
                "topic": recorder.topic_name,
                "alive": self.liveness.is_alive(topic_name=recorder.topic_name),
                "last_message_age": ages.get(recorder.topic_name),
            }
            for recorder in self.config.session.recorders
        ]
        with self._lock:
            recording = self._session is not None
            episode_name = self._session.episode_name if recording else None
            elapsed = (
                round(time.monotonic() - self._session_started_at, 1)
                if recording
                else None
            )
        return {
            "state": RECORDING_STATE if recording else IDLE_STATE,
            "recording_available": True,
            "stereo_available": self.stereo_view is not None,
            "episode_name": episode_name,
            "elapsed_seconds": elapsed,
            "camera_topic": self.config.camera_topic,
            "sensors": sensors,
            **self.library.library_status(),
        }

    def start_recording(
        self,
        episode_name: Optional[str] = None,
        recorder_names: Optional[list[str]] = None,
    ) -> dict:
        """Start recording one episode.

        Args:
            episode_name: Episode name; defaults to a time-based string.
            recorder_names: Subset of recorders to use; all when ``None``.

        Returns:
            The started episode's name.
        """
        with self._lock:
            if self._session is not None:
                raise RuntimeError("Already recording")
            if self.library.generation_running():
                raise RuntimeError("Dataset generation is running")
            session = self._session_factory(episode_name, recorder_names)
            session.start()
            self._session = session
            self._session_started_at = time.monotonic()
            return {"episode_name": session.episode_name}

    def set_output_folder(self, path: str) -> dict:
        """Point new episodes and the library at a different folder.

        Args:
            path: Existing directory receiving future episodes.

        Returns:
            The new output folder.
        """
        with self._lock:
            if self._session is not None:
                raise RuntimeError("Cannot change the output folder while recording")
        root = Path(path).expanduser()
        if not root.is_dir():
            raise ValueError(f"Not a directory: {path}")
        self.config.session.output_folder = str(root)
        self.library.set_root(recordings_root=root)
        return {"output_folder": str(root)}

    def stop_recording(self) -> dict:
        """Stop the running episode and finalize its files."""
        with self._lock:
            if self._session is None:
                raise RuntimeError("Not recording")
            session = self._session
            self._session = None
            self._session_started_at = None
        session.close()
        return {"episode_name": session.episode_name}

    def close(self) -> None:
        """Stop any running episode and release all subscriptions."""
        with self._lock:
            session = self._session
            self._session = None
            self._session_started_at = None
        if session is not None:
            session.close()
        self.liveness.close()
        if self.camera_feed is not None:
            self.camera_feed.close()


def create_app(service: RecordingService) -> Flask:
    """Build the Flask app exposing the dashboard and its API.

    Args:
        service: Recording service the endpoints control.

    Returns:
        Configured Flask application.
    """
    app = Flask(__name__)
    register_error_handlers(app=app)
    register_dashboard_route(app=app)
    register_library_routes(app=app, library=service.library)

    @app.get("/api/status")
    def status():
        return jsonify(service.status())

    @app.post("/api/recording/start")
    def start_recording():
        payload = request.get_json(silent=True) or {}
        return jsonify(
            service.start_recording(
                episode_name=payload.get("episode_name") or None,
                recorder_names=payload.get("recorders"),
            )
        )

    @app.post("/api/recording/stop")
    def stop_recording():
        return jsonify(service.stop_recording())

    @app.post("/api/recording/output_folder")
    def set_output_folder():
        payload = request.get_json(silent=True) or {}
        return jsonify(service.set_output_folder(path=str(payload.get("path", ""))))

    def _mjpeg_response(frames: Iterator[bytes]) -> Response:
        return Response(
            frames,
            mimetype=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )

    def _mjpeg_part(jpeg: bytes) -> bytes:
        return (
            b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )

    @app.get("/stream/camera")
    def camera_stream():
        if service.camera_feed is None:
            return jsonify({"error": "No camera feed configured"}), 404

        def frames() -> Iterator[bytes]:
            keepalive = b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
            idle_seconds = 0.0
            while True:
                jpeg = service.camera_feed.latest_jpeg
                if jpeg is not None:
                    yield _mjpeg_part(jpeg=jpeg)
                    idle_seconds = 0.0
                else:
                    # Force a periodic socket write even with no frame, so a
                    # disconnected client is detected instead of the handler
                    # thread sleeping forever.
                    idle_seconds += MJPEG_FRAME_INTERVAL_SECONDS
                    if idle_seconds >= MJPEG_KEEPALIVE_SECONDS:
                        yield keepalive
                        idle_seconds = 0.0
                time.sleep(MJPEG_FRAME_INTERVAL_SECONDS)

        return _mjpeg_response(frames=frames())

    def _stereo_stream(compose_view) -> Response:
        jpeg_quality = service.config.stereo.jpeg_quality

        def frames() -> Iterator[bytes]:
            previous = None
            while True:
                jpeg = service.camera_feed.latest_jpeg
                if jpeg is not None and jpeg is not previous:
                    previous = jpeg
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    view = compose_view(frame=frame)
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", view, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    )
                    if encoded_ok:
                        yield _mjpeg_part(jpeg=np.asarray(encoded).tobytes())
                time.sleep(MJPEG_FRAME_INTERVAL_SECONDS)

        return _mjpeg_response(frames=frames())

    @app.get("/stream/stereo")
    def stereo_stream():
        if service.stereo_view is None:
            return jsonify({"error": "No stereo view configured"}), 404
        return _stereo_stream(compose_view=service.stereo_view.side_by_side)

    @app.get("/stream/anaglyph")
    def anaglyph_stream():
        if service.stereo_view is None:
            return jsonify({"error": "No stereo view configured"}), 404
        return _stereo_stream(compose_view=service.stereo_view.anaglyph)

    return app
