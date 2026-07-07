"""Browser dashboard and HTTP API for episode recording.

Runs a Flask app inside the recording process: browsers on the network
get a live camera feed, per-sensor liveness, start/stop control, topic
selection, and the full episode library (annotation and dataset
generation). The server is unauthenticated and should be exposed on
trusted networks only.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator, Optional

from flask import Flask, Response, jsonify, request

from sensor_msgs.msg import Image

from tso_sensorium.recording.config import (
    RecordingServiceConfig,
    TopicRecorderConfig,
)
from tso_sensorium.recording.library_service import (
    LibraryService,
    register_dashboard_route,
    register_error_handlers,
    register_library_routes,
)
from tso_sensorium.recording.message_fields import resolve_message_type
from tso_sensorium.recording.ros1.liveness import (
    CameraFeed,
    TopicLivenessMonitor,
)
from tso_sensorium.recording.ros1.session import EpisodeSession

IDLE_STATE = "idle"
RECORDING_STATE = "recording"
MJPEG_BOUNDARY = "frame"
MJPEG_FRAME_INTERVAL_SECONDS = 0.05


class RecordingService:
    """State machine behind the dashboard: idle or recording one episode.

    Args:
        config: Service configuration.
    """

    def __init__(self, config: RecordingServiceConfig):
        self.config = config
        self.library = LibraryService(
            recordings_root=config.session.output_folder,
            generation=config.generation,
        )
        self._session: Optional[EpisodeSession] = None
        self._session_started_at: Optional[float] = None
        self._lock = threading.Lock()
        topic_message_types = {
            recorder.topic_name: (
                resolve_message_type(dotted_path=recorder.message_type)
                if isinstance(recorder, TopicRecorderConfig)
                else Image
            )
            for recorder in config.session.recorders
        }
        self.liveness = TopicLivenessMonitor(
            topic_message_types=topic_message_types,
            staleness_seconds=config.staleness_seconds,
        )
        self.camera_feed = (
            CameraFeed(topic_name=config.camera_topic) if config.camera_topic else None
        )

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
            session = EpisodeSession(
                config=self.config.session,
                episode_name=episode_name,
                recorder_names=recorder_names,
            )
            session.start()
            self._session = session
            self._session_started_at = time.monotonic()
            return {"episode_name": session.episode_name}

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

    @app.get("/stream/camera")
    def camera_stream():
        if service.camera_feed is None:
            return jsonify({"error": "No camera feed configured"}), 404

        def frames() -> Iterator[bytes]:
            while True:
                jpeg = service.camera_feed.latest_jpeg
                if jpeg is not None:
                    yield (
                        b"--" + MJPEG_BOUNDARY.encode() + b"\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                time.sleep(MJPEG_FRAME_INTERVAL_SECONDS)

        return Response(
            frames(),
            mimetype=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
        )

    return app
