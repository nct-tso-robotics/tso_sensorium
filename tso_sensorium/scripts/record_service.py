"""Recording service with a browser dashboard.

Starts a long-running recording service on the robot PC: episodes are
started and stopped from any browser on the network, which also shows
the live camera feed, per-sensor liveness, and recorded episodes with
replay. Requires the ``gui`` extra (Flask).

    python -m tso_sensorium.scripts.record_service \\
        --config_path configs/recording/tso_testbed_service.yaml \\
        --session.output_folder /data/recordings
"""

import os
import signal

from tso_sensorium.configuration import parse_config_from_cli
import rospy

from tso_sensorium.recording.config import RecordingServiceConfig
from tso_sensorium.recording.ros1.web_service import RecordingService, create_app


def run(config: RecordingServiceConfig) -> None:
    """Serve the recording dashboard until interrupted.

    Args:
        config: Service configuration, typically loaded from a YAML file
            via ``--config_path``.
    """
    if not config.session.output_folder:
        raise ValueError("session.output_folder is required")
    # rospy must not own SIGINT here, otherwise Ctrl-C stops the ROS node
    # but leaves the HTTP server (and its port) alive.
    rospy.init_node("recording_service", anonymous=True, disable_signals=True)
    service = RecordingService(config=config)
    app = create_app(service=service)

    def shutdown_handler(signum: int, frame) -> None:
        print("Shutting down recording service...")
        service.close()
        rospy.signal_shutdown("Service stopped")
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    print(f"Recording dashboard on http://{config.host}:{config.port}")
    ssl_context = None
    if config.ssl_certificate and config.ssl_private_key:
        ssl_context = (config.ssl_certificate, config.ssl_private_key)
    app.run(host=config.host, port=config.port, threaded=True, ssl_context=ssl_context)
    service.close()


def main() -> None:
    """Parse the CLI into a RecordingServiceConfig and run."""
    return run(config=parse_config_from_cli(config_class=RecordingServiceConfig))


if __name__ == "__main__":
    main()
