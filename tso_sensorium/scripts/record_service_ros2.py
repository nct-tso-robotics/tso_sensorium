"""ROS 2 recording service with a browser dashboard.

Starts a long-running recording service: episodes are started and stopped
from any browser on the network, which also shows the live camera feed,
per-sensor liveness, and recorded episodes with replay. Requires the
``gui`` extra (Flask).

    python -m tso_sensorium.scripts.record_service_ros2 \\
        --config_path configs/recording/tso_testbed_service.yaml \\
        --session.output_folder /data/recordings
"""

import os
import signal
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from tso_sensorium.configuration import parse_config_from_cli
from tso_sensorium.recording.config import RecordingServiceConfig
from tso_sensorium.recording.ros2.web_service import (
    build_recording_service,
    create_app,
)


def run(config: RecordingServiceConfig) -> None:
    """Serve the recording dashboard until interrupted.

    Args:
        config: Service configuration, typically loaded from a YAML file
            via ``--config_path``.
    """
    if not config.session.output_folder:
        raise ValueError("session.output_folder is required")
    rclpy.init()
    node = rclpy.create_node("recording_service")
    service = build_recording_service(node=node, config=config)
    app = create_app(service=service)

    # Subscriptions only deliver while the node spins; the HTTP server
    # owns the main thread, so spin in the background.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown_handler(signum: int, frame) -> None:
        print("Shutting down recording service...")
        service.close()
        executor.shutdown(timeout_sec=2.0)
        rclpy.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    print(f"Recording dashboard on http://{config.host}:{config.port}")
    app.run(host=config.host, port=config.port, threaded=True)
    service.close()


def main() -> None:
    """Parse the CLI into a RecordingServiceConfig and run."""
    return run(config=parse_config_from_cli(config_class=RecordingServiceConfig))


if __name__ == "__main__":
    main()
