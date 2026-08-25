"""Browser UI for interactive ROS 2 recording.

Starts the recording UI. A browser can start and stop episodes, view the live
camera and sensor readiness, and browse recorded episodes. Requires the
``gui`` extra (Flask).

    python -m tso_sensorium.scripts.record_ui_ros2 \\
        --config_path configs/recording/tso_testbed_ui.yaml \\
        --session.output_folder "$HOME/tso_sensorium_recordings"
"""

import os
import signal
import threading

import rclpy
from rclpy.executors import SingleThreadedExecutor

from tso_sensorium.configuration import parse_config_from_cli
from tso_sensorium.recording.config import RecordingUIConfig
from tso_sensorium.recording.network import dashboard_url
from tso_sensorium.recording.ros2.web_service import (
    build_recording_service,
    create_app,
)


def run(config: RecordingUIConfig) -> None:
    """Run the recording UI until interrupted.

    Args:
        config: UI configuration, typically loaded from a YAML file
            via ``--config_path``.
    """
    if not config.session.output_folder:
        raise ValueError("session.output_folder is required")
    rclpy.init()
    node = rclpy.create_node("recording_ui")
    service = build_recording_service(node=node, config=config)
    app = create_app(service=service)

    # Subscriptions only deliver while the node spins; the HTTP server
    # owns the main thread, so spin in the background.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    def shutdown_handler(signum: int, frame) -> None:
        print("Shutting down recording UI...")
        service.close()
        executor.shutdown(timeout_sec=2.0)
        rclpy.shutdown()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    print(f"Recording UI on {dashboard_url(bind_host=config.host, port=config.port)}")
    app.run(host=config.host, port=config.port, threaded=True)
    service.close()


def main() -> None:
    """Parse the recording UI configuration and run it."""
    return run(config=parse_config_from_cli(config_class=RecordingUIConfig))


if __name__ == "__main__":
    main()
