"""Record ROS 1 topics and video streams into an episode folder.

Records one episode until interrupted. The recorded topics are described
by a ``RecordingSessionConfig`` YAML (see ``configs/recording/``), with
dot-notation CLI overrides:

    python -m tso_sensorium.scripts.record \\
        --config_path configs/recording/tso_testbed.yaml \\
        --output_folder /data/recordings --record_rosbag true

For interactive recording with a browser dashboard, see
``tso_sensorium.scripts.record_service``.
"""

import signal

from tso_sensorium.configuration import parse_config_from_cli
import rospy

from tso_sensorium.recording.config import RecordingSessionConfig
from tso_sensorium.recording.ros1.session import EpisodeSession


def shutdown_handler(signum: int, frame, session: EpisodeSession) -> None:
    """Close the session when the program is interrupted.

    Args:
        signum: Signal number.
        frame: Current stack frame.
        session: Episode session to close.
    """
    rospy.loginfo("Shutdown signal received. Stopping node...")
    rospy.signal_shutdown("User interrupted")
    session.close()
    exit()


def run(config: RecordingSessionConfig) -> None:
    """Record one episode of the configured topics until interrupted.

    Args:
        config: Recording session configuration, typically loaded from a
            YAML file via ``--config_path``.
    """
    if not config.output_folder:
        raise ValueError("output_folder is required")
    rospy.init_node("episode_recording", anonymous=True)

    session = EpisodeSession(config=config, episode_name=config.episode_name)
    print(f"Recording episode.\nStored at: {session.output_folder}")
    session.start()
    signal.signal(
        signal.SIGINT,
        lambda signum, frame: shutdown_handler(
            signum=signum, frame=frame, session=session
        ),
    )
    rospy.spin()


def main() -> None:
    """Parse the CLI into a RecordingSessionConfig and run."""
    return run(config=parse_config_from_cli(config_class=RecordingSessionConfig))


if __name__ == "__main__":
    main()
