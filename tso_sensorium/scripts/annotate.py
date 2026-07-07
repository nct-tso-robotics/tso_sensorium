"""Standalone annotation and dataset generation app (no ROS required).

Serves the library dashboard for a folder of recorded episodes: browse,
replay, edit the phase legend, annotate timelines, and generate datasets.
Requires the ``gui`` extra (Flask).

    python -m tso_sensorium.scripts.annotate \\
        --recordings_root /data/recordings \\
        --generation.config_path? see configs/dataset/  # optional

Example with a generation config:

    python -m tso_sensorium.scripts.annotate \\
        --config_path configs/annotation/bowel_retraction.yaml \\
        --recordings_root /data/recordings
"""

from tso_sensorium.configuration import parse_config_from_cli

from tso_sensorium.recording.config import LibraryAppConfig
from tso_sensorium.recording.library_service import (
    LibraryService,
    create_library_app,
)


def run(config: LibraryAppConfig) -> None:
    """Serve the annotation dashboard until interrupted.

    Args:
        config: App configuration, typically loaded from a YAML file via
            ``--config_path``.
    """
    if not config.recordings_root:
        raise ValueError("recordings_root is required")
    library = LibraryService(
        recordings_root=config.recordings_root,
        generation=config.generation,
    )
    app = create_library_app(library=library)
    print(f"Annotation dashboard on http://{config.host}:{config.port}")
    app.run(host=config.host, port=config.port, threaded=True)


def main() -> None:
    """Parse the CLI into a LibraryAppConfig and run."""
    return run(config=parse_config_from_cli(config_class=LibraryAppConfig))


if __name__ == "__main__":
    main()
