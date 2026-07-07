"""Automatically label episode phases from recorded signals.

Runs a configured phase labeler over every episode folder and merges the
resulting segments into each episode's ``annotations.json``, preserving
manual segments:

    python -m tso_sensorium.scripts.label_phases \\
        --config_path configs/labeling/mock.yaml \\
        --recordings_root /tmp/tso_sensorium_mock
"""

from tso_sensorium.configuration import parse_config_from_cli

from tso_sensorium.episodes.phase_labelers import (
    PhaseLabelingConfig,
    run_phase_labeling,
)


def run(config: PhaseLabelingConfig) -> None:
    """CLI entry point; see the module docstring for usage."""
    report = run_phase_labeling(config=config)
    for episode_name, segment_count in report.items():
        print(f"{episode_name}: {segment_count} segments")


def main() -> None:
    """Parse the CLI into a PhaseLabelingConfig and run."""
    return run(config=parse_config_from_cli(config_class=PhaseLabelingConfig))


if __name__ == "__main__":
    main()
