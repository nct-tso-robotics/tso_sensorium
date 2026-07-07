"""Generate a dataset from recorded episode folders.

Fully config-driven: the dataset layout, preprocessing, table transforms,
and output format are described by a ``DatasetGenerationConfig`` YAML
(see ``configs/dataset/``). Values can be overridden from the CLI with
dot notation:

    python -m tso_sensorium.scripts.generate_dataset \\
        --config_path configs/dataset/bowel_retraction.yaml \\
        --recordings_root /data/recordings \\
        --writer.type lerobot --writer.output_root /data/lerobot
"""

from tso_sensorium.configuration import parse_config_from_cli

from tso_sensorium.episodes.dataset_builder import BuildReport
from tso_sensorium.episodes.generation import generate_dataset
from tso_sensorium.episodes.generation_config import DatasetGenerationConfig


def run(config: DatasetGenerationConfig) -> BuildReport:
    """CLI entry point; see the module docstring for usage."""
    report = generate_dataset(config=config)
    print(f"Wrote {len(report.written)} episodes")
    for episode_name, reason in report.failed.items():
        print(f"Discarded {episode_name}: {reason}")
    return report


def main() -> BuildReport:
    """Parse the CLI into a DatasetGenerationConfig and run."""
    return run(config=parse_config_from_cli(config_class=DatasetGenerationConfig))


if __name__ == "__main__":
    main()
