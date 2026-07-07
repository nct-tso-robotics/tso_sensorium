# TSO Sensorium

A library for sensor data recording, post-processing, and episodic dataset creation for robot learning.

## Package layout

```
tso_sensorium/
├── recording/     # sensor capture: core + ros1/ (Noetic) and ros2/ (Jazzy) adapters
├── processing/    # stereo rectification, deinterlacing, timestamp alignment
├── episodes/      # schema, episode assembly, parallel dataset building
├── export/        # dataset writers: CSV folders, LeRobot (v3.0 format)
└── scripts/       # command-line entry points
```

Processing, episode assembly, and export have no ROS dependency and run anywhere. Recording adapters import the ROS client libraries and are only usable inside the corresponding ROS environment.

## Installation

```bash
pip install -e .              # core: processing, episodes, export
pip install -e ".[lerobot]"   # + LeRobot dataset export (Python >= 3.10)
pip install -e ".[test]"      # + pytest
```

The ROS client libraries (`rospy`, `rclpy`) are not on PyPI, so there is no pip extra for recording: on a robot they come from the ROS installation itself (apt), and for development without a robot the [RoboStack](https://robostack.github.io/) conda environments in `environments/` provide them:

```bash
mamba env create -f environments/ros1-noetic.yml   # or ros2-jazzy.yml
mamba activate tso-sensorium-ros1
pip install -e . --no-deps                          # deps already come from conda
```

Note: the Noetic robot PC runs Python 3.8 and the LeRobot exporter needs Python 3.10+, so recording and LeRobot export are not meant to share one environment. Record on the robot, export on the processing machine.

## User guide

### Configuration

Every script is driven by a [pydantic](https://docs.pydantic.dev)-validated
YAML file under `configs/`, selected with `--config_path`. The same rules
apply everywhere:

- **Unknown keys fail loudly.** A typo'd key aborts with a validation error
  naming the key, instead of being silently ignored.
- **Polymorphic entries pick their variant with a `type:` key** — recorder
  kinds (`topic`, `video`), frame transforms (`deinterlace`, `resize`,
  `rectify`), table transforms (`parse_vector3`, `sum_columns`, ...),
  writers (`csv`, `lerobot`), phase labelers and triggers. The fields next
  to `type` are the variant's own options.
- **Configs compose with `!include`**, resolved relative to the including
  file. This keeps shared pieces in one place, e.g.
  `configs/recording/mock_service.yaml` includes
  `configs/dataset/mock.yaml` as its `generation:` section, and the same
  file is included by `configs/annotation/mock.yaml`.
- **Calibration files packaged with the library** can be referenced as
  `package://<asset name>`.

**CLI overrides**: any field can be overridden with `--dot.path value`
pairs on top of the YAML. Values are parsed as YAML scalars, so `true`,
`3.5`, and `[a, b]` become their typed equivalents. List elements are
addressed by index, and setting a `type` key resets that section to the
new variant (follow-up flags then fill its fields):

```bash
python -m tso_sensorium.scripts.generate_dataset \
    --config_path configs/dataset/bowel_retraction.yaml \
    --recordings_root /data/recordings \
    --save_frames true \
    --videos.0.frame_column framePath \
    --writer.type lerobot --writer.output_root /data/lerobot
```

The config modules require Python 3.9+ (pydantic); the core library stays
importable on Python 3.8.

### Recording

Record ROS topics and video streams in one shot (inside a ROS 1
environment):

```bash
python -m tso_sensorium.scripts.record \
    --config_path configs/recording/tso_testbed.yaml \
    --output_folder /data/recordings
```

For interactive sessions, `record_service` keeps a recording service running
and serves a browser dashboard (requires the `gui` extra):

```bash
pip install -e ".[gui]"
python -m tso_sensorium.scripts.record_service \
    --config_path configs/recording/tso_testbed_service.yaml \
    --session.output_folder /data/recordings
```

Open `http://<host>:8080` from any machine on the network. The dashboard
has two sections: **Record** (live camera feed, per-sensor liveness,
start/stop of demonstrations, per-episode topic selection) and **Library**
(episode browsing with in-browser replay, phase annotation, and dataset
generation — see below). The server is unauthenticated; expose it on
trusted networks only.

To try the full record → browse → annotate → generate loop without any
hardware, run the bundled synthetic sensors (in a ROS 1 environment, e.g.
`pixi run -e ros1`):

```bash
roscore &
python -m tso_sensorium.scripts.mock_sensors &
python -m tso_sensorium.scripts.record_service \
    --config_path configs/recording/mock_service.yaml
```

### Annotation and dataset studio (no ROS required)

The Library section also runs as a standalone app on any machine — a
processing workstation without ROS, pointed at a folder of recordings:

```bash
python -m tso_sensorium.scripts.annotate \
    --config_path configs/annotation/mock.yaml
# or directly:
python -m tso_sensorium.scripts.annotate \
    --recordings_root /data/recordings --port 8090
```

From the dashboard you can:

- **Switch the recordings folder** being browsed (the path field at the
  top; the episode list, annotations, and legend reload for that folder).
- **Edit the dataset metadata and phase legend**: dataset name, task, and
  the mapping of integer phase labels to a phase name plus its language
  instruction variants. One instruction per phase is deterministic;
  several lines make a stochastic mapping — one variant is sampled per
  episode, seeded by the episode name so regeneration is reproducible.
  The legend is saved to `dataset_metadata.json` at the recordings root.
- **Annotate episodes on a timeline** (pencil icon next to an episode):
  colored phase segments under the video, click to seek and select, drag
  segment edges, split at the playhead, and assign phases from the legend.
  Edits are saved to `annotations.json` inside the episode folder with
  `source: manual`, so re-running an automatic labeler never overwrites
  them.
- **Generate datasets** with per-run options (format, output root, frame
  extraction, sync tolerance); discarded episodes are listed with the
  reason.

### Automatic phase labeling

Labelers segment episodes from recorded signals and write automatic
segments into each episode's `annotations.json`:

```bash
python -m tso_sensorium.scripts.label_phases \
    --config_path configs/labeling/bowel_retraction.yaml \
    --recordings_root /data/recordings
```

`configs/labeling/` shows both labeler kinds: `column_threshold` (two
phases split by a signal threshold) and `sequential_trigger` (an ordered
phase sequence where each trigger hands over to the next phase — gripper
state changes, motion starting or settling). The typical workflow is
auto-label → correct visually in the dashboard → generate.

### Dataset generation

```bash
python -m tso_sensorium.scripts.generate_dataset \
    --config_path configs/dataset/bowel_retraction.yaml \
    --recordings_root /data/recordings

# Same recordings, exported as a LeRobot dataset instead:
python -m tso_sensorium.scripts.generate_dataset \
    --config_path configs/dataset/bowel_retraction.yaml \
    --recordings_root /data/recordings \
    --writer.type lerobot --writer.output_root /data/lerobot
```

When the config has an `annotations:` section, each episode's phase
segments are joined onto the aligned table as an integer phase column and
a language instruction column, and the writer stores the legend plus
generation statistics next to the dataset (`dataset_metadata.json`), so a
dataset is always reconstructable from recordings + annotations + config.

### Library usage

Generate an episode table programmatically:

```python
from tso_sensorium.episodes import EpisodeGenerator

episode = (
    EpisodeGenerator()
    .add_video(
        video_path="episode/left.mp4",
        timestamps_path="episode/left_timestamps.csv",
        sync_col_name="timestamp",
        frames_output_path="episode/frames/left",
        frame_col_name="left_frame",
        save_frames=True,
    )
    .add_state(
        state_data_path="episode/robot_state.csv",
        sync_col_name="timestamp",
        dataset_cols=["x", "y", "z", "roll"],
    )
    .generate_dataset()
    .save_dataset("episode/episode.csv")
)
```

Rectify and deinterlace stereo images:

```python
from tso_sensorium.processing import Rectifier, deinterlace_cv_image

rectifier = Rectifier(calibration_file_path="calibration.yml")
left, right = deinterlace_cv_image(image=interlaced_image)
left, right = rectifier.rectify(left=left, right=right)
```

## Development

```bash
pip install -e ".[test]"
pytest                 # unit tests
pytest -m ""           # all tests, including integration
ruff format tso_sensorium/ tests/ && ruff check tso_sensorium/ tests/
```

The ROS adapter tests skip unless the corresponding ROS client library is importable. The RoboStack environments in `environments/` make them runnable without a robot:

```bash
mamba env create -f environments/ros1-noetic.yml   # or ros2-jazzy.yml
mamba run -n tso-sensorium-ros1 python -m pytest tests/recording/ -m ""
```

### Pixi

The same environments are also declared in `pyproject.toml` under `[tool.pixi]`, with the exact solve locked in `pixi.lock`. After [installing pixi](https://pixi.sh):

```bash
pixi run test                                    # unit tests, default env
pixi run test-all                                # including integration tests
pixi run lint
pixi run -e ros1 pytest tests/recording -m ""    # ROS 1 adapter tests
pixi run -e ros2 pytest tests/recording -m ""    # ROS 2 adapter tests
```

Pixi creates the environments on first use and installs the package in editable mode automatically. On machines with a small home quota, point the package cache at larger storage first: `export PIXI_CACHE_DIR=/path/with/space`. LeRobot export intentionally stays a pip extra rather than a pixi environment, to avoid duplicating a multi-gigabyte torch install per checkout.
