# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Initial TSO Sensorium release: sensor data recording (ROS 1 and ROS 2
  adapters), post-processing (stereo rectification, deinterlacing,
  timestamp alignment), episode assembly, and dataset export (per-episode
  CSV and LeRobot v3.0 writers).
- Browser dashboard with a recording section (live feed, sensor liveness,
  topic selection) and a library section (episode replay, phase legend
  editing, timeline annotation, dataset generation), also runnable as a
  standalone ROS-free app.
- Phase annotation layer: per-episode `annotations.json`, dataset-level
  `dataset_metadata.json` with the phase legend, and automatic labelers
  (threshold and sequential-trigger).
- Pydantic-validated YAML configuration with `!include` composition and
  dot-notation CLI overrides.
- Coordinate-frame feature metadata describing exact component columns, frame
  names, and fixed, moving, or unknown frame temporality.
- Named auxiliary LeRobot features for labels that do not belong in robot
  state or action vectors.
- Dataset-level percentile denoising over configured action-column groups.
- Dashboard controls that clearly separate full dataset export from transactional
  action-only updates of an existing LeRobot v3 dataset.
- Real dataset-generation progress in the dashboard, including build phase,
  completed and total work units, percentage, current episode, and failures.
- Cooperative dashboard cancellation with operation-specific cleanup: staged
  action updates are discarded atomically, only fresh writer-owned LeRobot
  output may be removed, and existing datasets and recordings are preserved.
- Transactional LeRobot action-only updates that rewrite actions, their
  per-episode and global statistics, and generation provenance through an
  atomic staged swap while preserving videos and non-action data.

### Changed

- Record and export the combined endoscope stream at its nominal 30 fps instead
  of encoding and describing the 30 Hz input as 10 fps.

### Fixed

- Derive endoscope-guidance translation actions in the robot base before
  rotating them into the recorded camera frame at the start of each
  transition. Camera-frame state now uses the same recorded transform as
  deployment, and terminal rows without a successor are omitted.
- Use one phase-legend instruction per endoscope-guidance phase, independent
  of legacy per-segment language variants.
- Bound video timestamps to decodable frames so a trailing timestamp without
  a corresponding encoded frame cannot create an invalid frame path.
- Crop independently started sensor streams to their shared timestamp range
  before nearest-neighbor alignment. Startup and shutdown frames without a
  corresponding state sample no longer discard an otherwise valid episode;
  the configured tolerance still rejects synchronization gaps inside the
  shared interval.
- Make the ROS subscription queue depth configurable and retain 100 pending
  robot-state messages in the testbed configuration, preventing the observed
  500 Hz bursts from being reduced to the latest queued message.
