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
