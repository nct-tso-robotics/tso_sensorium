"""Sensor data recording.

The framework-free file writers live in ``core``. Framework adapters live
in the ``ros1`` and ``ros2`` subpackages, which require the corresponding
ROS installation to be importable.
"""

from tso_sensorium.recording.core import (
    TimestampedCsvRecorder,
    VideoFileWriter,
)

__all__ = ["TimestampedCsvRecorder", "VideoFileWriter"]
