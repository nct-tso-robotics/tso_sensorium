"""Post-processing of recorded sensor data.

Provides stereo rectification, interlaced image handling, and timestamp
alignment of recorded data sources.
"""

from tso_sensorium.processing.alignment import StateData, VideoData
from tso_sensorium.processing.image import (
    convert_stereo_imgs_to_anaglyph,
    deinterlace_cv_image,
)
from tso_sensorium.processing.rectification import Rectifier

__all__ = [
    "Rectifier",
    "StateData",
    "VideoData",
    "convert_stereo_imgs_to_anaglyph",
    "deinterlace_cv_image",
]
