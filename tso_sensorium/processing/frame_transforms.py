"""Configurable frame preprocessing transforms.

Each transform is selected by its ``type`` key in YAML and is callable on
a single frame. This module requires Python 3.9+; the rest of the
processing package stays importable without it.
"""

from __future__ import annotations

import abc
from typing import Annotated, Any, Literal, Optional, Union

import cv2
import numpy as np
from pydantic import Field, PrivateAttr

from tso_sensorium.configuration import ConfigModel

from tso_sensorium.processing.image import deinterlace_cv_image
from tso_sensorium.processing.rectification import Rectifier
from tso_sensorium.resources import resolve_asset_path

LEFT_SIDE = "left"
RIGHT_SIDE = "right"
STEREO_SIDES = (LEFT_SIDE, RIGHT_SIDE)


class FrameTransform(ConfigModel, abc.ABC):
    """A single preprocessing step applied to a frame."""

    @abc.abstractmethod
    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Transform one frame."""


class DeinterlaceHalf(FrameTransform):
    """Extract one camera's half from an interlaced stereo frame.

    Args:
        side: Which half to keep, "left" or "right".
        left_odd: Whether the left image is stored in the odd rows.
    """

    type: Literal["deinterlace"] = "deinterlace"
    side: str = LEFT_SIDE
    left_odd: bool = True

    def model_post_init(self, context: Any) -> None:
        if self.side not in STEREO_SIDES:
            raise ValueError(f"side must be one of {STEREO_SIDES}, got '{self.side}'")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        left, right = deinterlace_cv_image(image=frame, left_odd=self.left_odd)
        return left if self.side == LEFT_SIDE else right


class Resize(FrameTransform):
    """Resize a frame to a fixed size.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.
    """

    type: Literal["resize"] = "resize"
    width: int = 960
    height: int = 540

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, (self.width, self.height))


class RectifySide(FrameTransform):
    """Rectify a frame as one side of a calibrated stereo pair.

    The rectifier is built lazily on first use so the transform stays
    cheap to serialize into parallel workers.

    Args:
        side: Which camera the frame belongs to, "left" or "right".
        calibration_path: OpenCV stereo calibration file; supports
            ``package://`` references to packaged assets.
    """

    type: Literal["rectify"] = "rectify"
    side: str = LEFT_SIDE
    calibration_path: str = ""

    # Private attribute: it must stay out of serialization and equality.
    _rectifier: Optional[Rectifier] = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        if self.side not in STEREO_SIDES:
            raise ValueError(f"side must be one of {STEREO_SIDES}, got '{self.side}'")
        if not self.calibration_path:
            raise ValueError("rectify transform requires a calibration_path")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self._rectifier is None:
            self._rectifier = Rectifier(
                calibration_file_path=resolve_asset_path(path=self.calibration_path)
            )
        if self.side == LEFT_SIDE:
            return self._rectifier.rectify_left(image=frame)
        return self._rectifier.rectify_right(image=frame)


AnyFrameTransform = Annotated[
    Union[DeinterlaceHalf, Resize, RectifySide], Field(discriminator="type")
]


def compose_transforms(transforms: list[FrameTransform]):
    """Compose transforms into a single frame preprocessing function.

    Args:
        transforms: Transforms applied in order.

    Returns:
        Function applying all transforms, or ``None`` when empty so video
        sources can skip preprocessing entirely.
    """
    if not transforms:
        return None

    def preprocess(frame: np.ndarray) -> np.ndarray:
        for transform in transforms:
            frame = transform(frame)
        return frame

    return preprocess
