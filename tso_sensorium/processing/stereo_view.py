"""Stereo view composition for live 3D preview streams.

Turns camera frames into side-by-side stereo pairs (for per-eye headset
rendering) and red-cyan anaglyphs (for depth perception on a normal
monitor with paper glasses). Interlaced stereo sources are deinterlaced
and optionally rectified with a stereo calibration; a duplicate mode
mirrors one mono frame to both eyes for testing the pipeline without a
stereo camera.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from tso_sensorium.processing.image import (
    convert_stereo_imgs_to_anaglyph,
    deinterlace_cv_image,
)
from tso_sensorium.processing.rectification import Rectifier
from tso_sensorium.resources import resolve_asset_path

INTERLACED_MODE = "interlaced"
DUPLICATE_MODE = "duplicate"
STEREO_VIEW_MODES = (INTERLACED_MODE, DUPLICATE_MODE)


class StereoViewProcessor:
    """Builds stereo views from raw camera frames.

    Args:
        mode: "interlaced" splits odd/even rows into the two eyes;
            "duplicate" shows the same frame to both eyes (no depth,
            useful for testing without a stereo camera).
        left_odd: Whether the left image is stored in the odd rows.
        calibration_path: OpenCV stereo calibration file used to rectify
            the pair; empty skips rectification. Supports ``package://``
            references to packaged assets.
    """

    def __init__(
        self,
        mode: str = INTERLACED_MODE,
        left_odd: bool = True,
        calibration_path: str = "",
    ):
        if mode not in STEREO_VIEW_MODES:
            raise ValueError(f"mode must be one of {STEREO_VIEW_MODES}, got '{mode}'")
        self.mode = mode
        self.left_odd = left_odd
        self.calibration_path = calibration_path
        self._rectifier: Optional[Rectifier] = None

    def _eye_views(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == DUPLICATE_MODE:
            return frame, frame
        left, right = deinterlace_cv_image(image=frame, left_odd=self.left_odd)
        if not self.calibration_path:
            return left, right
        if self._rectifier is None:
            self._rectifier = Rectifier(
                calibration_file_path=resolve_asset_path(path=self.calibration_path)
            )
        return (
            self._rectifier.rectify_left(image=left),
            self._rectifier.rectify_right(image=right),
        )

    def side_by_side(self, frame: np.ndarray) -> np.ndarray:
        """Compose the left|right eye views into one wide frame."""
        left, right = self._eye_views(frame=frame)
        return np.concatenate([left, right], axis=1)

    def anaglyph(self, frame: np.ndarray) -> np.ndarray:
        """Compose the eye views into a red-cyan anaglyph frame."""
        left, right = self._eye_views(frame=frame)
        return convert_stereo_imgs_to_anaglyph(left_img=left, right_img=right)
