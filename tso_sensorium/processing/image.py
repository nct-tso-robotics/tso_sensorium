"""Image operations for interlaced stereo streams."""

from __future__ import annotations

import numpy as np


def deinterlace_cv_image(
    image: np.ndarray, left_odd: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Split an interlaced stereo image into left and right images.

    Args:
        image: Interlaced image where odd rows belong to one camera and
            even rows to the other, (H, W, C).
        left_odd: Whether the left image is stored in the odd rows.

    Returns:
        Tuple of (left, right) images, each (H/2, W, C).
    """
    odd_rows = image[1::2, :, :]
    even_rows = image[::2, :, :]
    if left_odd:
        return odd_rows, even_rows
    return even_rows, odd_rows


def convert_stereo_imgs_to_anaglyph(
    left_img: np.ndarray, right_img: np.ndarray
) -> np.ndarray:
    """Combine a stereo pair into a red-cyan anaglyph image.

    Args:
        left_img: Left camera image in BGR channel order, (H, W, 3).
        right_img: Right camera image in BGR channel order, (H, W, 3).

    Returns:
        Anaglyph image with the red channel from the left image and the
        blue and green channels from the right image, (H, W, 3).
    """
    anaglyph = np.zeros_like(left_img)
    anaglyph[:, :, 0] = right_img[:, :, 0]  # Blue
    anaglyph[:, :, 1] = right_img[:, :, 1]  # Green
    anaglyph[:, :, 2] = left_img[:, :, 2]  # Red
    return anaglyph
