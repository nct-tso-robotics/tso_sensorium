"""Synthetic scene shared by the ROS 1 and ROS 2 mock sensor scripts."""

from __future__ import annotations

import time

import cv2
import numpy as np

FRAME_HEIGHT = 540
FRAME_WIDTH = 960
PUBLISH_RATE_HZ = 15
GRIPPER_INTERVAL_TICKS = 45
CAMERA_TOPIC = "/mock/camera"
POSE_TOPIC = "/mock/pose"
GRIPPER_TOPIC = "/mock/gripper"


def render_frame(tick: int, base_x: np.ndarray, base_y: np.ndarray) -> np.ndarray:
    """Render one synthetic camera frame.

    Args:
        tick: Frame counter driving the animation.
        base_x: Column index grid, (H, W).
        base_y: Row index grid, (H, W).

    Returns:
        BGR frame, (H, W, 3).
    """
    phase = tick * 4
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    frame[:, :, 0] = ((base_x + phase) % 512 // 2).astype(np.uint8)
    frame[:, :, 1] = ((base_y + phase // 2) % 512 // 2).astype(np.uint8)
    frame[:, :, 2] = 40
    center = (
        int(FRAME_WIDTH / 2 + 260 * np.cos(tick / 20)),
        int(FRAME_HEIGHT / 2 + 160 * np.sin(tick / 20)),
    )
    cv2.circle(frame, center, 42, (60, 200, 90), -1)
    cv2.putText(
        frame,
        f"mock camera  {time.strftime('%H:%M:%S')}",
        (24, FRAME_HEIGHT - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (235, 235, 235),
        2,
    )
    return frame


def tool_position(tick: int) -> tuple[float, float, float]:
    """Circular tool position for one animation tick."""
    return float(np.cos(tick / 20)), float(np.sin(tick / 20)), 0.1


def gripper_open(tick: int) -> bool:
    """Toggling gripper state for one animation tick."""
    return (tick // GRIPPER_INTERVAL_TICKS) % 2 == 0
