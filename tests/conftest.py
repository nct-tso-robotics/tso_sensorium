"""Shared fixtures for the tso_sensorium test suite."""

import numpy as np
import pandas as pd
import pytest

from tso_sensorium.episodes.schema import (
    ArmFeature,
    CameraFeature,
    DatasetSchema,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def schema_factory():
    def factory(
        cameras=None,
        arms=None,
        task="retract the bowel",
        fps=30,
    ) -> DatasetSchema:
        if cameras is None:
            cameras = [
                CameraFeature(name="left", frame_column="left_frame", height=8, width=8)
            ]
        if arms is None:
            arms = [
                ArmFeature(
                    name="main",
                    state_columns=["x", "y"],
                    action_columns=["dx", "dy"],
                )
            ]
        return DatasetSchema(
            name="dataset", fps=fps, cameras=cameras, arms=arms, task=task
        )

    return factory


@pytest.fixture
def episode_table_factory():
    def factory(length=2, frame_paths=None) -> pd.DataFrame:
        if frame_paths is None:
            frame_paths = [f"frames/{index}.png" for index in range(length)]
        return pd.DataFrame(
            {
                "x": [float(index) for index in range(length)],
                "y": [0.5 * index for index in range(length)],
                "dx": [0.1 * index for index in range(length)],
                "dy": [-0.1 * index for index in range(length)],
                "left_frame": list(frame_paths),
            }
        )

    return factory
