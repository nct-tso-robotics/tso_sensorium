"""Tests for tso_sensorium.episodes.schema module."""

import re

import pandas as pd
import pytest

from tso_sensorium.episodes.schema import ArmFeature


class TestColumnAggregation:
    @pytest.mark.unit
    def test_columns_concatenate_across_arms_in_order(self, schema_factory):
        schema = schema_factory(
            arms=[
                ArmFeature(
                    name="left_arm",
                    state_columns=["l_x"],
                    action_columns=["l_dx"],
                ),
                ArmFeature(
                    name="right_arm",
                    state_columns=["r_x", "r_y"],
                    action_columns=["r_dx"],
                ),
            ]
        )
        assert schema.state_columns == ["l_x", "r_x", "r_y"]
        assert schema.action_columns == ["l_dx", "r_dx"]

    @pytest.mark.unit
    def test_single_arm_columns_pass_through(self, schema_factory):
        schema = schema_factory(
            arms=[
                ArmFeature(
                    name="main",
                    state_columns=["x", "y"],
                    action_columns=["dx"],
                )
            ]
        )
        assert schema.state_columns == ["x", "y"]
        assert schema.action_columns == ["dx"]


class TestEpisodeTableValidation:
    @pytest.mark.unit
    def test_accepts_table_with_all_schema_columns(self, schema_factory):
        schema = schema_factory()
        table = pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
                "dx": [0.0],
                "dy": [0.0],
                "left_frame": ["frames/0.png"],
                "extra": [1],
            }
        )
        schema.validate_episode_table(table=table)

    @pytest.mark.unit
    def test_rejects_table_listing_missing_columns(self, schema_factory):
        schema = schema_factory()
        table = pd.DataFrame({"x": [0.0], "dx": [0.0]})
        missing = ["y", "dy", "left_frame"]
        with pytest.raises(
            ValueError,
            match=re.escape(f"Episode table is missing schema columns: {missing}"),
        ):
            schema.validate_episode_table(table=table)
