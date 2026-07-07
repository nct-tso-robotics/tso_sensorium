"""Tests for tso_sensorium.episodes.phase_labelers module."""

from pydantic import TypeAdapter
import pandas as pd
import pytest

from tso_sensorium.episodes.annotations import EpisodeAnnotations, PhaseSegment
from tso_sensorium.episodes.phase_labelers import (
    AnyPhaseLabeler,
    ColumnThresholdLabeler,
    ColumnThresholdTrigger,
    MovementTrigger,
    PhaseLabelingConfig,
    SequentialPhase,
    SequentialTriggerLabeler,
    run_phase_labeling,
)


@pytest.fixture
def signal_table_factory():
    def factory() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "time": [0, 10, 20, 30, 40],
                "x": [-1.0, -0.5, 0.5, 1.0, -0.2],
            }
        )

    return factory


class TestColumnThresholdLabeler:
    @pytest.mark.unit
    def test_segments_follow_threshold_crossings(self, signal_table_factory):
        labeler = ColumnThresholdLabeler(
            column="x", threshold=0.0, above_phase=1, below_phase=0
        )
        segments = labeler.label(table=signal_table_factory(), sync_column="time")
        assert [
            (segment.start, segment.end, segment.phase) for segment in segments
        ] == [(0, 20, 0), (20, 40, 1), (40, 41, 0)]
        assert all(segment.source == "auto" for segment in segments)

    @pytest.mark.unit
    def test_empty_table_gives_no_segments(self):
        labeler = ColumnThresholdLabeler(column="x")
        table = pd.DataFrame({"time": [], "x": []})
        assert labeler.label(table=table, sync_column="time") == []

    @pytest.mark.unit
    def test_choice_registry_decodes_by_type_key(self):
        labeler = TypeAdapter(AnyPhaseLabeler).validate_python(
            {"type": "column_threshold", "column": "x", "threshold": 0.5}
        )
        assert labeler == ColumnThresholdLabeler(column="x", threshold=0.5)


class TestRunPhaseLabeling:
    @pytest.mark.unit
    def test_labels_episodes_and_preserves_manual_segments(
        self, tmp_path, signal_table_factory
    ):
        episode = tmp_path / "episode_000"
        episode.mkdir()
        signal_table_factory().to_csv(episode / "pose.csv", index=False)
        EpisodeAnnotations(
            segments=[PhaseSegment(start=0, end=5, phase=9, source="manual")]
        ).save(path=episode / "annotations.json")

        report = run_phase_labeling(
            config=PhaseLabelingConfig(
                recordings_root=str(tmp_path),
                state_file="pose.csv",
                labeler=ColumnThresholdLabeler(column="x"),
            )
        )

        assert report == {"episode_000": 3}
        annotations = EpisodeAnnotations.load(path=episode / "annotations.json")
        sources = [segment.source for segment in annotations.segments]
        assert sources.count("manual") == 1
        assert sources.count("auto") == 3

    @pytest.mark.unit
    def test_requires_recordings_root(self):
        with pytest.raises(ValueError, match="recordings_root is required"):
            run_phase_labeling(config=PhaseLabelingConfig())


class TestSequentialTriggerLabeler:
    @pytest.mark.unit
    def test_replicates_bowel_retraction_sequence(self):
        # Gripper: closed, opens at row 2, closes at row 4; tool moves
        # (>1e-3) from rows 6-7, settles (<=1e-4) at row 8.
        table = pd.DataFrame(
            {
                "time": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
                "open": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.02],
            }
        )
        labeler = SequentialTriggerLabeler(
            phases=[
                SequentialPhase(
                    phase=0,
                    trigger=ColumnThresholdTrigger(column="open", above=True),
                ),
                SequentialPhase(
                    phase=1,
                    trigger=ColumnThresholdTrigger(column="open", above=False),
                ),
                SequentialPhase(
                    phase=2,
                    trigger=MovementTrigger(columns=["x"], epsilon=1e-3, above=True),
                ),
                SequentialPhase(
                    phase=3,
                    trigger=MovementTrigger(columns=["x"], epsilon=1e-4, above=False),
                ),
                SequentialPhase(phase=4, trigger=None),
            ]
        )
        segments = labeler.label(table=table, sync_column="time")
        assert [
            (segment.start, segment.end, segment.phase) for segment in segments
        ] == [
            (0, 20, 0),
            (20, 40, 1),
            # Movement boundaries land on the row FROM which motion occurs,
            # matching the original bowel retraction algorithm.
            (40, 50, 2),
            (50, 70, 3),
            (70, 91, 4),
        ]

    @pytest.mark.unit
    def test_untriggered_phase_consumes_rest_and_later_phases_vanish(self):
        table = pd.DataFrame({"time": [0, 10, 20], "open": [0, 0, 0]})
        labeler = SequentialTriggerLabeler(
            phases=[
                SequentialPhase(
                    phase=0,
                    trigger=ColumnThresholdTrigger(column="open", above=True),
                ),
                SequentialPhase(phase=4, trigger=None),
            ]
        )
        segments = labeler.label(table=table, sync_column="time")
        assert [
            (segment.start, segment.end, segment.phase) for segment in segments
        ] == [(0, 21, 0)]

    @pytest.mark.unit
    def test_choice_registry_decodes_nested_triggers(self):
        labeler = TypeAdapter(AnyPhaseLabeler).validate_python(
            {
                "type": "sequential_trigger",
                "phases": [
                    {
                        "phase": 0,
                        "trigger": {
                            "type": "column_threshold",
                            "column": "open",
                            "above": True,
                        },
                    },
                    {"phase": 1, "trigger": None},
                ],
            },
        )
        assert [p.phase for p in labeler.phases] == [0, 1]
        assert isinstance(labeler.phases[0].trigger, ColumnThresholdTrigger)
