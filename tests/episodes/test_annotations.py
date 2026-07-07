"""Tests for tso_sensorium.episodes.annotations module."""

import re

import pytest

from tso_sensorium.episodes.annotations import (
    EpisodeAnnotations,
    PhaseSegment,
)


@pytest.fixture
def annotations_factory():
    def factory() -> EpisodeAnnotations:
        return EpisodeAnnotations(
            segments=[
                PhaseSegment(start=0, end=100, phase=0),
                PhaseSegment(
                    start=100,
                    end=200,
                    phase=1,
                    source="manual",
                    language="close the gripper now",
                ),
            ]
        )

    return factory


class TestPhaseSegment:
    @pytest.mark.unit
    def test_contains_is_start_inclusive_end_exclusive(self):
        segment = PhaseSegment(start=10, end=20, phase=0)
        assert segment.contains(timestamp=10) is True
        assert segment.contains(timestamp=19) is True
        assert segment.contains(timestamp=20) is False

    @pytest.mark.unit
    def test_rejects_inverted_span(self):
        with pytest.raises(
            ValueError, match=re.escape("Segment end (5) must be after start (10)")
        ):
            PhaseSegment(start=10, end=5, phase=0)

    @pytest.mark.unit
    def test_rejects_unknown_source(self):
        with pytest.raises(
            ValueError,
            match=re.escape("source must be one of ('auto', 'manual'), got 'guess'"),
        ):
            PhaseSegment(start=0, end=1, phase=0, source="guess")


class TestEpisodeAnnotations:
    @pytest.mark.unit
    def test_segment_lookup(self, annotations_factory):
        annotations = annotations_factory()
        assert annotations.segment_at(timestamp=50).phase == 0
        assert annotations.segment_at(timestamp=150).language == (
            "close the gripper now"
        )
        assert annotations.segment_at(timestamp=250) is None

    @pytest.mark.unit
    def test_save_load_round_trip(self, annotations_factory, tmp_path):
        path = tmp_path / "annotations.json"
        annotations_factory().save(path=path)
        loaded = EpisodeAnnotations.load(path=path)
        assert loaded == annotations_factory()

    @pytest.mark.unit
    def test_load_missing_file_is_empty(self, tmp_path):
        assert EpisodeAnnotations.load(path=tmp_path / "missing.json") == (
            EpisodeAnnotations()
        )

    @pytest.mark.unit
    def test_replace_auto_segments_preserves_manual(self, annotations_factory):
        annotations = annotations_factory()
        annotations.replace_auto_segments(
            segments=[PhaseSegment(start=0, end=90, phase=2)]
        )
        phases = [(segment.phase, segment.source) for segment in annotations.segments]
        assert phases == [(2, "auto"), (1, "manual")]


@pytest.mark.unit
def test_segment_at_tolerance_extends_only_outer_edges():
    annotations = EpisodeAnnotations(
        segments=[
            PhaseSegment(start=100, end=200, phase=0),
            PhaseSegment(start=200, end=300, phase=1),
        ]
    )
    # Interior boundary keeps exact semantics: 200 belongs to phase 1.
    assert annotations.segment_at(timestamp=200, tolerance=50).phase == 1
    # Outer edges extend by tolerance.
    assert annotations.segment_at(timestamp=80, tolerance=50).phase == 0
    assert annotations.segment_at(timestamp=340, tolerance=50).phase == 1
    # Beyond tolerance stays uncovered.
    assert annotations.segment_at(timestamp=40, tolerance=50) is None
