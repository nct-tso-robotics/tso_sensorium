"""Tests for tso_sensorium.recording.episode_files module."""

import os

import pytest

from tso_sensorium.recording.episode_files import list_episodes


@pytest.mark.unit
def test_lists_episodes_newest_first_with_files(tmp_path):
    older = tmp_path / "episode_a"
    newer = tmp_path / "episode_b"
    for episode_directory in (older, newer):
        episode_directory.mkdir()
    (older / "state.csv").write_text("time,x\n")
    (older / "camera.mp4").write_bytes(b"0" * (1024 * 1024))
    (newer / "camera.avi").write_bytes(b"0" * 512)
    (tmp_path / "notes.txt").touch()
    os.utime(older, (1000, 1000))
    os.utime(newer, (2000, 2000))

    listings = list_episodes(output_folder=tmp_path)

    assert [listing.name for listing in listings] == ["episode_b", "episode_a"]
    older_files = {file.name: file for file in listings[1].files}
    assert older_files["camera.mp4"].playable is True
    assert older_files["camera.mp4"].size_megabytes == 1.0
    assert older_files["state.csv"].playable is False
    assert listings[0].files[0].playable is True


@pytest.mark.unit
def test_missing_root_returns_empty(tmp_path):
    assert list_episodes(output_folder=tmp_path / "missing") == []
