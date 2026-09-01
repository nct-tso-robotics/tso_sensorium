"""Static integrity checks for the dashboard page asset."""

import re

import pytest

from tso_sensorium.resources import ASSETS_DIR


@pytest.fixture
def dashboard_page():
    return ASSETS_DIR.joinpath("recording_dashboard.html").read_text()


@pytest.mark.unit
def test_script_references_existing_elements(dashboard_page):
    defined_ids = set(re.findall(r'id="([^"]+)"', dashboard_page))
    referenced_ids = set(re.findall(r'getElementById\("([^"]+)"\)', dashboard_page))
    referenced_ids |= {
        selector[1:]
        for selector in re.findall(
            r"querySelector(?:All)?\([\"']([^\"']+)[\"']\)", dashboard_page
        )
        if selector.startswith("#") and " " not in selector and "[" not in selector
    }
    missing = sorted(referenced_ids - defined_ids)
    assert missing == [], (
        f"Dashboard JS references ids missing from the HTML: {missing}."
        " A single missing element kills the whole page script at load."
    )


@pytest.mark.unit
def test_no_merge_conflict_markers(dashboard_page):
    assert "<<<<<<<" not in dashboard_page
    assert ">>>>>>>" not in dashboard_page


@pytest.mark.unit
def test_denoising_sliders_feed_discovered_generation_override_paths(
    dashboard_page,
):
    assert 'requestJson("/api/generation/denoising-preview"' in dashboard_page
    assert "state.denoisingPercentiles[group.override_path]" in dashboard_page
    assert "overrides[path] = percentile" in dashboard_page
    assert "group.histogram" in dashboard_page
    assert "log10(action magnitude)" in dashboard_page
    assert "nonzero-action percentile" not in dashboard_page
    assert "Observation Deltas" not in dashboard_page
    assert (
        "dataset_transforms.0.column_groups.translation.percentile"
        not in dashboard_page
    )


@pytest.mark.unit
def test_actions_only_controls_are_explicit_and_remove_format_ambiguity(
    dashboard_page,
):
    assert "actions only — existing LeRobot" in dashboard_page
    assert "re-extract camera frames into recordings" in dashboard_page
    assert "framesOption.hidden = actionsOnly" in dashboard_page
    assert "formatOption.hidden = actionsOnly" in dashboard_page
    assert "actionRootOption.hidden = !actionsOnly" in dashboard_page
    assert "Actions-only update." in dashboard_page
    assert "Updates actions and statistics" in dashboard_page
    assert "videos and observations stay unchanged" in dashboard_page
    assert "Aborting leaves the existing dataset untouched" in dashboard_page
    assert "use-videos" not in dashboard_page.lower()


@pytest.mark.unit
def test_generation_modal_uses_real_status_and_cooperative_cancel_endpoint(
    dashboard_page,
):
    assert 'requestJson("/api/generation/start"' in dashboard_page
    assert 'requestJson("/api/generation/cancel"' in dashboard_page
    assert "generation.completed" in dashboard_page
    assert "generation.total" in dashboard_page
    assert "generation.percentage" in dashboard_page
    assert "generation.current_episode" in dashboard_page
    assert "Cancelling after current episode" in dashboard_page
    assert "Abort and discard staged update" in dashboard_page
    assert "Abort and delete fresh partial output" in dashboard_page
    assert "Stop after current episode" in dashboard_page
    assert "CSV files already written stay in place" in dashboard_page
    assert 'const finalizing = ["finalizing", "completed"]' in dashboard_page
    assert "cancellation is no longer available" in dashboard_page
    assert "setTimeout" not in dashboard_page
