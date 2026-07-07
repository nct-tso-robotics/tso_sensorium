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
