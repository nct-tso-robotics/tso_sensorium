"""Tests for tso_sensorium.recording.network module."""

from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from tso_sensorium.recording.network import HOSTNAME_COMMAND, dashboard_url

RUN_PATH = "tso_sensorium.recording.network.subprocess.run"
GET_HOSTNAME_PATH = "tso_sensorium.recording.network.socket.gethostname"


@pytest.mark.unit
def test_dashboard_url_preserves_explicit_bind_host() -> None:
    with patch(RUN_PATH) as run:
        url = dashboard_url(bind_host="192.168.10.20", port=8080)

    assert url == "http://192.168.10.20:8080"
    run.assert_not_called()


@pytest.mark.unit
def test_dashboard_url_resolves_wildcard_to_first_non_loopback_ipv4() -> None:
    completed_process = CompletedProcess(
        args=HOSTNAME_COMMAND,
        returncode=0,
        stdout="127.0.0.1 192.168.10.20 2001:db8::1 10.0.0.4\n",
    )
    with patch(RUN_PATH, return_value=completed_process) as run:
        url = dashboard_url(bind_host="0.0.0.0", port=8080)

    assert url == "http://192.168.10.20:8080"
    run.assert_called_once_with(
        HOSTNAME_COMMAND,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.unit
def test_dashboard_url_uses_hostname_without_reachable_ipv4() -> None:
    completed_process = CompletedProcess(
        args=HOSTNAME_COMMAND,
        returncode=0,
        stdout="127.0.0.1 ::1\n",
    )
    with (
        patch(RUN_PATH, return_value=completed_process),
        patch(GET_HOSTNAME_PATH, return_value="recording-host") as get_hostname,
    ):
        url = dashboard_url(bind_host="0.0.0.0", port=8080)

    assert url == "http://recording-host:8080"
    get_hostname.assert_called_once_with()
