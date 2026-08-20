"""Shared fixtures for recording tests requiring a live ROS 1 graph."""

import os
import signal
import shutil
import subprocess
import time
from collections.abc import Generator

import pytest

ROS_MASTER_PORT = 11413
ROS_MASTER_START_TIMEOUT_SECONDS = 30.0
ROS_MASTER_STOP_TIMEOUT_SECONDS = 5.0
ROS_MASTER_POLL_INTERVAL_SECONDS = 0.1
ROS_MASTER_URI_VARIABLE = "ROS_MASTER_URI"
ROSCORE_COMMAND = "roscore"


def _stop_process(process: subprocess.Popen[str]) -> str:
    """Stop a process group within a bounded time and return its output."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + ROS_MASTER_STOP_TIMEOUT_SECONDS
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(ROS_MASTER_POLL_INTERVAL_SECONDS)
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
    output, _ = process.communicate()
    return output


@pytest.fixture(scope="session")
def ros_master() -> Generator[str, None, None]:
    """Run an isolated ROS master for the recording integration tests."""
    rosgraph = pytest.importorskip("rosgraph")
    if shutil.which(ROSCORE_COMMAND) is None:
        pytest.skip(reason="roscore not available")
    master_uri = f"http://localhost:{ROS_MASTER_PORT}"
    previous_master_uri = os.environ.get(ROS_MASTER_URI_VARIABLE)
    os.environ[ROS_MASTER_URI_VARIABLE] = master_uri
    process = subprocess.Popen(
        [ROSCORE_COMMAND, "-p", str(ROS_MASTER_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        deadline = time.monotonic() + ROS_MASTER_START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if rosgraph.is_master_online(master_uri):
                break
            if process.poll() is not None:
                output, _ = process.communicate()
                pytest.fail(reason=f"roscore exited during startup:\n{output.strip()}")
            time.sleep(ROS_MASTER_POLL_INTERVAL_SECONDS)
        else:
            output = _stop_process(process=process)
            pytest.fail(
                reason=(
                    f"roscore did not become available at {master_uri} within "
                    f"{ROS_MASTER_START_TIMEOUT_SECONDS:.0f} seconds.\n"
                    f"{output.strip()}"
                )
            )
        yield master_uri
    finally:
        if process.poll() is None:
            _stop_process(process=process)
        if previous_master_uri is None:
            os.environ.pop(ROS_MASTER_URI_VARIABLE, None)
        else:
            os.environ[ROS_MASTER_URI_VARIABLE] = previous_master_uri


@pytest.fixture(scope="session")
def ros_node(ros_master: str) -> Generator[str, None, None]:
    """Initialize and cleanly shut down the test ROS node."""
    rospy = pytest.importorskip("rospy")
    rospy.init_node("tso_recording_tests", anonymous=True)
    yield ros_master
    rospy.signal_shutdown("ROS recording tests completed")
