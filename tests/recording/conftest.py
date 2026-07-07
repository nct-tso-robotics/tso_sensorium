"""Shared fixtures for recording tests requiring a live ROS 1 graph."""

import os
import shutil
import subprocess
import time

import pytest

ROS_MASTER_PORT = 11413


@pytest.fixture(scope="session")
def ros_master():
    # Imported lazily so this conftest also loads in ROS-free environments,
    # where the fixture skips instead of failing collection.
    rosgraph = pytest.importorskip("rosgraph")
    if shutil.which("roscore") is None:
        pytest.skip("roscore not available")
    master_uri = f"http://localhost:{ROS_MASTER_PORT}"
    os.environ["ROS_MASTER_URI"] = master_uri
    process = subprocess.Popen(
        ["roscore", "-p", str(ROS_MASTER_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        if rosgraph.is_master_online(master_uri):
            break
        time.sleep(0.5)
    yield master_uri
    process.terminate()
    process.wait()


@pytest.fixture(scope="session")
def ros_node(ros_master):
    rospy = pytest.importorskip("rospy")
    rospy.init_node("tso_recording_tests", anonymous=True)
    yield ros_master
