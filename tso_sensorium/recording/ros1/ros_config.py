"""ROS 1 environment configuration for out-of-workspace nodes."""

from __future__ import annotations

import os
import sys
from typing import Optional

ROS_IP_VARIABLE = "ROS_IP"
ROS_HOSTNAME_VARIABLE = "ROS_HOSTNAME"
ROS_MASTER_URI_VARIABLE = "ROS_MASTER_URI"


def setup_ros_environment(
    hostname: str,
    ros_ip: str,
    ros_master_uri: str,
    catkin_ws_paths: Optional[list[str]] = None,
) -> None:
    """Set the ROS environment variables and source catkin workspaces.

    Args:
        hostname: Hostname of the machine running this code (not the ROS
            master).
        ros_ip: IP address of the machine running this code (not the ROS
            master).
        ros_master_uri: ROS master URI, e.g. "http://host:11311".
        catkin_ws_paths: Catkin workspaces whose Python packages are added
            to the import path.
    """
    os.environ[ROS_IP_VARIABLE] = ros_ip
    os.environ[ROS_HOSTNAME_VARIABLE] = hostname
    os.environ[ROS_MASTER_URI_VARIABLE] = ros_master_uri
    if catkin_ws_paths is None:
        catkin_ws_paths = []
    for catkin_ws_path in catkin_ws_paths:
        sys.path.insert(0, f"{catkin_ws_path}/devel/lib/python3/dist-packages")
