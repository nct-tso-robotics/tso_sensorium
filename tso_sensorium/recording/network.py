"""Network address helpers for browser dashboards."""

from __future__ import annotations

import ipaddress
import socket
import subprocess

HOSTNAME_COMMAND = ["hostname", "-I"]
WILDCARD_IPV4_ADDRESS = "0.0.0.0"


def dashboard_url(*, bind_host: str, port: int) -> str:
    """Build a reachable dashboard URL for a server bind address.

    Args:
        bind_host: Interface address passed to the HTTP server.
        port: TCP port exposed by the HTTP server.

    Returns:
        URL using a host reachable by clients.

    """
    display_host = bind_host
    if bind_host == WILDCARD_IPV4_ADDRESS:
        result = subprocess.run(
            HOSTNAME_COMMAND,
            check=False,
            capture_output=True,
            text=True,
        )
        addresses = (ipaddress.ip_address(value) for value in result.stdout.split())
        display_host = next(
            (
                str(address)
                for address in addresses
                if address.version == 4 and not address.is_loopback
            ),
            "",
        )
        if not display_host:
            display_host = socket.gethostname()
    return f"http://{display_host}:{port}"
