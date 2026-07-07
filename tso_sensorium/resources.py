"""Access to data files packaged with tso_sensorium."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Union

import tso_sensorium

PACKAGE_ASSET_SCHEME = "package://"

ASSETS_DIR = files(tso_sensorium).joinpath("assets")
STORZ_ENDOSCOPE_CALIBRATION_PATH = ASSETS_DIR / "storz_endoscope_calibration.yml"


def resolve_asset_path(path: Union[Path, str]) -> str:
    """Resolve a path that may reference a packaged asset.

    Paths starting with ``package://`` are resolved against the package's
    assets directory (e.g. ``package://storz_endoscope_calibration.yml``);
    anything else is returned unchanged.

    Args:
        path: Filesystem path or ``package://`` asset reference.

    Returns:
        Resolved path as a string.
    """
    path_string = str(path)
    if path_string.startswith(PACKAGE_ASSET_SCHEME):
        asset_name = path_string[len(PACKAGE_ASSET_SCHEME) :]
        return str(ASSETS_DIR.joinpath(asset_name))
    return path_string
