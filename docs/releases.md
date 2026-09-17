# Releases

GitHub Actions runs Ruff, compilation, and the full test suite in the locked
default, ROS 1, and ROS 2 Pixi environments. It also builds a source distribution
and wheel, checks their metadata, and tests a clean wheel installation without ROS.
Optional dependencies not installed in those environments have their tests skipped.

## One-time setup

Create a GitHub environment named `pypi` in this repository's settings. Restrict
its deployment tags to `v*`.

In [PyPI's publishing settings](https://pypi.org/manage/account/publishing/), add
a pending Trusted Publisher with these values:

| Field | Value |
| --- | --- |
| PyPI project name | `tso-sensorium` |
| GitHub owner | `nct-tso-robotics` |
| Repository | `tso_sensorium` |
| Workflow filename | `ci.yml` |
| Environment | `pypi` |

If the PyPI project already exists, add the publisher in its project settings
instead. No PyPI token or GitHub secret is needed.

## Publish

1. Update the version in `pyproject.toml` and `tso_sensorium/__init__.py`, then
   merge the change into `main` after CI passes.
2. Create and publish a GitHub release from that commit with a matching tag,
   for example `v0.1.0` for version `0.1.0`.

The release reruns all checks and publishes the tested wheel and source
distribution to PyPI. Published prereleases also trigger publishing; use a
prerelease package version such as `0.2.0rc1` with tag `v0.2.0rc1`.
Draft releases, ordinary pushes, and pull requests never publish.
Each release needs a new version; PyPI does not allow replacing uploaded files.
