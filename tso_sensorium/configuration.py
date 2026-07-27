"""Config loading: strict pydantic models, YAML includes, dot overrides.

All configuration classes inherit ``ConfigModel``: unknown YAML keys fail
loudly, and polymorphic sections select their variant with a ``type`` key
(pydantic discriminated unions). Requires Python 3.9+; the runtime
packages stay importable without this module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ConfigDict
from yamlinclude import YamlIncludeConstructor

ConfigType = TypeVar("ConfigType", bound=BaseModel)
OverrideValue = Union[str, int, float, bool, None, list, dict]


class ConfigModel(BaseModel):
    """Base for all configuration classes; rejects unknown keys."""

    model_config = ConfigDict(extra="forbid")


def load_yaml_with_includes(path: Union[Path, str]) -> dict:
    """Load a YAML file resolving ``!include`` tags relative to it.

    Args:
        path: YAML file to load.

    Returns:
        The parsed document.
    """
    config_path = Path(path)

    class IncludeLoader(yaml.SafeLoader):
        pass

    YamlIncludeConstructor.add_to_loader_class(
        loader_class=IncludeLoader, relative=True
    )
    with open(config_path) as config_file:
        return yaml.load(config_file, IncludeLoader)


def set_by_path(tree: Union[dict, list], path: str, value: OverrideValue) -> None:
    """Set a value inside a nested structure by its dot path.

    Paths address dict keys and list indices, e.g. ``videos.0.frame_column``.
    Setting a ``type`` key clears its sibling keys, since changing the
    variant of a polymorphic section invalidates the previous variant's
    fields; follow-up overrides then fill the new variant.

    Args:
        tree: Structure modified in place.
        path: Dot-separated path to the value.
        value: Replacement value.
    """
    keys = path.split(".")
    node = tree
    for key in keys[:-1]:
        if isinstance(node, list):
            if not key.isdigit() or int(key) >= len(node):
                raise ValueError(f"Invalid override path '{path}'")
            node = node[int(key)]
        elif isinstance(node, dict) and key in node:
            node = node[key]
        else:
            raise ValueError(f"Invalid override path '{path}'")
    last_key = keys[-1]
    if isinstance(node, list):
        if not last_key.isdigit() or int(last_key) >= len(node):
            raise ValueError(f"Invalid override path '{path}'")
        node[int(last_key)] = value
    elif isinstance(node, dict):
        if last_key == "type":
            node.clear()
        node[last_key] = value
    else:
        raise ValueError(f"Invalid override path '{path}'")


def load_config(
    config_class: Type[ConfigType],
    config_path: Optional[Union[Path, str]] = None,
    overrides: Optional[Dict[str, OverrideValue]] = None,
) -> ConfigType:
    """Build a validated config from a YAML file and dot-path overrides.

    Args:
        config_class: Configuration model to validate against.
        config_path: YAML file; when ``None``, defaults are used.
        overrides: Mapping of dot path to replacement value, applied in
            order onto the YAML document.

    Returns:
        The validated configuration.
    """
    payload = load_yaml_with_includes(path=config_path) if config_path else {}
    if payload is None:
        payload = {}
    for path, value in (overrides or {}).items():
        set_by_path(tree=payload, path=path, value=value)
    return config_class.model_validate(payload)


def parse_config_from_cli(
    config_class: Type[ConfigType],
    arguments: Optional[List[str]] = None,
) -> ConfigType:
    """Build a config from ``--config_path`` plus ``--dot.path value`` flags.

    Override values are parsed as YAML scalars, so ``true``, ``3.5``, and
    ``[a, b]`` become their typed equivalents.

    Args:
        config_class: Configuration model to validate against.
        arguments: CLI arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The validated configuration.
    """
    parser = argparse.ArgumentParser(
        description=f"Configuration: {config_class.__name__}",
        epilog=(
            "Any field can be overridden with dot notation,"
            " e.g. --writer.type lerobot --videos.0.frame_column framePath"
        ),
    )
    parser.add_argument("--config_path", default=None, help="YAML configuration file")
    known, rest = parser.parse_known_args(arguments)
    overrides: Dict[str, OverrideValue] = {}
    index = 0
    while index < len(rest):
        flag = rest[index]
        if not flag.startswith("--") or index + 1 >= len(rest):
            parser.error(
                f"Overrides must come as '--dot.path value' pairs, got {rest[index:]}"
            )
        overrides[flag[2:]] = yaml.safe_load(rest[index + 1])
        index += 2
    return load_config(
        config_class=config_class,
        config_path=known.config_path,
        overrides=overrides,
    )
