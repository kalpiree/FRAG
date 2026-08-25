from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(left)
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def set_nested(config: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    node = config
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[keys[-1]] = value


def parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"Override must be key=value: {value}")
    key, raw = value.split("=", 1)
    if not key:
        raise ValueError(f"Override key is empty: {value}")
    return key, yaml.safe_load(raw)


def load_config(paths: list[str | Path], overrides: list[str] | None = None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"Configuration root must be a mapping: {path}")
        config = merge_dicts(config, loaded)
    for value in overrides or []:
        key, parsed = parse_override(value)
        set_nested(config, key, parsed)
    return config


def canonical_config(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config: dict[str, Any], length: int = 12) -> str:
    return hashlib.sha256(canonical_config(config).encode("utf-8")).hexdigest()[:length]


def save_config(config: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
