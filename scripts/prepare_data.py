from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from frag.config import load_config
from frag.data.prepare import prepare_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--no-write", action="store_true")
    return parser


def _resolve_dataset_paths(config: dict[str, Any], project_root: Path) -> None:
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise TypeError("Configuration must contain a dataset mapping")
    for key in ("raw_dir", "processed_dir"):
        if key not in dataset:
            continue
        value = Path(str(dataset[key])).expanduser()
        dataset[key] = str(value if value.is_absolute() else project_root / value)


def main() -> int:
    arguments = _parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    paths = arguments.config or [str(project_root / "configs" / "base.yaml")]
    config = load_config(paths, arguments.overrides)
    _resolve_dataset_paths(config, project_root)
    prepared = prepare_dataset(config, write=not arguments.no_write)
    print(json.dumps(prepared.manifest, indent=2, sort_keys=True))
    for path in prepared.output_files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
