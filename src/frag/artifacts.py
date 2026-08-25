from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from frag.config import config_hash, save_config
from frag.runtime import save_environment


class RunArtifacts:
    def __init__(self, config: dict[str, Any], stage: str) -> None:
        root = Path(config["project"]["output_root"])
        experiment = config["project"].get("experiment", "main")
        dataset = config["dataset"]["name"]
        method = config["model"]["method"]
        seed = config["training"]["seed"]
        run_hash = config_hash(config)
        self.path = root / experiment / dataset / method / f"seed={seed}" / run_hash / stage
        self.path.mkdir(parents=True, exist_ok=True)
        save_config(config, self.path / "resolved_config.yaml")
        save_environment(self.path / "environment.json")

    def json(self, name: str, value: Any) -> Path:
        path = self.path / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=_json_default)
        return path

    def jsonl(self, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.path / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=_json_default))
                handle.write("\n")
        return path

    def parquet(self, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        path = self.path / name
        pd.DataFrame(list(rows)).to_parquet(path, index=False)
        return path


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")
