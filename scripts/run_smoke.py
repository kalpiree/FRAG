from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    from frag.config import load_config
    from frag.runner import run_experiment

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    config = load_config([root / "configs/base.yaml", root / "configs/examples/smoke.yaml"])
    result = run_experiment(config, "full")
    print(json.dumps(result.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
