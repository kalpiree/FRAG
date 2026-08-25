from __future__ import annotations

import argparse
import json

from frag.config import load_config
from frag.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "full", "external"])
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--external")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.overrides)
    result = run_experiment(config, args.command, args.external)
    print(json.dumps(result.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
