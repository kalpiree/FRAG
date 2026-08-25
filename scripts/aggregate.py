from __future__ import annotations

import argparse
import json
from pathlib import Path

from frag.runner import aggregate_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="*", default=["outputs"])
    parser.add_argument("--baseline-method")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_artifacts(args.roots, args.baseline_method)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(destination), "records": len(result["aggregates"])}))


if __name__ == "__main__":
    main()
