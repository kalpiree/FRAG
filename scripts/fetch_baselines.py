from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/baselines.yaml")
    parser.add_argument("--destination", default="external")
    parser.add_argument("--name", action="append")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.manifest).open("r", encoding="utf-8") as handle:
        repositories = yaml.safe_load(handle)["repositories"]
    names = args.name or list(repositories)
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        entry = repositories[name]
        path = destination / name
        if not path.exists():
            subprocess.run(["git", "clone", entry["url"], str(path)], check=True)
        subprocess.run(["git", "fetch", "--all", "--tags"], cwd=path, check=True)
        subprocess.run(["git", "checkout", "--detach", entry["revision"]], cwd=path, check=True)
        print(f"{name}: {entry['revision']}")


if __name__ == "__main__":
    main()
