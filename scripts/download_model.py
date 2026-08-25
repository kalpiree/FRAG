from __future__ import annotations

import argparse
from pathlib import Path

from frag.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    config = load_config(args.config, args.override)
    model = config["model"]["generator"]
    destination = Path(model["local_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=model["model_id"],
        revision=model.get("revision"),
        local_dir=destination,
        ignore_patterns=["original/*", "*.pth"],
    )
    print(path)


if __name__ == "__main__":
    main()
