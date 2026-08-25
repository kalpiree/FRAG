from __future__ import annotations

import argparse
from pathlib import Path

from frag.data.download import DATA_SOURCES, download_dataset, resolve_source


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--variant", choices=("spoiler", "full"))
    parser.add_argument("--raw-dir")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--list", action="store_true")
    return parser


def _default_directory(project_root: Path, source_name: str) -> Path:
    directory_name = "goodreads" if source_name.startswith("goodreads-") else source_name
    return project_root / "data" / "raw" / directory_name


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.list:
        for name in sorted(DATA_SOURCES):
            source = DATA_SOURCES[name]
            print(f"{name}\t{source.citation_url}\t{source.terms}")
        return 0
    if not arguments.dataset:
        raise SystemExit("--dataset is required unless --list is used")
    source = resolve_source(arguments.dataset, arguments.variant)
    project_root = Path(__file__).resolve().parents[1]
    destination = (
        Path(arguments.raw_dir).expanduser().resolve()
        if arguments.raw_dir
        else _default_directory(project_root, source.name)
    )
    print(f"source={source.name}")
    print(f"terms={source.terms}")
    print(f"destination={destination}")
    results = download_dataset(
        arguments.dataset,
        destination,
        variant=arguments.variant,
        force=arguments.force,
        timeout=arguments.timeout,
    )
    if not results:
        print("download=not-required")
    for result in results:
        status = "downloaded" if result.downloaded else "existing"
        print(f"{status}\t{result.size_bytes}\t{result.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
