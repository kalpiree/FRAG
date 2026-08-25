from __future__ import annotations

import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    name: str
    url: str
    expected_bytes: int | None = None


@dataclass(frozen=True)
class DataSource:
    name: str
    files: tuple[SourceFile, ...]
    terms: str
    citation_url: str


@dataclass(frozen=True)
class DownloadResult:
    source: str
    path: Path
    url: str
    size_bytes: int
    downloaded: bool


DATA_SOURCES = {
    "synthetic": DataSource(
        name="synthetic",
        files=(),
        terms="Generated locally and never downloaded or redistributed.",
        citation_url="",
    ),
    "movielens": DataSource(
        name="movielens",
        files=(
            SourceFile(
                name="ml-1m.zip",
                url="https://files.grouplens.org/datasets/movielens/ml-1m.zip",
                expected_bytes=5_917_549,
            ),
        ),
        terms=(
            "Research use with attribution; redistribution and commercial use require "
            "permission."
        ),
        citation_url="https://grouplens.org/datasets/movielens/1m/",
    ),
    "lastfm": DataSource(
        name="lastfm",
        files=(
            SourceFile(
                name="hetrec2011-lastfm-2k.zip",
                url=(
                    "https://files.grouplens.org/datasets/hetrec2011/"
                    "hetrec2011-lastfm-2k.zip"
                ),
                expected_bytes=2_589_075,
            ),
        ),
        terms="Non-commercial use; consult the archive README before use.",
        citation_url="https://grouplens.org/datasets/hetrec-2011/",
    ),
    "steam": DataSource(
        name="steam",
        files=(
            SourceFile(
                name="steam_reviews.json.gz",
                url="https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz",
                expected_bytes=1_338_063_248,
            ),
            SourceFile(
                name="steam_games.json.gz",
                url="https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
                expected_bytes=2_740_516,
            ),
        ),
        terms="Download from the official UCSD source and do not redistribute raw data.",
        citation_url="https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data",
    ),
    "goodreads-spoiler": DataSource(
        name="goodreads-spoiler",
        files=(
            SourceFile(
                name="goodreads_reviews_spoiler_raw.json.gz",
                url=(
                    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
                    "goodreads_reviews_spoiler_raw.json.gz"
                ),
                expected_bytes=660_370_149,
            ),
            SourceFile(
                name="goodreads_books.json.gz",
                url=(
                    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
                    "goodreads_books.json.gz"
                ),
                expected_bytes=2_083_197_934,
            ),
        ),
        terms="Academic use only; redistribution and commercial use are prohibited.",
        citation_url="https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html",
    ),
    "goodreads-full": DataSource(
        name="goodreads-full",
        files=(
            SourceFile(
                name="goodreads_interactions_dedup.json.gz",
                url=(
                    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
                    "goodreads_interactions_dedup.json.gz"
                ),
            ),
            SourceFile(
                name="goodreads_books.json.gz",
                url=(
                    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/"
                    "goodreads_books.json.gz"
                ),
                expected_bytes=2_083_197_934,
            ),
        ),
        terms="Academic use only; redistribution and commercial use are prohibited.",
        citation_url="https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html",
    ),
}


SOURCE_ALIASES = {
    "ml-1m": "movielens",
    "hetrec2011-lastfm-2k": "lastfm",
    "ucsd-steam-reviews": "steam",
    "goodreads": "goodreads-full",
    "goodreads-interactions": "goodreads-full",
    "goodreads-spoiler-raw": "goodreads-spoiler",
}


def resolve_source(name: str, variant: str | None = None) -> DataSource:
    requested = name.strip().lower()
    if requested == "goodreads" and variant:
        normalized_variant = variant.strip().lower()
        if normalized_variant in {"spoiler", "spoiler-raw", "reviews-spoiler"}:
            requested = "goodreads-spoiler"
        elif normalized_variant in {"full", "interactions", "detailed"}:
            requested = "goodreads-full"
        else:
            raise ValueError(f"Unsupported GoodReads variant: {variant}")
    requested = SOURCE_ALIASES.get(requested, requested)
    if requested not in DATA_SOURCES:
        choices = ", ".join(sorted(DATA_SOURCES))
        raise ValueError(f"Unsupported dataset source {name!r}; choose one of: {choices}")
    return DATA_SOURCES[requested]


def _download_file(
    source: DataSource,
    file_spec: SourceFile,
    destination: Path,
    force: bool,
    timeout: int,
    progress: Callable[[str, int], None] | None,
) -> DownloadResult:
    target = destination / file_spec.name
    if target.exists() and not force:
        size = target.stat().st_size
        if file_spec.expected_bytes is not None and size != file_spec.expected_bytes:
            raise ValueError(
                f"Existing file has unexpected size: {target} has {size}, "
                f"expected {file_spec.expected_bytes}"
            )
        return DownloadResult(source.name, target, file_spec.url, size, False)
    temporary = target.with_name(f".{target.name}.part")
    request = urllib.request.Request(file_spec.url, headers={"User-Agent": "frag/0.1"})
    written = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(file_spec.name, written)
        if file_spec.expected_bytes is not None and written != file_spec.expected_bytes:
            raise ValueError(
                f"Downloaded file has unexpected size: {written}, "
                f"expected {file_spec.expected_bytes}"
            )
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return DownloadResult(source.name, target, file_spec.url, written, True)


def download_dataset(
    name: str,
    destination: str | Path,
    variant: str | None = None,
    force: bool = False,
    timeout: int = 60,
    progress: Callable[[str, int], None] | None = None,
) -> list[DownloadResult]:
    source = resolve_source(name, variant)
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    return [
        _download_file(source, file_spec, target_dir, force, timeout, progress)
        for file_spec in source.files
    ]
