from __future__ import annotations

import ast
import gzip
import io
import json
import math
import re
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd


@dataclass
class LoadedData:
    interactions: pd.DataFrame
    items: pd.DataFrame
    source: dict[str, Any]


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("dataset", config)
    if not isinstance(value, dict):
        raise TypeError("Dataset configuration must be a mapping")
    return value


def _find_file(raw_dir: Path, requested: str | None, fallbacks: tuple[str, ...]) -> Path:
    names = tuple(value for value in (requested, *fallbacks) if value)
    for name in names:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
        matches = sorted(path for path in raw_dir.rglob(Path(name).name) if path.is_file())
        if matches:
            return matches[0]
    joined = ", ".join(names)
    raise FileNotFoundError(f"None of the required files were found in {raw_dir}: {joined}")


def _find_zip(raw_dir: Path, preferred: tuple[str, ...]) -> Path:
    for name in preferred:
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    archives = sorted(path for path in raw_dir.rglob("*.zip") if path.is_file())
    if not archives:
        raise FileNotFoundError(f"No supported archive found in {raw_dir}")
    return archives[0]


@contextmanager
def _open_text(
    raw_dir: Path,
    requested: str,
    fallbacks: tuple[str, ...] = (),
    archives: tuple[str, ...] = (),
) -> Iterator[TextIO]:
    try:
        path = _find_file(raw_dir, requested, fallbacks)
    except FileNotFoundError as error:
        archive = _find_zip(raw_dir, archives)
        with zipfile.ZipFile(archive) as handle:
            members = sorted(
                name
                for name in handle.namelist()
                if Path(name).name in {Path(requested).name, *(Path(x).name for x in fallbacks)}
            )
            if not members:
                raise FileNotFoundError(
                    f"Required member {requested} not found in {archive}"
                ) from error
            data = handle.read(members[0]).decode("utf-8", errors="replace")
        stream = io.StringIO(data)
        try:
            yield stream
        finally:
            stream.close()
        return
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield handle
        return
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield handle


def _parse_mapping(raw: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Invalid JSON record at {path}:{line_number}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping at {path}:{line_number}")
    return value


def _iter_mappings(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        handle = path.open("r", encoding="utf-8", errors="replace")
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            stripped = raw.strip()
            if stripped:
                yield line_number, _parse_mapping(stripped, path, line_number)


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def _timestamp_value(value: Any) -> int:
    if value is None or isinstance(value, bool):
        raise ValueError("Timestamp is missing")
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not math.isfinite(float(value)):
            raise ValueError(f"Invalid timestamp: {value}")
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("Timestamp is empty")
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return int(float(text))
    if text.lower().startswith("posted "):
        text = text[7:].strip()
    text = text.rstrip(".")
    try:
        parsed = pd.Timestamp(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid timestamp: {value}") from error
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    else:
        parsed = parsed.tz_convert("UTC")
    return int(parsed.value)


def _tie_key(value: Any, source_row: int) -> str:
    if value is None or value == "":
        return f"row:{source_row:020d}"
    return f"value:{str(value)}:row:{source_row:020d}"


def _normalize_loaded(data: LoadedData) -> LoadedData:
    interactions = data.interactions.copy()
    required = {"user_id", "item_id", "timestamp", "_source_row", "_tie_key"}
    missing = sorted(required - set(interactions.columns))
    if missing:
        raise ValueError(f"Missing normalized interaction columns: {missing}")
    interactions = interactions.dropna(subset=["user_id", "item_id", "timestamp"])
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    interactions["timestamp"] = interactions["timestamp"].map(_timestamp_value).astype("int64")
    interactions["_source_row"] = interactions["_source_row"].astype("int64")
    interactions["_tie_key"] = interactions["_tie_key"].astype(str)
    interactions = interactions.sort_values(
        ["user_id", "timestamp", "_tie_key", "item_id", "_source_row"],
        kind="mergesort",
    ).reset_index(drop=True)
    items = data.items.copy()
    if "item_id" not in items.columns:
        raise ValueError("Missing normalized item column: item_id")
    if "title" not in items.columns:
        items["title"] = items["item_id"]
    items = items.dropna(subset=["item_id"])
    items["item_id"] = items["item_id"].astype(str)
    items["title"] = items["title"].fillna(items["item_id"]).astype(str)
    items = items.drop_duplicates("item_id", keep="first")
    items = items.sort_values("item_id", kind="mergesort").reset_index(drop=True)
    return LoadedData(interactions=interactions, items=items, source=data.source)


def load_synthetic(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    users = int(dataset.get("synthetic_users", 12))
    items_count = int(dataset.get("synthetic_items", 32))
    events_per_user = int(dataset.get("synthetic_events_per_user", 20))
    seed = int(dataset.get("synthetic_seed", 0))
    if users < 1 or items_count < 2 or events_per_user < 3:
        raise ValueError("Synthetic data requires users>=1, items>=2, and events_per_user>=3")
    rng = np.random.default_rng(seed)
    rows = []
    source_row = 0
    for user_index in range(users):
        offset = int(rng.integers(0, items_count))
        for position in range(events_per_user):
            item_index = (offset + position + position // 4 + user_index * 3) % items_count
            rows.append(
                {
                    "user_id": f"u{user_index:04d}",
                    "item_id": f"i{item_index:05d}",
                    "timestamp": user_index * 1_000_000 + position,
                    "_source_row": source_row,
                    "_tie_key": _tie_key(None, source_row),
                }
            )
            source_row += 1
    item_rows = [
        {"item_id": f"i{index:05d}", "title": f"Synthetic item {index}"}
        for index in range(items_count)
    ]
    return _normalize_loaded(
        LoadedData(
            interactions=pd.DataFrame.from_records(rows),
            items=pd.DataFrame.from_records(item_rows),
            source={"name": "synthetic", "variant": "deterministic", "seed": seed},
        )
    )


def load_movielens(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    raw_dir = Path(dataset["raw_dir"])
    interaction_file = str(dataset.get("interaction_file", "ratings.dat"))
    item_file = str(dataset.get("item_file", "movies.dat"))
    with _open_text(
        raw_dir,
        interaction_file,
        ("ratings.dat",),
        ("ml-1m.zip",),
    ) as handle:
        interactions = pd.read_csv(
            handle,
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"],
            header=None,
        )
    interactions["_source_row"] = np.arange(len(interactions), dtype=np.int64)
    interactions["_tie_key"] = [
        _tie_key(None, int(value)) for value in interactions["_source_row"]
    ]
    with _open_text(raw_dir, item_file, ("movies.dat",), ("ml-1m.zip",)) as handle:
        items = pd.read_csv(
            handle,
            sep="::",
            engine="python",
            names=["item_id", "title", "genres"],
            header=None,
        )
    return _normalize_loaded(
        LoadedData(
            interactions=interactions,
            items=items,
            source={"name": "movielens", "variant": "ml-1m"},
        )
    )


def load_lastfm(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    raw_dir = Path(dataset["raw_dir"])
    policy = str(
        dataset.get("lastfm_event_policy", dataset.get("event_policy", "earliest_tag_event"))
    )
    if policy != "earliest_tag_event":
        raise ValueError(
            "HetRec listening relations have no timestamps; supported policy is earliest_tag_event"
        )
    interaction_file = str(
        dataset.get("interaction_file", "user_taggedartists-timestamps.dat")
    )
    with _open_text(
        raw_dir,
        interaction_file,
        ("user_taggedartists-timestamps.dat",),
        ("hetrec2011-lastfm-2k.zip",),
    ) as handle:
        interactions = pd.read_csv(handle, sep="\t")
    rename = {"userID": "user_id", "artistID": "item_id"}
    interactions = interactions.rename(columns=rename)
    required = {"user_id", "item_id", "timestamp"}
    if not required.issubset(interactions.columns):
        raise ValueError(f"Invalid HetRec timestamp file; required columns are {sorted(required)}")
    interactions["_source_row"] = np.arange(len(interactions), dtype=np.int64)
    interactions["_tie_key"] = [
        _tie_key(value, int(row))
        for value, row in zip(
            interactions.get("tagID", pd.Series([None] * len(interactions))),
            interactions["_source_row"],
            strict=True,
        )
    ]
    interactions = interactions.sort_values(
        ["user_id", "item_id", "timestamp", "_tie_key", "_source_row"],
        kind="mergesort",
    ).drop_duplicates(["user_id", "item_id"], keep="first")
    item_file = str(dataset.get("item_file", "artists.dat"))
    with _open_text(
        raw_dir,
        item_file,
        ("artists.dat",),
        ("hetrec2011-lastfm-2k.zip",),
    ) as handle:
        items = pd.read_csv(handle, sep="\t").rename(columns={"id": "item_id", "name": "title"})
    return _normalize_loaded(
        LoadedData(
            interactions=interactions,
            items=items,
            source={
                "name": "lastfm",
                "variant": "hetrec2011-lastfm-2k",
                "event_policy": policy,
            },
        )
    )


def _steam_events(path: Path) -> tuple[pd.DataFrame, list[dict[str, str]], int]:
    rows = []
    discovered_items: list[dict[str, str]] = []
    dropped = 0
    source_row = 0
    for _, outer in _iter_mappings(path):
        nested = outer.get("reviews")
        records = nested if isinstance(nested, list) else [outer]
        outer_user = _first(outer, ("user_id", "username", "user"))
        for record in records:
            if not isinstance(record, dict):
                dropped += 1
                continue
            user_id = _first(record, ("user_id", "username", "user")) or outer_user
            item_id = _first(record, ("item_id", "product_id", "app_id", "id"))
            timestamp = _first(
                record,
                (
                    "timestamp",
                    "unixReviewTime",
                    "unix_timestamp",
                    "date",
                    "posted",
                    "review_date",
                    "time",
                ),
            )
            if user_id is None or item_id is None or timestamp is None:
                dropped += 1
                continue
            try:
                normalized_timestamp = _timestamp_value(timestamp)
            except ValueError:
                dropped += 1
                continue
            tie_value = _first(record, ("review_id", "recommendationid", "page_order"))
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": normalized_timestamp,
                    "_source_row": source_row,
                    "_tie_key": _tie_key(tie_value, source_row),
                }
            )
            title = _first(record, ("product_title", "app_name", "title", "name"))
            if title is not None:
                discovered_items.append({"item_id": str(item_id), "title": str(title)})
            source_row += 1
    return pd.DataFrame.from_records(rows), discovered_items, dropped


def _metadata_items(
    path: Path,
    id_keys: tuple[str, ...],
    title_keys: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for _, record in _iter_mappings(path):
        item_id = _first(record, id_keys)
        if item_id is None:
            continue
        title = _first(record, title_keys)
        rows.append({"item_id": item_id, "title": title if title is not None else item_id})
    return pd.DataFrame.from_records(rows, columns=["item_id", "title"])


def load_steam(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    raw_dir = Path(dataset["raw_dir"])
    interaction_path = _find_file(
        raw_dir,
        str(dataset.get("interaction_file", "steam_reviews.json.gz")),
        ("steam_reviews.json.gz", "australian_user_reviews.json.gz"),
    )
    interactions, discovered_items, dropped = _steam_events(interaction_path)
    item_path = _find_file(
        raw_dir,
        str(dataset.get("item_file", "steam_games.json.gz")),
        ("steam_games.json.gz",),
    )
    items = _metadata_items(
        item_path,
        ("id", "item_id", "product_id", "app_id"),
        ("app_name", "title", "name"),
    )
    if discovered_items:
        items = pd.concat([items, pd.DataFrame.from_records(discovered_items)], ignore_index=True)
    return _normalize_loaded(
        LoadedData(
            interactions=interactions,
            items=items,
            source={
                "name": "steam",
                "variant": "ucsd-steam-reviews",
                "interaction_file": interaction_path.name,
                "dropped_incomplete_events": dropped,
            },
        )
    )


def _goodreads_events(
    path: Path,
    variant: str,
    configured_time: str | None,
) -> tuple[pd.DataFrame, int]:
    rows = []
    dropped = 0
    source_row = 0
    if variant == "spoiler":
        time_keys = (configured_time,) if configured_time else ("timestamp", "date_added")
    else:
        time_keys = (
            (configured_time,)
            if configured_time
            else ("date_added", "read_at", "started_at", "date_updated", "timestamp")
        )
    for _, record in _iter_mappings(path):
        user_id = _first(record, ("user_id", "userID"))
        item_id = _first(record, ("book_id", "item_id", "bookID"))
        timestamp = _first(record, time_keys)
        if user_id is None or item_id is None or timestamp is None:
            dropped += 1
            continue
        try:
            normalized_timestamp = _timestamp_value(timestamp)
        except ValueError:
            dropped += 1
            continue
        tie_value = _first(record, ("review_id", "interaction_id"))
        row = {
            "user_id": user_id,
            "item_id": item_id,
            "timestamp": normalized_timestamp,
            "_source_row": source_row,
            "_tie_key": _tie_key(tie_value, source_row),
        }
        if record.get("rating") is not None:
            row["rating"] = record["rating"]
        rows.append(row)
        source_row += 1
    return pd.DataFrame.from_records(rows), dropped


def load_goodreads(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    raw_dir = Path(dataset["raw_dir"])
    source_variant = str(dataset.get("source_variant", "goodreads-interactions")).lower()
    requested = str(dataset.get("interaction_file", ""))
    spoiler = "spoiler" in source_variant or "spoiler" in requested
    variant = "spoiler" if spoiler else "full-detailed"
    if spoiler:
        interaction_path = _find_file(
            raw_dir,
            requested or "goodreads_reviews_spoiler_raw.json.gz",
            ("goodreads_reviews_spoiler_raw.json.gz", "goodreads_reviews_spoiler.json.gz"),
        )
    else:
        interaction_path = _find_file(
            raw_dir,
            requested or "goodreads_interactions_dedup.json.gz",
            ("goodreads_interactions_dedup.json.gz",),
        )
        if interaction_path.suffix == ".csv":
            raise ValueError(
                "The compact GoodReads interaction CSV has no event timestamp; "
                "use goodreads_interactions_dedup.json.gz"
            )
    configured_time = dataset.get("goodreads_time_column")
    interactions, dropped = _goodreads_events(
        interaction_path,
        "spoiler" if spoiler else "full",
        str(configured_time) if configured_time else None,
    )
    item_path = _find_file(
        raw_dir,
        str(dataset.get("item_file", "goodreads_books.json.gz")),
        ("goodreads_books.json.gz",),
    )
    items = _metadata_items(
        item_path,
        ("book_id", "item_id", "bookID"),
        ("title", "name"),
    )
    return _normalize_loaded(
        LoadedData(
            interactions=interactions,
            items=items,
            source={
                "name": "goodreads",
                "variant": variant,
                "interaction_file": interaction_path.name,
                "dropped_incomplete_events": dropped,
            },
        )
    )


def load_normalized_csv(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    raw_dir = Path(dataset["raw_dir"])
    interaction_path = _find_file(raw_dir, str(dataset["interaction_file"]), ())
    interactions = pd.read_csv(interaction_path)
    user_column = str(dataset.get("user_column", "user_id"))
    item_column = str(dataset.get("item_column", "item_id"))
    time_column = str(dataset.get("time_column", "timestamp"))
    missing = sorted({user_column, item_column, time_column} - set(interactions.columns))
    if missing:
        raise ValueError(f"Generic CSV is missing configured columns: {missing}")
    interactions = interactions.rename(
        columns={user_column: "user_id", item_column: "item_id", time_column: "timestamp"}
    )
    interactions["_source_row"] = np.arange(len(interactions), dtype=np.int64)
    configured_tie = dataset.get("tie_column")
    tie_values = (
        interactions[str(configured_tie)]
        if configured_tie and str(configured_tie) in interactions.columns
        else pd.Series([None] * len(interactions))
    )
    interactions["_tie_key"] = [
        _tie_key(value, int(row))
        for value, row in zip(tie_values, interactions["_source_row"], strict=True)
    ]
    requested_item = dataset.get("item_file")
    if requested_item:
        item_path = _find_file(raw_dir, str(requested_item), ())
        items = pd.read_csv(item_path)
        item_id_column = str(dataset.get("item_id_column", item_column))
        title_column = str(dataset.get("title_column", "title"))
        if item_id_column not in items.columns:
            raise ValueError(f"Generic item CSV is missing {item_id_column}")
        items = items.rename(columns={item_id_column: "item_id"})
        if title_column in items.columns:
            items = items.rename(columns={title_column: "title"})
    else:
        items = pd.DataFrame({"item_id": interactions["item_id"].drop_duplicates()})
    return _normalize_loaded(
        LoadedData(
            interactions=interactions,
            items=items,
            source={"name": "generic", "variant": "normalized-csv"},
        )
    )


def load_dataset(config: dict[str, Any]) -> LoadedData:
    dataset = _dataset_config(config)
    name = str(dataset.get("name", "synthetic")).strip().lower()
    variant = str(dataset.get("source_variant", "")).strip().lower()
    if name == "synthetic":
        return load_synthetic(config)
    if name in {"movielens", "movie", "ml-1m"} or variant == "ml-1m":
        return load_movielens(config)
    if name in {"lastfm", "last.fm"} or variant == "hetrec2011-lastfm-2k":
        return load_lastfm(config)
    if name == "steam" or variant == "ucsd-steam-reviews":
        return load_steam(config)
    if name in {"goodreads", "good-reads"} or variant.startswith("goodreads"):
        return load_goodreads(config)
    if name in {"generic", "csv", "normalized-csv"}:
        return load_normalized_csv(config)
    raise ValueError(f"Unsupported dataset loader: {name}")
