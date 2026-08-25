from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from frag.data.loaders import LoadedData, load_dataset


@dataclass
class PreparedData:
    interactions: pd.DataFrame
    items: pd.DataFrame
    groups: pd.DataFrame
    manifest: dict[str, Any]
    output_files: tuple[Path, ...] = ()


def _dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("dataset", config)
    if not isinstance(value, dict):
        raise TypeError("Dataset configuration must be a mapping")
    return value


def preparation_config(config: dict[str, Any]) -> dict[str, Any]:
    dataset = _dataset_config(config)
    selected = {
        key: value
        for key, value in dataset.items()
        if key not in {"raw_dir", "processed_dir"}
    }
    return json.loads(json.dumps(selected, sort_keys=True, default=str))


def _deduplicate(interactions: pd.DataFrame, policy: str) -> pd.DataFrame:
    normalized_policy = policy.strip().lower()
    ordered = interactions.sort_values(
        ["user_id", "timestamp", "_tie_key", "item_id", "_source_row"],
        kind="mergesort",
    )
    if normalized_policy == "keep":
        return ordered.reset_index(drop=True)
    if normalized_policy == "earliest":
        value = ordered.drop_duplicates(["user_id", "item_id"], keep="first")
    elif normalized_policy == "latest":
        value = ordered.drop_duplicates(["user_id", "item_id"], keep="last")
    else:
        raise ValueError(f"Unsupported duplicate policy: {policy}")
    return value.sort_values(
        ["user_id", "timestamp", "_tie_key", "item_id", "_source_row"],
        kind="mergesort",
    ).reset_index(drop=True)


def _filter_events(
    interactions: pd.DataFrame,
    minimum_user_events: int,
    minimum_item_events: int,
) -> pd.DataFrame:
    current = interactions
    while True:
        before = len(current)
        if minimum_user_events > 1:
            user_counts = current.groupby("user_id", sort=False).size()
            allowed_users = user_counts[user_counts >= minimum_user_events].index
            current = current[current["user_id"].isin(allowed_users)]
        if minimum_item_events > 1:
            item_counts = current.groupby("item_id", sort=False).size()
            allowed_items = item_counts[item_counts >= minimum_item_events].index
            current = current[current["item_id"].isin(allowed_items)]
        if len(current) == before:
            break
    counts = current.groupby("user_id", sort=False).size()
    splittable = counts[counts >= 3].index
    current = current[current["user_id"].isin(splittable)]
    return current.sort_values(
        ["user_id", "timestamp", "_tie_key", "item_id", "_source_row"],
        kind="mergesort",
    ).reset_index(drop=True)


def chronological_split(
    interactions: pd.DataFrame,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> pd.DataFrame:
    if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
        raise ValueError("Split ratios must be positive and leave a positive test ratio")
    ordered = interactions.sort_values(
        ["user_id", "timestamp", "_tie_key", "item_id", "_source_row"],
        kind="mergesort",
    ).reset_index(drop=True)
    split = np.empty(len(ordered), dtype=object)
    sequence_position = np.empty(len(ordered), dtype=np.int64)
    for _, indices in ordered.groupby("user_id", sort=False).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        count = len(positions)
        if count < 3:
            raise ValueError("Every retained user must have at least three chronological events")
        train_end = max(1, int(math.floor(count * train_ratio)))
        validation_end = int(math.floor(count * (train_ratio + validation_ratio)))
        validation_end = max(train_end + 1, validation_end)
        validation_end = min(count - 1, validation_end)
        train_end = min(train_end, validation_end - 1)
        split[positions[:train_end]] = "train"
        split[positions[train_end:validation_end]] = "validation"
        split[positions[validation_end:]] = "test"
        sequence_position[positions] = np.arange(count, dtype=np.int64)
    ordered["split"] = split
    ordered["sequence_position"] = sequence_position
    return ordered


def assign_frequency_groups(
    interactions: pd.DataFrame,
    item_ids: pd.Series | list[str] | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    catalog = pd.Series(item_ids, dtype="string").dropna().astype(str).drop_duplicates()
    catalog = catalog.sort_values(kind="mergesort").reset_index(drop=True)
    if catalog.empty:
        raise ValueError("Cannot create frequency groups for an empty catalog")
    train = interactions[interactions["split"] == "train"]
    frequencies = train.groupby("item_id", sort=False).size()
    table = pd.DataFrame({"item_id": catalog})
    table["training_frequency"] = (
        table["item_id"].map(frequencies).fillna(0).astype("int64")
    )
    ranked = table.sort_values(
        ["training_frequency", "item_id"], kind="mergesort"
    ).reset_index(drop=True)
    count = len(ranked)
    target = count / 2.0
    values = ranked["training_frequency"].to_numpy(dtype=np.int64)
    candidates = [index for index in range(1, count) if values[index - 1] != values[index]]
    if candidates:
        boundary_position = min(candidates, key=lambda index: (abs(index - target), index))
        lower_max = int(values[boundary_position - 1])
        upper_min = int(values[boundary_position])
        ranked["group_id"] = np.where(
            np.arange(count) < boundary_position, 0, 1
        ).astype("int8")
        degenerate = False
    else:
        boundary_position = count
        lower_max = int(values[0])
        upper_min = None
        ranked["group_id"] = np.zeros(count, dtype="int8")
        degenerate = True
    groups = ranked.sort_values("item_id", kind="mergesort").reset_index(drop=True)
    group_counts = groups.groupby("group_id", sort=True).size().to_dict()
    metadata = {
        "policy": "nearest_distinct_frequency_boundary_to_median_item_rank",
        "frequency_source": "train_only",
        "target_position": target,
        "boundary_position": int(boundary_position),
        "lower_max_frequency": lower_max,
        "upper_min_frequency": upper_min,
        "group_sizes": {str(key): int(value) for key, value in group_counts.items()},
        "degenerate": degenerate,
    }
    return groups, metadata


def _catalog_items(loaded: LoadedData, interactions: pd.DataFrame) -> pd.DataFrame:
    active_ids = interactions["item_id"].drop_duplicates()
    items = loaded.items[loaded.items["item_id"].isin(active_ids)].copy()
    missing = active_ids[~active_ids.isin(items["item_id"])]
    if not missing.empty:
        fallback = pd.DataFrame({"item_id": missing.astype(str), "title": missing.astype(str)})
        items = pd.concat([items, fallback], ignore_index=True)
    return items.drop_duplicates("item_id", keep="first").sort_values(
        "item_id", kind="mergesort"
    ).reset_index(drop=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.part")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.part")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_prepared(data: PreparedData, destination: Path) -> tuple[Path, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    interaction_path = destination / "interactions.parquet"
    item_path = destination / "items.parquet"
    group_path = destination / "groups.parquet"
    manifest_path = destination / "manifest.json"
    _atomic_parquet(data.interactions, interaction_path)
    _atomic_parquet(data.items, item_path)
    _atomic_parquet(data.groups, group_path)
    _atomic_json(data.manifest, manifest_path)
    return interaction_path, item_path, group_path, manifest_path


def prepare_dataset(config: dict[str, Any], write: bool = True) -> PreparedData:
    dataset = _dataset_config(config)
    loaded = load_dataset(config)
    duplicate_policy = str(dataset.get("duplicate_policy", "earliest"))
    deduplicated = _deduplicate(loaded.interactions, duplicate_policy)
    minimum_user_events = int(dataset.get("min_user_events", 3))
    minimum_item_events = int(dataset.get("min_item_events", 1))
    filtered = _filter_events(deduplicated, minimum_user_events, minimum_item_events)
    if filtered.empty:
        raise ValueError("No interactions remain after deterministic filtering")
    train_ratio = float(dataset.get("train_ratio", 0.8))
    validation_ratio = float(dataset.get("validation_ratio", 0.1))
    split = chronological_split(filtered, train_ratio, validation_ratio)
    items = _catalog_items(loaded, split)
    groups, group_metadata = assign_frequency_groups(split, items["item_id"])
    items = items.merge(groups, on="item_id", how="left", validate="one_to_one")
    public_interactions = split.rename(
        columns={"_source_row": "source_row", "_tie_key": "tie_key"}
    )
    public_interactions = public_interactions.reset_index(drop=True)
    split_counts = public_interactions["split"].value_counts().to_dict()
    manifest = {
        "dataset": str(dataset.get("name", loaded.source.get("name", "unknown"))),
        "source": loaded.source,
        "preparation_config": preparation_config(config),
        "rows": int(len(public_interactions)),
        "users": int(public_interactions["user_id"].nunique()),
        "items": int(len(items)),
        "split_counts": {
            name: int(split_counts.get(name, 0))
            for name in ("train", "validation", "test")
        },
        "split_policy": {
            "type": "chronological_per_user",
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": 1.0 - train_ratio - validation_ratio,
            "tie_order": ["timestamp", "tie_key", "item_id", "source_row"],
        },
        "duplicate_policy": duplicate_policy,
        "minimum_user_events": minimum_user_events,
        "minimum_item_events": minimum_item_events,
        "groups": group_metadata,
        "redistribution": "Raw and derived benchmark records must not be redistributed.",
    }
    prepared = PreparedData(
        interactions=public_interactions,
        items=items,
        groups=groups,
        manifest=manifest,
    )
    if write:
        processed_dir = Path(dataset["processed_dir"])
        prepared.output_files = _write_prepared(prepared, processed_dir)
    return prepared
