from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from frag.data.prepare import PreparedData


@dataclass
class PreparedTables:
    interactions: pd.DataFrame
    items: pd.DataFrame
    groups: pd.DataFrame
    manifest: dict[str, Any]
    user_to_index: dict[str, int]
    index_to_user: tuple[str, ...]
    item_to_index: dict[str, int]
    index_to_item: tuple[str | None, ...]
    titles: dict[int, str]
    item_groups: torch.Tensor

    @property
    def num_users(self) -> int:
        return len(self.index_to_user)

    @property
    def num_items(self) -> int:
        return len(self.index_to_item) - 1


@dataclass(frozen=True)
class ExampleRef:
    user_index: int
    user_id: str
    position: int
    timestamp: int
    target: int
    target_id: str
    split: str


def _tables_from_frames(
    interactions: pd.DataFrame,
    items: pd.DataFrame,
    groups: pd.DataFrame,
    manifest: dict[str, Any],
) -> PreparedTables:
    required_interactions = {
        "user_id",
        "item_id",
        "timestamp",
        "split",
        "sequence_position",
    }
    missing_interactions = sorted(required_interactions - set(interactions.columns))
    if missing_interactions:
        raise ValueError(f"Prepared interactions are missing columns: {missing_interactions}")
    required_items = {"item_id", "title", "group_id", "training_frequency"}
    missing_items = sorted(required_items - set(items.columns))
    if missing_items:
        raise ValueError(f"Prepared items are missing columns: {missing_items}")
    interaction_frame = interactions.copy()
    item_frame = items.copy()
    group_frame = groups.copy()
    interaction_frame["user_id"] = interaction_frame["user_id"].astype(str)
    interaction_frame["item_id"] = interaction_frame["item_id"].astype(str)
    item_frame["item_id"] = item_frame["item_id"].astype(str)
    group_frame["item_id"] = group_frame["item_id"].astype(str)
    user_ids = tuple(sorted(interaction_frame["user_id"].unique().tolist()))
    item_ids = tuple(sorted(item_frame["item_id"].unique().tolist()))
    user_to_index = {value: index for index, value in enumerate(user_ids)}
    item_to_index = {value: index + 1 for index, value in enumerate(item_ids)}
    interaction_frame["user_index"] = interaction_frame["user_id"].map(user_to_index)
    interaction_frame["item_index"] = interaction_frame["item_id"].map(item_to_index)
    if interaction_frame[["user_index", "item_index"]].isna().any().any():
        raise ValueError("Prepared interactions contain unmapped user or item identifiers")
    interaction_frame["user_index"] = interaction_frame["user_index"].astype("int64")
    interaction_frame["item_index"] = interaction_frame["item_index"].astype("int64")
    item_frame["item_index"] = item_frame["item_id"].map(item_to_index).astype("int64")
    group_lookup = group_frame.set_index("item_id")["group_id"]
    item_frame["group_id"] = item_frame["item_id"].map(group_lookup).fillna(
        item_frame["group_id"]
    )
    item_frame["group_id"] = item_frame["group_id"].astype("int64")
    groups_tensor = torch.zeros(len(item_ids) + 1, dtype=torch.long)
    titles: dict[int, str] = {}
    for row in item_frame.itertuples(index=False):
        item_index = int(row.item_index)
        groups_tensor[item_index] = int(row.group_id)
        titles[item_index] = str(row.title)
    interaction_frame = interaction_frame.sort_values(
        ["user_index", "sequence_position"], kind="mergesort"
    ).reset_index(drop=True)
    item_frame = item_frame.sort_values("item_index", kind="mergesort").reset_index(drop=True)
    return PreparedTables(
        interactions=interaction_frame,
        items=item_frame,
        groups=group_frame,
        manifest=manifest,
        user_to_index=user_to_index,
        index_to_user=user_ids,
        item_to_index=item_to_index,
        index_to_item=(None, *item_ids),
        titles=titles,
        item_groups=groups_tensor,
    )


def load_prepared(path: str | Path) -> PreparedTables:
    directory = Path(path)
    with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    interactions = pd.read_parquet(directory / "interactions.parquet")
    items = pd.read_parquet(directory / "items.parquet")
    groups = pd.read_parquet(directory / "groups.parquet")
    return _tables_from_frames(interactions, items, groups, manifest)


def prepared_tables(value: PreparedTables | PreparedData | str | Path) -> PreparedTables:
    if isinstance(value, PreparedTables):
        return value
    if isinstance(value, PreparedData):
        return _tables_from_frames(value.interactions, value.items, value.groups, value.manifest)
    return load_prepared(value)


def _ensure_target(config: dict[str, Any], split: str) -> bool:
    by_split = config.get("ensure_target_by_split")
    if isinstance(by_split, dict) and split in by_split:
        return bool(by_split[split])
    if split == "train":
        return bool(config.get("ensure_train_target", True))
    return bool(config.get("ensure_evaluation_target", False))


class SequenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        prepared: PreparedTables | PreparedData | str | Path,
        split: str | Sequence[str],
        min_history: int,
        candidate_pool: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if min_history < 1:
            raise ValueError("min_history must be positive")
        self.prepared = prepared_tables(prepared)
        self.min_history = int(min_history)
        self.candidate_pool = dict(candidate_pool or {})
        if isinstance(split, str):
            selected_splits = {"train", "validation", "test"} if split == "all" else {split}
        else:
            selected_splits = {str(value) for value in split}
        invalid_splits = selected_splits - {"train", "validation", "test"}
        if invalid_splits:
            raise ValueError(f"Unsupported splits: {sorted(invalid_splits)}")
        self.selected_splits = selected_splits
        self.domain = str(self.prepared.manifest.get("dataset", "unknown"))
        self._sequences: dict[int, np.ndarray] = {}
        examples: list[ExampleRef] = []
        for user_index, frame in self.prepared.interactions.groupby("user_index", sort=True):
            ordered = frame.sort_values("sequence_position", kind="mergesort")
            expected = np.arange(len(ordered), dtype=np.int64)
            positions = ordered["sequence_position"].to_numpy(dtype=np.int64)
            if not np.array_equal(positions, expected):
                raise ValueError(f"Non-contiguous sequence positions for user {user_index}")
            sequence = ordered["item_index"].to_numpy(dtype=np.int64)
            numeric_user = int(user_index)
            self._sequences[numeric_user] = sequence
            for row in ordered.itertuples(index=False):
                position = int(row.sequence_position)
                row_split = str(row.split)
                if position < self.min_history or row_split not in selected_splits:
                    continue
                examples.append(
                    ExampleRef(
                        user_index=numeric_user,
                        user_id=str(row.user_id),
                        position=position,
                        timestamp=int(row.timestamp),
                        target=int(row.item_index),
                        target_id=str(row.item_id),
                        split=row_split,
                    )
                )
        self.examples = tuple(
            sorted(examples, key=lambda value: (value.position, value.user_index, value.timestamp))
        )
        mode = str(self.candidate_pool.get("mode", "full_catalog")).lower()
        if mode not in {
            "full_catalog",
            "frequency",
            "frequency_pool",
            "deterministic_frequency",
            "stratified_frequency",
        }:
            raise ValueError(f"Unsupported candidate pool mode: {mode}")
        self.candidate_mode = mode
        item_order = self.prepared.items.sort_values(
            ["training_frequency", "item_index"],
            ascending=[False, True],
            kind="mergesort",
        )
        self._frequency_order = item_order["item_index"].astype(int).tolist()
        self._group_orders = [
            frame.sort_values(
                ["training_frequency", "item_index"],
                ascending=[False, True],
                kind="mergesort",
            )["item_index"].astype(int).tolist()
            for _, frame in self.prepared.items.groupby("group_id", sort=True)
        ]
        self._catalog_order = list(range(1, self.prepared.num_items + 1))

    def __len__(self) -> int:
        return len(self.examples)

    def _candidates(self, example: ExampleRef, history: np.ndarray) -> list[int]:
        if self.candidate_mode == "full_catalog":
            ordered = self._catalog_order
            size = None
        elif self.candidate_mode == "stratified_frequency":
            configured_size = self.candidate_pool.get("size")
            size = int(configured_size) if configured_size is not None else len(self._catalog_order)
            if size < 1:
                raise ValueError("Candidate pool size must be positive")
            buckets = self._group_orders
            ordered = [
                bucket[position]
                for position in range(max((len(bucket) for bucket in buckets), default=0))
                for bucket in buckets
                if position < len(bucket)
            ]
        else:
            ordered = self._frequency_order
            configured_size = self.candidate_pool.get("size")
            size = int(configured_size) if configured_size is not None else len(ordered)
            if size < 1:
                raise ValueError("Candidate pool size must be positive")
        if bool(self.candidate_pool.get("exclude_history", True)):
            history_items = set(int(value) for value in history.tolist())
            candidates = [value for value in ordered if value not in history_items]
        else:
            candidates = list(ordered)
        if size is not None:
            candidates = candidates[:size]
        if _ensure_target(self.candidate_pool, example.split) and example.target not in candidates:
            if size is not None and len(candidates) >= size:
                candidates[-1] = example.target
            else:
                candidates.append(example.target)
        candidates = list(dict.fromkeys(candidates))
        return candidates

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        sequence = self._sequences[example.user_index]
        history = sequence[: example.position].copy()
        candidates = self._candidates(example, history)
        return {
            "user_id": example.user_index,
            "timestamp": example.position,
            "event_timestamp": example.timestamp,
            "history": history.tolist(),
            "candidates": candidates,
            "target": example.target,
            "domain": self.domain,
            "split": example.split,
            "user_key": example.user_id,
            "target_key": example.target_id,
        }


class TemporalBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: SequenceDataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rounds(self) -> list[list[int]]:
        per_user: dict[int, list[int]] = {}
        for index, example in enumerate(self.dataset.examples):
            per_user.setdefault(example.user_index, []).append(index)
        for indices in per_user.values():
            indices.sort(key=lambda value: self.dataset.examples[value].position)
        maximum = max((len(indices) for indices in per_user.values()), default=0)
        rng = np.random.default_rng(self.seed + self.epoch)
        rounds = []
        for round_index in range(maximum):
            active = [
                indices[round_index]
                for indices in per_user.values()
                if round_index < len(indices)
            ]
            active.sort(key=lambda value: self.dataset.examples[value].user_index)
            if self.shuffle and active:
                permutation = rng.permutation(len(active))
                active = [active[int(index)] for index in permutation]
            rounds.append(active)
        return rounds

    def __iter__(self) -> Iterator[list[int]]:
        for active in self._rounds():
            for start in range(0, len(active), self.batch_size):
                batch = active[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for active in self._rounds():
            if self.drop_last:
                total += len(active) // self.batch_size
            else:
                total += math.ceil(len(active) / self.batch_size)
        return total


def sequence_collate(examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(examples)
    if not rows:
        raise ValueError("Cannot collate an empty batch")
    history_size = max(len(row["history"]) for row in rows)
    candidate_size = max(len(row["candidates"]) for row in rows)
    if candidate_size < 1:
        raise ValueError("Every example must have at least one candidate")
    batch_size = len(rows)
    histories = torch.zeros((batch_size, history_size), dtype=torch.long)
    history_mask = torch.zeros((batch_size, history_size), dtype=torch.bool)
    candidates = torch.zeros((batch_size, candidate_size), dtype=torch.long)
    candidate_mask = torch.zeros((batch_size, candidate_size), dtype=torch.bool)
    for index, row in enumerate(rows):
        history = torch.as_tensor(row["history"], dtype=torch.long)
        candidate = torch.as_tensor(row["candidates"], dtype=torch.long)
        histories[index, : history.numel()] = history
        history_mask[index, : history.numel()] = True
        candidates[index, : candidate.numel()] = candidate
        candidate_mask[index, : candidate.numel()] = True
    return {
        "user_ids": torch.tensor([int(row["user_id"]) for row in rows], dtype=torch.long),
        "timestamps": torch.tensor([int(row["timestamp"]) for row in rows], dtype=torch.long),
        "histories": histories,
        "history_mask": history_mask,
        "candidates": candidates,
        "candidate_mask": candidate_mask,
        "targets": torch.tensor([int(row["target"]) for row in rows], dtype=torch.long),
        "domains": [str(row["domain"]) for row in rows],
    }


def build_dataloader(
    prepared: PreparedTables | PreparedData | str | Path,
    config: dict[str, Any],
    split: str | Sequence[str],
    batch_size: int | None = None,
    shuffle: bool | None = None,
) -> DataLoader:
    dataset_config = config.get("dataset", {})
    candidate_config = config.get("candidate_pool", {})
    training_config = config.get("training", {})
    runtime_config = config.get("runtime", {})
    min_history = int(dataset_config.get("min_history", 3))
    dataset = SequenceDataset(prepared, split, min_history, candidate_config)
    selected_batch_size = int(
        batch_size if batch_size is not None else training_config.get("micro_batch_size", 1)
    )
    selected_shuffle = bool(split == "train") if shuffle is None else bool(shuffle)
    sampler = TemporalBatchSampler(
        dataset,
        selected_batch_size,
        shuffle=selected_shuffle,
        seed=int(training_config.get("seed", 0)),
        drop_last=bool(training_config.get("drop_last", False)),
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=sequence_collate,
        num_workers=int(runtime_config.get("num_workers", 0)),
        pin_memory=bool(runtime_config.get("pin_memory", False)),
    )
