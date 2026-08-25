from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from frag.data.download import resolve_source
from frag.data.loaders import load_dataset
from frag.data.prepare import PreparedData, assign_frequency_groups, prepare_dataset
from frag.data.runtime import (
    SequenceDataset,
    TemporalBatchSampler,
    build_dataloader,
    load_prepared,
    prepared_tables,
    sequence_collate,
)


def _base_dataset(tmp_path: Path, name: str) -> dict[str, object]:
    return {
        "name": name,
        "raw_dir": str(tmp_path / "raw"),
        "processed_dir": str(tmp_path / "processed"),
        "min_user_events": 3,
        "min_item_events": 1,
        "train_ratio": 0.8,
        "validation_ratio": 0.1,
        "duplicate_policy": "keep",
    }


def _write_gzip_json(path: Path, records: list[dict[str, object]], literal: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(repr(record) if literal else json.dumps(record))
            handle.write("\n")


def test_synthetic_chronological_811_split(tmp_path: Path) -> None:
    dataset = _base_dataset(tmp_path, "synthetic")
    dataset.update(
        {
            "synthetic_users": 3,
            "synthetic_items": 40,
            "synthetic_events_per_user": 10,
            "synthetic_seed": 7,
        }
    )
    prepared = prepare_dataset({"dataset": dataset}, write=False)
    counts = prepared.interactions.groupby(["user_id", "split"]).size().unstack(fill_value=0)
    assert counts[["train", "validation", "test"]].values.tolist() == [
        [8, 1, 1],
        [8, 1, 1],
        [8, 1, 1],
    ]
    for _, frame in prepared.interactions.groupby("user_id", sort=False):
        assert frame["timestamp"].tolist() == sorted(frame["timestamp"].tolist())
        assert frame["sequence_position"].tolist() == list(range(10))
    assert prepared.manifest["groups"]["frequency_source"] == "train_only"


def test_generic_csv_ties_are_deterministic(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interactions = pd.DataFrame(
        {
            "user": ["u"] * 10,
            "item": ["b", "a", "c", "d", "e", "f", "g", "h", "i", "j"],
            "time": [1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "tie": ["b", "a", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )
    items = pd.DataFrame(
        {"item": list("abcdefghij"), "name": [f"title-{x}" for x in "abcdefghij"]}
    )
    interactions.to_csv(raw_dir / "interactions.csv", index=False)
    items.to_csv(raw_dir / "items.csv", index=False)
    dataset = _base_dataset(tmp_path, "generic")
    dataset.update(
        {
            "interaction_file": "interactions.csv",
            "item_file": "items.csv",
            "user_column": "user",
            "item_column": "item",
            "time_column": "time",
            "tie_column": "tie",
            "item_id_column": "item",
            "title_column": "name",
        }
    )
    first = prepare_dataset({"dataset": dataset}, write=False)
    second = prepare_dataset({"dataset": dataset}, write=False)
    assert first.interactions["item_id"].tolist()[:2] == ["a", "b"]
    pd.testing.assert_frame_equal(first.interactions, second.interactions)
    assert first.items.set_index("item_id").loc["a", "title"] == "title-a"


def test_frequency_groups_never_split_equal_frequencies() -> None:
    counts = {"a": 1, "b": 1, "c": 2, "d": 2, "e": 2, "f": 8, "g": 9}
    rows = []
    for item_id, count in counts.items():
        rows.extend({"item_id": item_id, "split": "train"} for _ in range(count))
    interactions = pd.DataFrame.from_records(rows)
    groups, metadata = assign_frequency_groups(interactions, list(counts))
    by_frequency = groups.groupby("training_frequency")["group_id"].nunique()
    assert by_frequency.max() == 1
    assert metadata["boundary_position"] == 2
    assert metadata["lower_max_frequency"] == 1
    assert metadata["upper_min_frequency"] == 2
    assert set(groups.loc[groups["group_id"] == 0, "item_id"]) == {"a", "b"}


def test_movielens_archive_loader(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with zipfile.ZipFile(raw_dir / "ml-1m.zip", "w") as archive:
        archive.writestr(
            "ml-1m/ratings.dat",
            "1::2::5::20\n1::1::4::10\n1::3::3::30\n",
        )
        archive.writestr(
            "ml-1m/movies.dat",
            "1::First (2000)::Drama\n2::Second (2001)::Comedy\n3::Third (2002)::Action\n",
        )
    dataset = _base_dataset(tmp_path, "movielens")
    dataset.update({"interaction_file": "ratings.dat", "item_file": "movies.dat"})
    loaded = load_dataset({"dataset": dataset})
    assert loaded.interactions["item_id"].tolist() == ["1", "2", "3"]
    assert loaded.items["title"].tolist() == ["First (2000)", "Second (2001)", "Third (2002)"]


def test_lastfm_uses_earliest_tag_timestamp(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with zipfile.ZipFile(raw_dir / "hetrec2011-lastfm-2k.zip", "w") as archive:
        archive.writestr(
            "user_taggedartists-timestamps.dat",
            "userID\tartistID\ttagID\ttimestamp\n1\t7\t2\t20\n1\t7\t1\t10\n1\t8\t3\t30\n",
        )
        archive.writestr(
            "artists.dat",
            "id\tname\turl\tpictureURL\n7\tSeven\turl\tpic\n8\tEight\turl\tpic\n",
        )
        archive.writestr("user_artists.dat", "userID\tartistID\tweight\n1\t7\t99\n")
    dataset = _base_dataset(tmp_path, "lastfm")
    dataset.update(
        {
            "interaction_file": "user_taggedartists-timestamps.dat",
            "item_file": "artists.dat",
            "lastfm_event_policy": "earliest_tag_event",
        }
    )
    loaded = load_dataset({"dataset": dataset})
    selected = loaded.interactions[loaded.interactions["item_id"] == "7"]
    assert selected["timestamp"].tolist() == [10]
    dataset["lastfm_event_policy"] = "listening_count"
    with pytest.raises(ValueError, match="no timestamps"):
        load_dataset({"dataset": dataset})


def test_steam_nested_review_loader(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_gzip_json(
        raw_dir / "australian_user_reviews.json.gz",
        [
            {
                "user_id": "u1",
                "reviews": [
                    {"item_id": "11", "posted": "Posted January 2, 2014."},
                    {"item_id": "12", "posted": "Posted January 3, 2014."},
                ],
            }
        ],
        literal=True,
    )
    _write_gzip_json(
        raw_dir / "steam_games.json.gz",
        [{"id": "11", "app_name": "Game A"}, {"id": "12", "app_name": "Game B"}],
        literal=True,
    )
    dataset = _base_dataset(tmp_path, "steam")
    dataset.update(
        {
            "interaction_file": "australian_user_reviews.json.gz",
            "item_file": "steam_games.json.gz",
        }
    )
    loaded = load_dataset({"dataset": dataset})
    assert loaded.interactions["item_id"].tolist() == ["11", "12"]
    assert loaded.interactions["timestamp"].is_monotonic_increasing
    assert loaded.items.set_index("item_id").loc["11", "title"] == "Game A"


def test_goodreads_spoiler_and_full_detailed_loaders(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_gzip_json(
        raw_dir / "goodreads_reviews_spoiler_raw.json.gz",
        [
            {"user_id": "u", "book_id": "1", "timestamp": "2013-01-01", "rating": 4},
            {"user_id": "u", "book_id": "2", "timestamp": "2013-01-02", "rating": 5},
            {"user_id": "u", "book_id": "3", "timestamp": "2013-01-03", "rating": 3},
        ],
    )
    _write_gzip_json(
        raw_dir / "goodreads_interactions_dedup.json.gz",
        [
            {"user_id": "v", "book_id": "1", "date_added": "2012-01-01"},
            {"user_id": "v", "book_id": "2", "date_added": "2012-01-02"},
            {"user_id": "v", "book_id": "3", "date_added": "2012-01-03"},
        ],
    )
    _write_gzip_json(
        raw_dir / "goodreads_books.json.gz",
        [
            {"book_id": "1", "title": "Book A"},
            {"book_id": "2", "title": "Book B"},
            {"book_id": "3", "title": "Book C"},
        ],
    )
    spoiler = _base_dataset(tmp_path, "goodreads")
    spoiler.update(
        {
            "source_variant": "goodreads-spoiler",
            "interaction_file": "goodreads_reviews_spoiler_raw.json.gz",
            "item_file": "goodreads_books.json.gz",
        }
    )
    spoiler_loaded = load_dataset({"dataset": spoiler})
    assert spoiler_loaded.source["variant"] == "spoiler"
    assert spoiler_loaded.interactions["rating"].tolist() == [4, 5, 3]
    full = dict(spoiler)
    full.update(
        {
            "source_variant": "goodreads-interactions",
            "interaction_file": "goodreads_interactions_dedup.json.gz",
        }
    )
    full_loaded = load_dataset({"dataset": full})
    assert full_loaded.source["variant"] == "full-detailed"
    assert full_loaded.interactions["user_id"].tolist() == ["v", "v", "v"]


def test_goodreads_compact_csv_is_rejected(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pd.DataFrame({"user_id": ["u"], "book_id": ["b"]}).to_csv(
        raw_dir / "goodreads_interactions.csv", index=False
    )
    _write_gzip_json(raw_dir / "goodreads_books.json.gz", [{"book_id": "b", "title": "B"}])
    dataset = _base_dataset(tmp_path, "goodreads")
    dataset.update(
        {
            "source_variant": "goodreads-interactions",
            "interaction_file": "goodreads_interactions.csv",
            "item_file": "goodreads_books.json.gz",
        }
    )
    with pytest.raises(ValueError, match="no event timestamp"):
        load_dataset({"dataset": dataset})


def test_download_registry_resolves_official_variants() -> None:
    assert resolve_source("movielens").files[0].name == "ml-1m.zip"
    assert resolve_source("goodreads", "spoiler").name == "goodreads-spoiler"
    assert resolve_source("goodreads", "full").name == "goodreads-full"
    assert all(
        file.url.startswith("https://")
        for source in ("steam", "lastfm")
        for file in resolve_source(source).files
    )


def _runtime_prepared() -> PreparedData:
    sequences = {
        "u0": list("abcdefghij"),
        "u1": ["a", "b", "c", "d", "e", "f", "a", "b", "a", "b"],
        "u2": ["a", "c", "e", "g", "i", "b", "d", "f", "h", "j"],
    }
    rows = []
    source_row = 0
    for user_id, item_ids in sequences.items():
        for position, item_id in enumerate(item_ids):
            split = "train" if position < 8 else "validation" if position == 8 else "test"
            rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "timestamp": position // 2,
                    "source_row": source_row,
                    "tie_key": f"row:{source_row:020d}",
                    "split": split,
                    "sequence_position": position,
                }
            )
            source_row += 1
    interactions = pd.DataFrame.from_records(rows)
    training = interactions[interactions["split"] == "train"]
    frequencies = training.groupby("item_id").size()
    item_ids = sorted(interactions["item_id"].unique())
    items = pd.DataFrame(
        {
            "item_id": item_ids,
            "title": [f"title-{item_id}" for item_id in item_ids],
            "training_frequency": [int(frequencies.get(item_id, 0)) for item_id in item_ids],
            "group_id": [0 if int(frequencies.get(item_id, 0)) <= 2 else 1 for item_id in item_ids],
        }
    )
    groups = items[["item_id", "training_frequency", "group_id"]].copy()
    return PreparedData(
        interactions=interactions,
        items=items,
        groups=groups,
        manifest={"dataset": "runtime-fixture"},
    )


def test_sequence_dataset_uses_all_true_prior_events() -> None:
    prepared = _runtime_prepared()
    dataset = SequenceDataset(
        prepared,
        "test",
        min_history=3,
        candidate_pool={"mode": "full_catalog", "exclude_history": False},
    )
    assert len(dataset) == 3
    example = next(
        dataset[index]
        for index in range(len(dataset))
        if dataset[index]["user_key"] == "u0"
    )
    tables = prepared_tables(prepared)
    expected = [tables.item_to_index[value] for value in "abcdefghi"]
    assert example["history"] == expected
    assert example["timestamp"] == 9
    assert example["event_timestamp"] == 4
    assert example["split"] == "test"


def test_candidate_target_inclusion_is_configurable_by_split() -> None:
    prepared = _runtime_prepared()
    without_target = SequenceDataset(
        prepared,
        "test",
        min_history=3,
        candidate_pool={
            "mode": "deterministic_frequency",
            "size": 1,
            "exclude_history": False,
            "ensure_evaluation_target": False,
        },
    )
    u0_without = next(
        without_target[index]
        for index in range(len(without_target))
        if without_target[index]["user_key"] == "u0"
    )
    assert u0_without["target"] not in u0_without["candidates"]
    with_target = SequenceDataset(
        prepared,
        "test",
        min_history=3,
        candidate_pool={
            "mode": "frequency_pool",
            "size": 1,
            "exclude_history": False,
            "ensure_target_by_split": {"test": True},
        },
    )
    u0_with = next(
        with_target[index]
        for index in range(len(with_target))
        if with_target[index]["user_key"] == "u0"
    )
    assert u0_with["candidates"] == [u0_with["target"]]


def test_stratified_frequency_pool_balances_available_groups() -> None:
    prepared = _runtime_prepared()
    dataset = SequenceDataset(
        prepared,
        "test",
        min_history=3,
        candidate_pool={
            "mode": "stratified_frequency",
            "size": 4,
            "exclude_history": False,
            "ensure_evaluation_target": False,
        },
    )
    tables = prepared_tables(prepared)
    candidates = dataset[0]["candidates"]
    counts = [
        sum(int(tables.item_groups[item]) == group for item in candidates)
        for group in (0, 1)
    ]
    assert counts == [2, 2]


def test_temporal_sampler_has_one_event_per_user_and_preserves_order() -> None:
    dataset = SequenceDataset(
        _runtime_prepared(),
        "train",
        min_history=3,
        candidate_pool={"mode": "full_catalog", "exclude_history": True},
    )
    sampler = TemporalBatchSampler(dataset, batch_size=2, shuffle=True, seed=11)
    observed: dict[int, list[int]] = {}
    for batch in sampler:
        users = [dataset.examples[index].user_index for index in batch]
        assert len(users) == len(set(users))
        for index in batch:
            example = dataset.examples[index]
            observed.setdefault(example.user_index, []).append(example.position)
    assert all(positions == sorted(positions) for positions in observed.values())
    sampler.set_epoch(2)
    assert sum(1 for _ in sampler) == len(sampler)


def test_collator_and_build_dataloader_match_model_inputs() -> None:
    prepared = _runtime_prepared()
    dataset = SequenceDataset(
        prepared,
        "train",
        min_history=3,
        candidate_pool={
            "mode": "deterministic_frequency",
            "size": 3,
            "exclude_history": True,
            "ensure_train_target": True,
        },
    )
    collated = sequence_collate([dataset[0], dataset[1]])
    assert set(collated) == {
        "user_ids",
        "timestamps",
        "histories",
        "history_mask",
        "candidates",
        "candidate_mask",
        "targets",
        "domains",
    }
    assert collated["histories"].shape[0] == 2
    for row in range(2):
        active = collated["candidates"][row][collated["candidate_mask"][row]]
        assert collated["targets"][row] in active
    config = {
        "dataset": {"min_history": 3},
        "candidate_pool": {
            "mode": "frequency_pool",
            "size": 3,
            "exclude_history": True,
            "ensure_train_target": True,
        },
        "training": {"micro_batch_size": 2, "seed": 5},
        "runtime": {"num_workers": 0},
    }
    loader = build_dataloader(prepared, config, "train", shuffle=False)
    batch = next(iter(loader))
    assert batch["user_ids"].unique().numel() == batch["user_ids"].numel()


def test_load_prepared_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    prepared = _runtime_prepared()
    directory = tmp_path / "prepared"
    directory.mkdir()
    prepared.interactions.to_parquet(directory / "interactions.parquet", index=False)
    prepared.items.to_parquet(directory / "items.parquet", index=False)
    prepared.groups.to_parquet(directory / "groups.parquet", index=False)
    with (directory / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(prepared.manifest, handle)
    loaded = load_prepared(directory)
    assert loaded.num_users == 3
    assert loaded.num_items == 10
    assert loaded.item_groups.shape == (11,)
    assert loaded.index_to_item[loaded.item_to_index["j"]] == "j"
