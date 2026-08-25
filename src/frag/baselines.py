from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

required_fields = {"user_id", "round", "target_id", "ranking", "retrieved_ids"}


def load_baseline_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            missing = required_fields.difference(record)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"{source}:{line_number} missing {names}")
            record["round"] = int(record["round"])
            record["target_id"] = int(record["target_id"])
            record["ranking"] = [int(value) for value in record["ranking"]]
            record["retrieved_ids"] = [int(value) for value in record["retrieved_ids"]]
            records.append(record)
    records.sort(key=lambda row: (str(row["user_id"]), row["round"]))
    _validate_chronology(records)
    return records


def save_baseline_contract(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            missing = required_fields.difference(record)
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"Missing {names}")
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _validate_chronology(records: list[dict[str, Any]]) -> None:
    latest: dict[str, int] = {}
    for record in records:
        user = str(record["user_id"])
        round_value = int(record["round"])
        if user in latest and round_value <= latest[user]:
            raise ValueError(f"Non-increasing round for user {user}")
        latest[user] = round_value
