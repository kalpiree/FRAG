from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_METRICS = ("recall", "precision", "mrr", "ndcg")


def _relevant_set(value: Any) -> set[Any]:
    if isinstance(value, str | bytes) or not isinstance(value, Iterable):
        return {value}
    return set(value)


def _top_k(ranked_items: Sequence[Any], k: int) -> list[Any]:
    if k <= 0:
        raise ValueError("k must be positive")
    items = list(ranked_items)[:k]
    if len(items) != len(set(items)):
        raise ValueError("ranked items must be unique within the cutoff")
    return items


def utility_at_k(
    ranked_items: Sequence[Any], relevant_items: Any, k: int = 25
) -> dict[str, float]:
    ranked = _top_k(ranked_items, k)
    relevant = _relevant_set(relevant_items)
    if not relevant:
        raise ValueError("relevant items must not be empty")
    relevance = [item in relevant for item in ranked]
    hits = sum(relevance)
    recall = hits / len(relevant)
    precision = hits / k
    first_hit = next((rank for rank, hit in enumerate(relevance, 1) if hit), None)
    mrr = 0.0 if first_hit is None else 1.0 / first_hit
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(relevance, 1))
    ideal_count = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / idcg
    return {
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "ndcg": ndcg,
    }


def per_user_utility(
    rankings_by_user: Mapping[Any, Sequence[Sequence[Any]]],
    targets_by_user: Mapping[Any, Sequence[Any]],
    k: int = 25,
) -> dict[Any, dict[str, float]]:
    if set(rankings_by_user) != set(targets_by_user):
        raise ValueError("rankings and targets must contain the same users")
    results: dict[Any, dict[str, float]] = {}
    for user, rounds in rankings_by_user.items():
        targets = targets_by_user[user]
        if len(rounds) != len(targets):
            raise ValueError(f"round and target counts differ for user {user!r}")
        if not rounds:
            raise ValueError(f"user {user!r} has no evaluation interactions")
        values = [
            utility_at_k(ranking, target, k)
            for ranking, target in zip(rounds, targets, strict=True)
        ]
        results[user] = {
            metric: sum(value[metric] for value in values) / len(values)
            for metric in _METRICS
        }
    if not results:
        raise ValueError("at least one user is required")
    return results


def aggregate_utility(
    per_user_scores: Mapping[Any, Mapping[str, float]],
) -> dict[str, float]:
    if not per_user_scores:
        raise ValueError("at least one user is required")
    result: dict[str, float] = {}
    for metric in _METRICS:
        values = [float(scores[metric]) for scores in per_user_scores.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{metric} scores must be finite")
        result[metric] = sum(values) / len(values)
    return result
