from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _targets(target_shares: Mapping[Any, float]) -> tuple[tuple[Any, ...], list[float]]:
    if not target_shares:
        raise ValueError("target shares must not be empty")
    groups = tuple(target_shares)
    targets = [float(target_shares[group]) for group in groups]
    if not all(math.isfinite(target) and target > 0.0 for target in targets):
        raise ValueError("target shares must be positive and finite")
    if sum(targets) > 1.0 + 1e-12:
        raise ValueError("target shares must sum to at most one")
    return groups, targets


def _round_items(items: Sequence[Any], cutoff: int | None) -> list[Any]:
    if cutoff is not None and cutoff <= 0:
        raise ValueError("cutoff must be positive")
    selected = list(items) if cutoff is None else list(items)[:cutoff]
    if len(selected) != len(set(selected)):
        raise ValueError("recommended items must be unique within each round")
    return selected


def per_user_exposure_fairness(
    recommendations_by_user: Mapping[Any, Sequence[Sequence[Any]]],
    item_groups: Mapping[Any, Any],
    target_shares: Mapping[Any, float],
    cutoff: int | None = 25,
) -> dict[Any, dict[str, Any]]:
    groups, targets = _targets(target_shares)
    group_indexes = {group: index for index, group in enumerate(groups)}
    results: dict[Any, dict[str, Any]] = {}
    for user, rounds in recommendations_by_user.items():
        if not rounds:
            raise ValueError(f"user {user!r} has no evaluation interactions")
        counts = [0.0] * len(groups)
        coverage = 0.0
        for items in rounds:
            selected = _round_items(items, cutoff)
            represented: set[Any] = set()
            for item in selected:
                if item not in item_groups:
                    raise KeyError(f"missing group for item {item!r}")
                group = item_groups[item]
                if group not in group_indexes:
                    raise ValueError(f"item {item!r} belongs to an untargeted group")
                counts[group_indexes[group]] += 1.0
                represented.add(group)
            coverage += len(represented) / len(groups)
        total = sum(counts)
        shares = [count / total for count in counts] if total else [0.0] * len(groups)
        results[user] = {
            "ed": sum(
                abs(share - target)
                for share, target in zip(shares, targets, strict=True)
            )
            / len(groups),
            "wger": min(
                share / target
                for share, target in zip(shares, targets, strict=True)
            ),
            "gc": coverage / len(rounds),
            "interaction_count": len(rounds),
            "shares": {group: float(shares[index]) for index, group in enumerate(groups)},
        }
    if not results:
        raise ValueError("at least one user is required")
    return results


def aggregate_fairness(
    per_user_scores: Mapping[Any, Mapping[str, Any]],
) -> dict[str, float]:
    if not per_user_scores:
        raise ValueError("at least one user is required")
    ed = [float(score["ed"]) for score in per_user_scores.values()]
    wger = [float(score["wger"]) for score in per_user_scores.values()]
    gc = [float(score["gc"]) for score in per_user_scores.values()]
    counts = [float(score["interaction_count"]) for score in per_user_scores.values()]
    if not all(math.isfinite(value) for value in ed + wger + gc + counts):
        raise ValueError("per-user fairness scores must be finite")
    if any(count <= 0.0 for count in counts):
        raise ValueError("interaction counts must be positive")
    return {
        "ed": sum(ed) / len(ed),
        "wger": sum(wger) / len(wger),
        "gc": sum(
            value * count for value, count in zip(gc, counts, strict=True)
        )
        / sum(counts),
    }
