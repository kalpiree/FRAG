from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .significance import mean_confidence_interval


def _validated_targets(
    target_shares: Mapping[Any, float],
) -> tuple[tuple[Any, ...], list[float]]:
    if not target_shares:
        raise ValueError("target shares must not be empty")
    groups = tuple(target_shares)
    targets = [float(target_shares[group]) for group in groups]
    if not all(math.isfinite(target) and target > 0.0 for target in targets):
        raise ValueError("target shares must be positive and finite")
    if sum(targets) > 1.0 + 1e-12:
        raise ValueError("target shares must sum to at most one")
    return groups, targets


def per_user_cfd_trajectory(
    hard_sets_by_user: Mapping[Any, Sequence[Sequence[Any]]],
    item_groups: Mapping[Any, Any],
    target_shares: Mapping[Any, float],
    horizon: int | None = None,
) -> dict[Any, list[float]]:
    if not hard_sets_by_user:
        raise ValueError("at least one user is required")
    if horizon is None:
        horizon = min(len(rounds) for rounds in hard_sets_by_user.values())
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    groups, targets = _validated_targets(target_shares)
    group_indexes = {group: index for index, group in enumerate(groups)}
    cohort = {
        user: rounds
        for user, rounds in hard_sets_by_user.items()
        if len(rounds) >= horizon
    }
    if not cohort:
        raise ValueError("no users are observed throughout the requested horizon")
    trajectories: dict[Any, list[float]] = {}
    for user, rounds in cohort.items():
        counts = [0.0] * len(groups)
        total = 0.0
        values: list[float] = []
        for index, candidates in enumerate(rounds[:horizon]):
            selected = list(candidates)
            if len(selected) != len(set(selected)):
                raise ValueError("hard candidate sets must contain unique items")
            for item in selected:
                if item not in item_groups:
                    raise KeyError(f"missing group for item {item!r}")
                group = item_groups[item]
                if group not in group_indexes:
                    raise ValueError(f"item {item!r} belongs to an untargeted group")
                counts[group_indexes[group]] += 1.0
                total += 1.0
            if total == 0.0:
                raise ValueError(
                    f"user {user!r} has zero cumulative hard-set exposure at round {index + 1}"
                )
            shares = [count / total for count in counts]
            values.append(
                max(
                    max(target - share, 0.0)
                    for target, share in zip(targets, shares, strict=True)
                )
            )
        trajectories[user] = values
    return trajectories


def average_cfd_trajectory(
    hard_sets_by_user: Mapping[Any, Sequence[Sequence[Any]]],
    item_groups: Mapping[Any, Any],
    target_shares: Mapping[Any, float],
    horizon: int | None = None,
) -> list[float]:
    per_user = per_user_cfd_trajectory(
        hard_sets_by_user, item_groups, target_shares, horizon
    )
    trajectories = tuple(per_user.values())
    return [
        sum(trajectory[round_index] for trajectory in trajectories) / len(trajectories)
        for round_index in range(len(trajectories[0]))
    ]


def summarize_seed_cfd(
    seed_trajectories: Sequence[Sequence[float]],
    confidence: float = 0.95,
) -> dict[str, Any]:
    rows = [list(trajectory) for trajectory in seed_trajectories]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("seed trajectories must have shape [seeds, rounds]")
    return mean_confidence_interval(rows, confidence=confidence, axis=0)
