from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

StateMode = Literal["cumulative", "one_step"]
Reduction = Literal["none", "mean", "sum"]


def _candidate_mask(weights: Tensor, candidate_mask: Tensor | None) -> Tensor:
    if weights.ndim != 2:
        raise ValueError("weights must have shape [batch, candidates]")
    if candidate_mask is None:
        return torch.ones_like(weights, dtype=torch.bool)
    if candidate_mask.shape != weights.shape:
        raise ValueError("candidate_mask must match weights")
    return candidate_mask.to(device=weights.device, dtype=torch.bool)


def soft_group_exposure(
    weights: Tensor,
    group_ids: Tensor,
    num_groups: int,
    candidate_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    if num_groups < 1:
        raise ValueError("num_groups must be positive")
    if group_ids.shape != weights.shape:
        raise ValueError("group_ids must match weights")
    active = _candidate_mask(weights, candidate_mask)
    groups = group_ids.to(device=weights.device, dtype=torch.long)
    active_groups = groups[active]
    if active_groups.numel() and (
        torch.any(active_groups < 0) or torch.any(active_groups >= num_groups)
    ):
        raise ValueError("active group_ids must be in [0, num_groups)")
    safe_groups = torch.where(active, groups, torch.zeros_like(groups))
    active_weights = torch.where(active, weights, torch.zeros_like(weights))
    exposure = weights.new_zeros((weights.shape[0], num_groups))
    exposure.scatter_add_(1, safe_groups, active_weights)
    candidate_mass = active_weights.sum(dim=1)
    return exposure, candidate_mass


def candidate_mass_hinge_square(
    weights: Tensor,
    k_max: float,
    candidate_mask: Tensor | None = None,
    reduction: Reduction = "mean",
) -> Tensor:
    if k_max <= 0:
        raise ValueError("k_max must be positive")
    active = _candidate_mask(weights, candidate_mask)
    candidate_mass = torch.where(active, weights, torch.zeros_like(weights)).sum(dim=1)
    losses = torch.clamp_min(candidate_mass - k_max, 0).square()
    if reduction == "none":
        return losses
    if reduction == "mean":
        return losses.mean()
    if reduction == "sum":
        return losses.sum()
    raise ValueError("reduction must be one of: none, mean, sum")


class FairnessRiskState(nn.Module):
    def __init__(
        self,
        num_users: int,
        target_shares: Tensor | list[float] | tuple[float, ...],
        mode: StateMode = "cumulative",
    ) -> None:
        super().__init__()
        if num_users < 1:
            raise ValueError("num_users must be positive")
        targets = torch.as_tensor(target_shares, dtype=torch.float32)
        if targets.ndim != 1 or targets.numel() < 1:
            raise ValueError("target_shares must be a nonempty vector")
        if not torch.isfinite(targets).all() or torch.any(targets < 0):
            raise ValueError("target_shares must be finite and nonnegative")
        if targets.sum() > 1.0 + 1e-6:
            raise ValueError("target_shares must sum to at most one")
        if mode not in ("cumulative", "one_step"):
            raise ValueError("mode must be cumulative or one_step")
        self.num_users = int(num_users)
        self.num_groups = int(targets.numel())
        self.mode = mode
        self.register_buffer("target_shares", targets)
        self.register_buffer("risk", torch.zeros(num_users, self.num_groups))
        self.register_buffer(
            "last_timestamps",
            torch.full((num_users,), -1, dtype=torch.long),
            persistent=False,
        )

    def _user_ids(self, user_ids: Tensor | list[int] | tuple[int, ...]) -> Tensor:
        ids = torch.as_tensor(user_ids, device=self.risk.device, dtype=torch.long)
        if ids.ndim != 1:
            raise ValueError("user_ids must be a vector")
        if ids.numel() and (torch.any(ids < 0) or torch.any(ids >= self.num_users)):
            raise ValueError("user_ids are out of range")
        return ids

    @torch.no_grad()
    def reset(self, user_ids: Tensor | list[int] | tuple[int, ...] | None = None) -> None:
        if user_ids is None:
            self.risk.zero_()
            self.last_timestamps.fill_(-1)
            return
        ids = self._user_ids(user_ids)
        self.risk.index_fill_(0, ids, 0)
        self.last_timestamps.index_fill_(0, ids, -1)

    def current(self, user_ids: Tensor | list[int] | tuple[int, ...]) -> Tensor:
        ids = self._user_ids(user_ids)
        return self.risk.index_select(0, ids).detach().clone()

    @torch.no_grad()
    def update(
        self,
        user_ids: Tensor | list[int] | tuple[int, ...],
        group_exposure: Tensor,
        candidate_mass: Tensor,
        timestamps: Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> Tensor:
        ids = self._user_ids(user_ids)
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("a state update may contain at most one interaction per user")
        exposure = torch.as_tensor(
            group_exposure,
            device=self.risk.device,
            dtype=self.risk.dtype,
        ).detach()
        mass = torch.as_tensor(
            candidate_mass,
            device=self.risk.device,
            dtype=self.risk.dtype,
        ).detach()
        if exposure.shape != (ids.numel(), self.num_groups):
            raise ValueError("group_exposure must have shape [batch, num_groups]")
        if mass.shape != (ids.numel(),):
            raise ValueError("candidate_mass must have shape [batch]")
        if not torch.isfinite(exposure).all() or not torch.isfinite(mass).all():
            raise ValueError("exposure and candidate_mass must be finite")
        if torch.any(exposure < 0) or torch.any(mass < 0):
            raise ValueError("exposure and candidate_mass must be nonnegative")
        if timestamps is not None:
            times = torch.as_tensor(
                timestamps,
                device=self.risk.device,
                dtype=torch.long,
            )
            if times.shape != ids.shape:
                raise ValueError("timestamps must match user_ids")
            previous_times = self.last_timestamps.index_select(0, ids)
            if torch.any(times <= previous_times):
                raise ValueError("timestamps must increase strictly for each user")
            self.last_timestamps.index_copy_(0, ids, times)
        deficit = self.target_shares.unsqueeze(0) * mass.unsqueeze(1) - exposure
        if self.mode == "cumulative":
            updated = torch.clamp_min(self.risk.index_select(0, ids) + deficit, 0)
        else:
            updated = torch.clamp_min(deficit, 0)
        self.risk.index_copy_(0, ids, updated)
        return updated.detach().clone()

    @torch.no_grad()
    def update_from_weights(
        self,
        user_ids: Tensor | list[int] | tuple[int, ...],
        weights: Tensor,
        group_ids: Tensor,
        candidate_mask: Tensor | None = None,
        timestamps: Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> Tensor:
        exposure, candidate_mass = soft_group_exposure(
            weights,
            group_ids,
            self.num_groups,
            candidate_mask,
        )
        return self.update(user_ids, exposure, candidate_mass, timestamps)
