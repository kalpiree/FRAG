from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

SelectionMode = Literal["adaptive", "fixed_top_k"]


@dataclass(frozen=True)
class RetrievalOutput:
    scores: Tensor
    fairness_adjusted_scores: Tensor
    soft_weights: Tensor
    hard_mask: Tensor
    ranked_indices: Tensor
    ranked_mask: Tensor


class AdaptiveRetriever(nn.Module):
    def __init__(
        self,
        target_shares: Tensor | list[float] | tuple[float, ...],
        tau_init: float,
        gamma: float,
        lambda_fair: float,
        selection_mode: SelectionMode = "adaptive",
        fixed_k: int | None = None,
    ) -> None:
        super().__init__()
        targets = torch.as_tensor(target_shares, dtype=torch.float32)
        if targets.ndim != 1 or targets.numel() < 1:
            raise ValueError("target_shares must be a nonempty vector")
        if not torch.isfinite(targets).all() or torch.any(targets < 0):
            raise ValueError("target_shares must be finite and nonnegative")
        if targets.sum() > 1.0 + 1e-6:
            raise ValueError("target_shares must sum to at most one")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if lambda_fair < 0:
            raise ValueError("lambda_fair must be nonnegative")
        if selection_mode not in ("adaptive", "fixed_top_k"):
            raise ValueError("selection_mode must be adaptive or fixed_top_k")
        if fixed_k is not None and fixed_k < 1:
            raise ValueError("fixed_k must be positive")
        if selection_mode == "fixed_top_k" and fixed_k is None:
            raise ValueError("fixed_k is required for fixed_top_k mode")
        self.register_buffer("target_shares", targets)
        self.tau = nn.Parameter(torch.tensor(float(tau_init), dtype=torch.float32))
        self.gamma = float(gamma)
        self.lambda_fair = float(lambda_fair)
        self.selection_mode = selection_mode
        self.fixed_k = fixed_k

    def _validate(
        self,
        scores: Tensor,
        group_ids: Tensor,
        risk_states: Tensor,
        candidate_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if scores.ndim != 2:
            raise ValueError("scores must have shape [batch, candidates]")
        if group_ids.shape != scores.shape:
            raise ValueError("group_ids must match scores")
        expected_state_shape = (scores.shape[0], self.target_shares.numel())
        if risk_states.shape != expected_state_shape:
            raise ValueError("risk_states must have shape [batch, num_groups]")
        if candidate_mask is None:
            active = torch.ones_like(scores, dtype=torch.bool)
        else:
            if candidate_mask.shape != scores.shape:
                raise ValueError("candidate_mask must match scores")
            active = candidate_mask.to(device=scores.device, dtype=torch.bool)
        groups = group_ids.to(device=scores.device, dtype=torch.long)
        active_groups = groups[active]
        if active_groups.numel() and (
            torch.any(active_groups < 0)
            or torch.any(active_groups >= self.target_shares.numel())
        ):
            raise ValueError("active group_ids must be in [0, num_groups)")
        return groups, active

    def _fixed_top_k_mask(
        self,
        adjusted_scores: Tensor,
        active: Tensor,
        fixed_k: int,
    ) -> Tensor:
        if fixed_k < 1:
            raise ValueError("fixed_k must be positive")
        k = min(fixed_k, adjusted_scores.shape[1])
        masked_scores = adjusted_scores.masked_fill(~active, -torch.inf)
        indices = torch.topk(masked_scores, k=k, dim=1).indices
        selected = torch.zeros_like(active)
        selected.scatter_(1, indices, True)
        return selected & active

    def _rank(self, scores: Tensor, hard_mask: Tensor) -> tuple[Tensor, Tensor]:
        ranking_scores = scores.masked_fill(~hard_mask, -torch.inf)
        ranked_indices = torch.argsort(
            ranking_scores,
            dim=1,
            descending=True,
            stable=True,
        )
        ranked_mask = torch.gather(hard_mask, 1, ranked_indices)
        return ranked_indices, ranked_mask

    def forward(
        self,
        scores: Tensor,
        group_ids: Tensor,
        risk_states: Tensor,
        candidate_mask: Tensor | None = None,
        selection_mode: SelectionMode | None = None,
        fixed_k: int | None = None,
    ) -> RetrievalOutput:
        groups, active = self._validate(scores, group_ids, risk_states, candidate_mask)
        risks = risk_states.detach().to(device=scores.device, dtype=scores.dtype)
        targets = self.target_shares.to(device=scores.device, dtype=scores.dtype)
        safe_groups = torch.where(active, groups, torch.zeros_like(groups))
        group_risk = torch.gather(risks, 1, safe_groups)
        centered_risk = (risks * targets.unsqueeze(0)).sum(dim=1, keepdim=True)
        tau = self.tau.to(dtype=scores.dtype)
        adjusted_scores = scores - tau + self.lambda_fair * (group_risk - centered_risk)
        soft_weights = torch.where(
            active,
            torch.sigmoid(adjusted_scores / self.gamma),
            torch.zeros_like(scores),
        )
        mode = selection_mode or self.selection_mode
        if mode == "adaptive":
            hard_mask = active & (soft_weights >= 0.5)
        elif mode == "fixed_top_k":
            selected_k = fixed_k if fixed_k is not None else self.fixed_k
            if selected_k is None:
                raise ValueError("fixed_k is required for fixed_top_k mode")
            hard_mask = self._fixed_top_k_mask(adjusted_scores, active, selected_k)
        else:
            raise ValueError("selection_mode must be adaptive or fixed_top_k")
        ranked_indices, ranked_mask = self._rank(scores, hard_mask)
        visible_adjusted_scores = adjusted_scores.masked_fill(~active, -torch.inf)
        return RetrievalOutput(
            scores=scores,
            fairness_adjusted_scores=visible_adjusted_scores,
            soft_weights=soft_weights,
            hard_mask=hard_mask,
            ranked_indices=ranked_indices,
            ranked_mask=ranked_mask,
        )
