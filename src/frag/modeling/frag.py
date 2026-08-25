from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from frag.modeling.fairness import (
    FairnessRiskState,
    candidate_mass_hinge_square,
    soft_group_exposure,
)
from frag.modeling.generator import GeneratorOutput, build_generator
from frag.modeling.retriever import AdaptiveRetriever, RetrievalOutput


@dataclass
class FRAGOutput:
    loss: torch.Tensor
    generator_loss: torch.Tensor
    budget_loss: torch.Tensor
    retrieval: RetrievalOutput
    group_exposure: torch.Tensor
    candidate_mass: torch.Tensor
    next_risk: torch.Tensor
    generator_scores: torch.Tensor | None


class TwoTowerScorer(nn.Module):
    def __init__(self, num_items: int, config: dict[str, Any]) -> None:
        super().__init__()
        dimension = int(config.get("embedding_dim", 256))
        layers = int(config.get("history_layers", 1))
        dropout = float(config.get("dropout", 0.1)) if layers > 1 else 0.0
        self.items = nn.Embedding(num_items + 1, dimension, padding_idx=0)
        self.encoder = nn.GRU(
            dimension,
            dimension,
            num_layers=layers,
            dropout=dropout,
            batch_first=True,
        )
        self.activation = config.get("score_activation", "sigmoid")

    def forward(
        self,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        if history_mask.ndim != 2 or history_mask.shape != histories.shape:
            raise ValueError("history_mask must match histories")
        if history_mask.size(1) > 1 and torch.any(history_mask[:, 1:] & ~history_mask[:, :-1]):
            raise ValueError("history masks must use contiguous right padding")
        history_embeddings = self.items(histories)
        lengths = history_mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            history_embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, state = self.encoder(packed)
        user_vectors = F.normalize(state[-1], dim=-1)
        item_vectors = F.normalize(self.items(candidates), dim=-1)
        scores = torch.einsum("bd,bcd->bc", user_vectors, item_vectors)
        if self.activation == "sigmoid":
            return torch.sigmoid(scores)
        if self.activation == "identity":
            return scores
        raise ValueError(f"Unknown score activation: {self.activation}")


class FRAG(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        item_groups: torch.Tensor,
        titles: dict[int, str],
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        self.titles = titles
        groups = torch.as_tensor(item_groups, dtype=torch.long)
        if groups.shape != (num_items + 1,):
            raise ValueError("item_groups must have shape [num_items + 1]")
        self.register_buffer("item_groups", groups)
        self.scorer = TwoTowerScorer(num_items, config["model"]["retriever"])
        tau = self._initial_tau(config)
        retrieval = config["retrieval"]
        fixed = retrieval.get("fixed_k")
        if fixed is not None and not isinstance(fixed, int):
            raise ValueError("fixed_k must be resolved to an integer before model construction")
        mode = "fixed_top_k" if fixed is not None else "adaptive"
        self.retriever = AdaptiveRetriever(
            target_shares=config["fairness"]["target_shares"],
            tau_init=tau,
            gamma=float(retrieval["gamma"]),
            lambda_fair=float(config["fairness"]["lambda"]),
            selection_mode=mode,
            fixed_k=fixed,
        )
        self.risk = FairnessRiskState(
            num_users,
            config["fairness"]["target_shares"],
            mode=config["fairness"].get("state_mode", "cumulative"),
        )
        self.generator = build_generator(num_items, config["model"]["generator"])
        self.joint_training = bool(config["training"].get("joint_training", True))
        self.kmax = float(retrieval["kmax"])
        self.eta = float(config["training"]["eta"])
        self.minimum_hard_size = int(retrieval.get("minimum_hard_size", 0))
        self.short_set_policy = str(config.get("evaluation", {}).get("short_set_policy", "keep"))
        if self.minimum_hard_size < 0:
            raise ValueError("minimum_hard_size must be nonnegative")
        if self.short_set_policy not in {"keep", "error"}:
            raise ValueError("short_set_policy must be keep or error")

    @staticmethod
    def _initial_tau(config: dict[str, Any]) -> float:
        retrieval = config["retrieval"]
        if "tau_value" in retrieval:
            return float(retrieval["tau_value"])
        mode = retrieval.get("tau_init", "uniform")
        if mode == "uniform":
            low = float(retrieval["tau_low"])
            high = float(retrieval["tau_high"])
            return float(torch.empty(()).uniform_(low, high).item())
        return float(mode)

    def reset_risk(self) -> None:
        self.risk.reset()

    def retrieval_step(
        self,
        user_ids: torch.Tensor,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[RetrievalOutput, torch.Tensor, torch.Tensor]:
        scores = self.scorer(histories, history_mask, candidates)
        group_ids = self.item_groups[candidates]
        state = self.risk.current(user_ids)
        retrieval = self.retriever(scores, group_ids, state, candidate_mask)
        exposure, mass = soft_group_exposure(
            retrieval.soft_weights,
            group_ids,
            self.risk.num_groups,
            candidate_mask,
        )
        return retrieval, exposure, mass

    def forward(
        self,
        user_ids: torch.Tensor,
        timestamps: torch.Tensor,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        targets: torch.Tensor,
        domains: list[str],
        update_state: bool = True,
    ) -> FRAGOutput:
        retrieval, exposure, mass = self.retrieval_step(
            user_ids,
            histories,
            history_mask,
            candidates,
            candidate_mask,
        )
        weights = retrieval.soft_weights if self.joint_training else retrieval.soft_weights.detach()
        generated: GeneratorOutput = self.generator(
            histories,
            history_mask,
            candidates,
            candidate_mask,
            targets,
            weights,
            retrieval.hard_mask,
            retrieval.scores,
            self.titles,
            domains,
        )
        budget = candidate_mass_hinge_square(
            retrieval.soft_weights,
            self.kmax,
            candidate_mask,
            reduction="mean",
        )
        loss = generated.loss + self.eta * budget
        if update_state:
            next_risk = self.risk.update(
                user_ids,
                exposure,
                mass,
                timestamps,
            )
        else:
            next_risk = self.risk.current(user_ids)
        return FRAGOutput(
            loss=loss,
            generator_loss=generated.loss,
            budget_loss=budget,
            retrieval=retrieval,
            group_exposure=exposure,
            candidate_mass=mass,
            next_risk=next_risk,
            generator_scores=generated.candidate_scores,
        )

    @torch.no_grad()
    def recommend(
        self,
        user_ids: torch.Tensor,
        timestamps: torch.Tensor,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        domains: list[str],
        cutoff: int,
        update_state: bool = True,
    ) -> tuple[list[list[int]], FRAGOutput]:
        if cutoff < 1:
            raise ValueError("cutoff must be positive")
        was_training = self.training
        self.eval()
        try:
            return self._recommend(
                user_ids,
                timestamps,
                histories,
                history_mask,
                candidates,
                candidate_mask,
                domains,
                cutoff,
                update_state,
            )
        finally:
            self.train(was_training)

    def _recommend(
        self,
        user_ids: torch.Tensor,
        timestamps: torch.Tensor,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        domains: list[str],
        cutoff: int,
        update_state: bool,
    ) -> tuple[list[list[int]], FRAGOutput]:
        retrieval, exposure, mass = self.retrieval_step(
            user_ids,
            histories,
            history_mask,
            candidates,
            candidate_mask,
        )
        scores = self.generator.rank(
            histories,
            history_mask,
            candidates,
            candidate_mask,
            retrieval.soft_weights,
            retrieval.hard_mask,
            retrieval.scores,
            self.titles,
            domains,
        )
        rankings = []
        for row in range(candidates.size(0)):
            valid = candidate_mask[row].bool() & retrieval.hard_mask[row].bool()
            hard_size = int(valid.sum())
            required = max(self.minimum_hard_size, cutoff)
            if self.short_set_policy == "error" and hard_size < required:
                raise RuntimeError(
                    f"Hard retrieval size {hard_size} is below required size {required}"
                )
            count = min(cutoff, hard_size)
            if count == 0:
                rankings.append([])
                continue
            indices = torch.argsort(
                scores[row].masked_fill(~valid, -torch.inf),
                descending=True,
                stable=True,
            )[:count]
            rankings.append([int(value) for value in candidates[row, indices].tolist()])
        if update_state:
            next_risk = self.risk.update(user_ids, exposure, mass, timestamps)
        else:
            next_risk = self.risk.current(user_ids)
        zero = torch.zeros((), device=scores.device, dtype=retrieval.scores.dtype)
        output = FRAGOutput(
            loss=zero,
            generator_loss=zero,
            budget_loss=zero,
            retrieval=retrieval,
            group_exposure=exposure,
            candidate_mass=mass,
            next_risk=next_risk,
            generator_scores=scores,
        )
        return rankings, output

    def pretrain_loss(
        self,
        histories: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        targets: torch.Tensor,
        include_threshold: bool = False,
    ) -> torch.Tensor:
        scores = self.scorer(histories, history_mask, candidates)
        probabilities = scores
        if include_threshold:
            group_ids = self.item_groups[candidates]
            risk = scores.new_zeros((scores.size(0), self.risk.num_groups))
            probabilities = self.retriever(
                scores,
                group_ids,
                risk,
                candidate_mask,
            ).soft_weights
        labels = candidates.eq(targets.unsqueeze(1)).to(scores.dtype)
        losses = F.binary_cross_entropy(probabilities, labels, reduction="none")
        active = candidate_mask.to(losses.dtype)
        return (losses * active).sum() / active.sum().clamp_min(1.0)

    def freeze_retrieval(self) -> None:
        for parameter in self.scorer.parameters():
            parameter.requires_grad_(False)
        self.retriever.tau.requires_grad_(False)

    def parameter_groups(self) -> list[dict[str, Any]]:
        training = self.config["training"]
        retriever_parameters = [
            parameter for parameter in self.scorer.parameters() if parameter.requires_grad
        ]
        generator_parameters = [
            parameter for parameter in self.generator.parameters() if parameter.requires_grad
        ]
        groups = []
        if retriever_parameters:
            groups.append(
                {
                    "params": retriever_parameters,
                    "lr": float(training["retriever_learning_rate"]),
                }
            )
        if self.retriever.tau.requires_grad:
            groups.append(
                {
                    "params": [self.retriever.tau],
                    "lr": float(training["tau_learning_rate"]),
                }
            )
        if generator_parameters:
            groups.append(
                {
                    "params": generator_parameters,
                    "lr": float(training["learning_rate"]),
                }
            )
        return groups

    def save_components(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "scorer": self.scorer.state_dict(),
                "tau": self.retriever.tau.detach().cpu(),
            },
            destination / "retrieval.pt",
        )
        self.generator.save_adapter(destination / "generator")
