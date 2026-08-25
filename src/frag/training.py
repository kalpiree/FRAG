from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from frag.modeling.frag import FRAG


@dataclass(frozen=True)
class SequenceExample:
    user_index: int
    user_id: str
    round: int
    split: str
    history: tuple[int, ...]
    candidates: tuple[int, ...]
    target: int
    domain: str
    target_key: str = ""


@dataclass
class SequenceBatch:
    user_ids: Tensor
    user_keys: list[str]
    rounds: Tensor
    histories: Tensor
    history_mask: Tensor
    candidates: Tensor
    candidate_mask: Tensor
    targets: Tensor
    target_keys: list[str]
    domains: list[str]

    def to(self, device: torch.device) -> SequenceBatch:
        return SequenceBatch(
            user_ids=self.user_ids.to(device),
            user_keys=self.user_keys,
            rounds=self.rounds.to(device),
            histories=self.histories.to(device),
            history_mask=self.history_mask.to(device),
            candidates=self.candidates.to(device),
            candidate_mask=self.candidate_mask.to(device),
            targets=self.targets.to(device),
            target_keys=self.target_keys,
            domains=self.domains,
        )


@dataclass
class TrainingBundle:
    optimizer: Optimizer
    scheduler: LambdaLR
    optimizer_steps: int


def validate_effective_batch(config: dict[str, Any]) -> tuple[int, int, int]:
    training = config["training"]
    micro = int(training["micro_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    effective = int(training["effective_batch_size"])
    if micro < 1 or accumulation < 1 or effective < 1:
        raise ValueError("batch sizes and accumulation must be positive")
    if micro * accumulation != effective:
        raise ValueError("effective_batch_size must equal micro_batch_size times accumulation")
    return micro, accumulation, effective


def collate_examples(examples: Sequence[SequenceExample]) -> SequenceBatch:
    if not examples:
        raise ValueError("cannot collate an empty batch")
    if len({example.user_index for example in examples}) != len(examples):
        raise ValueError("a chronological batch may contain at most one interaction per user")
    history_width = max(len(example.history) for example in examples)
    candidate_width = max(len(example.candidates) for example in examples)
    if history_width < 1 or candidate_width < 1:
        raise ValueError("histories and candidate pools must be nonempty")
    histories = torch.zeros(len(examples), history_width, dtype=torch.long)
    history_mask = torch.zeros_like(histories, dtype=torch.bool)
    candidates = torch.zeros(len(examples), candidate_width, dtype=torch.long)
    candidate_mask = torch.zeros_like(candidates, dtype=torch.bool)
    for row, example in enumerate(examples):
        histories[row, : len(example.history)] = torch.tensor(example.history)
        history_mask[row, : len(example.history)] = True
        candidates[row, : len(example.candidates)] = torch.tensor(example.candidates)
        candidate_mask[row, : len(example.candidates)] = True
    return SequenceBatch(
        user_ids=torch.tensor([example.user_index for example in examples], dtype=torch.long),
        user_keys=[example.user_id for example in examples],
        rounds=torch.tensor([example.round for example in examples], dtype=torch.long),
        histories=histories,
        history_mask=history_mask,
        candidates=candidates,
        candidate_mask=candidate_mask,
        targets=torch.tensor([example.target for example in examples], dtype=torch.long),
        target_keys=[example.target_key for example in examples],
        domains=[example.domain for example in examples],
    )


def chronological_batches(
    examples: Sequence[SequenceExample],
    batch_size: int,
    seed: int = 0,
    shuffle_users: bool = False,
) -> Iterator[SequenceBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    per_user: dict[int, list[SequenceExample]] = {}
    seen = set()
    for example in examples:
        key = (example.user_index, example.round)
        if key in seen:
            raise ValueError("duplicate user interaction round")
        seen.add(key)
        per_user.setdefault(example.user_index, []).append(example)
    for values in per_user.values():
        values.sort(key=lambda value: value.round)
    generator = random.Random(seed)
    maximum = max((len(values) for values in per_user.values()), default=0)
    for step in range(maximum):
        round_examples = [values[step] for values in per_user.values() if step < len(values)]
        round_examples.sort(key=lambda value: value.user_index)
        if shuffle_users:
            generator.shuffle(round_examples)
        for start in range(0, len(round_examples), batch_size):
            yield collate_examples(round_examples[start : start + batch_size])


def optimizer_step_count(batch_count: int, accumulation_steps: int, epochs: int) -> int:
    if batch_count < 0 or accumulation_steps < 1 or epochs < 0:
        raise ValueError("invalid optimizer step arguments")
    return math.ceil(batch_count / accumulation_steps) * epochs if batch_count else 0


def cosine_scheduler(
    optimizer: Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.0,
) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / warmup_steps
        denominator = max(total_steps - warmup_steps, 1)
        progress = min(max((step - warmup_steps) / denominator, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, scale)


def resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("runtime", {}).get("device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


class Trainer:
    def __init__(
        self,
        model: FRAG,
        config: dict[str, Any],
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device or resolve_device(config)
        self.model.to(self.device)
        (
            self.micro_batch_size,
            self.accumulation_steps,
            self.effective_batch_size,
        ) = validate_effective_batch(config)
        precision = str(config["training"].get("mixed_precision", "none")).lower()
        self.autocast_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }.get(precision)
        self.autocast_enabled = self.device.type == "cuda" and self.autocast_dtype is not None
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.autocast_enabled and self.autocast_dtype == torch.float16
        )
        self.optimizer: Optimizer | None = None
        self.scheduler: LambdaLR | None = None
        self.optimizer_steps = 0

    def _autocast(self):
        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _batches(
        self,
        examples: Sequence[SequenceExample],
        epoch: int,
        shuffle_users: bool,
    ) -> list[SequenceBatch]:
        seed = int(self.config["training"]["seed"]) + epoch
        return list(
            chronological_batches(
                examples,
                self.micro_batch_size,
                seed=seed,
                shuffle_users=shuffle_users,
            )
        )

    def _step(self, optimizer: Optimizer, scheduler: LambdaLR) -> None:
        maximum = float(self.config["training"].get("max_grad_norm", 0.0))
        if self.scaler.is_enabled():
            self.scaler.unscale_(optimizer)
        if maximum > 0:
            clip_grad_norm_(self.model.parameters(), maximum)
        if self.scaler.is_enabled():
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        self.optimizer_steps += 1

    def _windows(self, batches: Sequence[SequenceBatch]) -> Iterator[Sequence[SequenceBatch]]:
        for start in range(0, len(batches), self.accumulation_steps):
            yield batches[start : start + self.accumulation_steps]

    def pretrain_retriever(self, examples: Sequence[SequenceExample]) -> list[dict[str, Any]]:
        epochs = int(self.config["training"].get("retriever_pretrain_epochs", 0))
        if epochs <= 0:
            return []
        probe = self._batches(examples, 0, True)
        total_steps = optimizer_step_count(len(probe), self.accumulation_steps, epochs)
        parameters = [
            parameter for parameter in self.model.scorer.parameters() if parameter.requires_grad
        ]
        include_threshold = bool(
            self.config["training"].get("retriever_pretrain_tau", False)
        )
        if include_threshold and self.model.retriever.tau.requires_grad:
            parameters.append(self.model.retriever.tau)
        optimizer = AdamW(
            parameters,
            lr=float(
                self.config["training"].get(
                    "retriever_pretrain_learning_rate",
                    self.config["training"]["retriever_learning_rate"],
                )
            ),
            weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
        )
        scheduler = cosine_scheduler(
            optimizer,
            total_steps,
            float(self.config["training"].get("warmup_ratio", 0.0)),
        )
        history = []
        local_steps = 0
        for epoch in range(epochs):
            self.model.reset_risk()
            self.model.train()
            losses = []
            batches = self._batches(examples, epoch, True)
            for window in self._windows(batches):
                optimizer.zero_grad(set_to_none=True)
                window_size = sum(batch.user_ids.numel() for batch in window)
                for batch in window:
                    current = batch.to(self.device)
                    if not torch.all(
                        (current.candidates == current.targets.unsqueeze(1)).any(dim=1)
                    ):
                        raise ValueError("retriever pretraining requires every target in its pool")
                    with self._autocast():
                        loss = self.model.pretrain_loss(
                            current.histories,
                            current.history_mask,
                            current.candidates,
                            current.candidate_mask,
                            current.targets,
                            include_threshold,
                        )
                    losses.append(float(loss.detach().cpu()))
                    scaled = loss * current.user_ids.numel() / window_size
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                self._step(optimizer, scheduler)
                local_steps += 1
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": sum(losses) / max(len(losses), 1),
                    "optimizer_steps": local_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
        return history

    def fit(self, examples: Sequence[SequenceExample]) -> list[dict[str, Any]]:
        epochs = int(self.config["training"]["epochs"])
        if epochs < 1:
            raise ValueError("training epochs must be positive")
        probe = self._batches(examples, 0, True)
        if not probe:
            raise ValueError("training split has no examples")
        total_steps = optimizer_step_count(len(probe), self.accumulation_steps, epochs)
        groups = self.model.parameter_groups()
        if not groups:
            raise ValueError("model has no trainable parameter groups")
        self.optimizer = AdamW(
            groups,
            weight_decay=float(self.config["training"].get("weight_decay", 0.0)),
        )
        self.scheduler = cosine_scheduler(
            self.optimizer,
            total_steps,
            float(self.config["training"].get("warmup_ratio", 0.0)),
        )
        history = []
        self.optimizer_steps = 0
        for epoch in range(epochs):
            self.model.reset_risk()
            self.model.train()
            total = generator_total = budget_total = 0.0
            example_count = 0
            batches = self._batches(examples, epoch, True)
            for window in self._windows(batches):
                self.optimizer.zero_grad(set_to_none=True)
                window_size = sum(batch.user_ids.numel() for batch in window)
                for batch in window:
                    current = batch.to(self.device)
                    with self._autocast():
                        output = self.model(
                            user_ids=current.user_ids,
                            timestamps=current.rounds,
                            histories=current.histories,
                            history_mask=current.history_mask,
                            candidates=current.candidates,
                            candidate_mask=current.candidate_mask,
                            targets=current.targets,
                            domains=current.domains,
                            update_state=True,
                        )
                    count = current.user_ids.numel()
                    total += float(output.loss.detach().cpu()) * count
                    generator_total += float(output.generator_loss.detach().cpu()) * count
                    budget_total += float(output.budget_loss.detach().cpu()) * count
                    example_count += count
                    scaled = output.loss * count / window_size
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                self._step(self.optimizer, self.scheduler)
            history.append(
                {
                    "epoch": epoch + 1,
                    "loss": total / example_count,
                    "generator_loss": generator_total / example_count,
                    "budget_loss": budget_total / example_count,
                    "optimizer_steps": self.optimizer_steps,
                    "tau": float(self.model.retriever.tau.detach().cpu()),
                    "learning_rates": [group["lr"] for group in self.optimizer.param_groups],
                }
            )
        return history

    @torch.no_grad()
    def predict(
        self,
        examples: Sequence[SequenceExample],
        cutoff: int,
        split: str,
        index_to_item: Sequence[str | None] | None = None,
    ) -> list[dict[str, Any]]:
        selected = [example for example in examples if example.split == split]
        if not selected:
            raise ValueError(f"{split} split has no examples")
        self.model.reset_risk()
        self.model.eval()
        records = []
        batches = chronological_batches(selected, self.micro_batch_size)
        for batch in batches:
            current = batch.to(self.device)
            rankings, output = self.model.recommend(
                user_ids=current.user_ids,
                timestamps=current.rounds,
                histories=current.histories,
                history_mask=current.history_mask,
                candidates=current.candidates,
                candidate_mask=current.candidate_mask,
                domains=current.domains,
                cutoff=cutoff,
                update_state=True,
            )
            for row, ranking in enumerate(rankings):
                ordered = output.retrieval.ranked_indices[row]
                ordered_mask = output.retrieval.ranked_mask[row]
                selected_indices = ordered[ordered_mask]
                retrieved = current.candidates[row, selected_indices].tolist()
                external_ranking = (
                    [str(index_to_item[int(value)]) for value in ranking]
                    if index_to_item is not None
                    else []
                )
                external_retrieved = (
                    [str(index_to_item[int(value)]) for value in retrieved]
                    if index_to_item is not None
                    else []
                )
                records.append(
                    {
                        "user_id": batch.user_keys[row],
                        "user_index": int(current.user_ids[row]),
                        "round": int(current.rounds[row]),
                        "split": split,
                        "target_id": int(current.targets[row]),
                        "target_key": batch.target_keys[row],
                        "ranking": ranking,
                        "ranking_keys": external_ranking,
                        "retrieved_ids": [int(value) for value in retrieved],
                        "retrieved_keys": external_retrieved,
                        "hard_size": len(retrieved),
                        "candidate_mass": float(output.candidate_mass[row].cpu()),
                        "group_exposure": output.group_exposure[row].cpu().tolist(),
                        "risk_after": output.next_risk[row].cpu().tolist(),
                    }
                )
        records.sort(key=lambda value: (str(value["user_id"]), int(value["round"])))
        return records

    def save_optimizer_state(self, path: str | Path) -> Path | None:
        if self.optimizer is None or self.scheduler is None:
            return None
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_steps": self.optimizer_steps,
            },
            destination,
        )
        return destination


def load_components(model: FRAG, path: str | Path, device: torch.device) -> None:
    source = Path(path)
    retrieval = torch.load(source / "retrieval.pt", map_location=device)
    model.scorer.load_state_dict(retrieval["scorer"])
    model.retriever.tau.data.copy_(retrieval["tau"].to(model.retriever.tau.device))
    generator_path = source / "generator"
    if model.config["model"]["generator"].get("backend", "llama") == "tiny":
        model.generator.load_state_dict(torch.load(generator_path, map_location=device))
        return
    from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

    state = load_peft_weights(generator_path, device=str(device))
    set_peft_model_state_dict(model.generator.model, state)
