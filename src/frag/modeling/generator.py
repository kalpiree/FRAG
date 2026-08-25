from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


@dataclass
class GeneratorOutput:
    loss: Tensor
    candidate_scores: Tensor | None


@dataclass
class PromptSpec:
    text: str
    spans: tuple[tuple[int, int, Tensor], ...]


def _validate_right_padding(mask: Tensor) -> None:
    if mask.ndim != 2:
        raise ValueError("mask must have shape [batch, sequence]")
    if mask.size(1) > 1 and torch.any(mask[:, 1:] & ~mask[:, :-1]):
        raise ValueError("sequence masks must use contiguous right padding")


class TinyGenerator(nn.Module):
    def __init__(self, num_items: int, config: dict[str, Any]) -> None:
        super().__init__()
        size = int(config.get("hidden_size", 64))
        dropout = float(config.get("dropout", 0.1))
        self.item_embeddings = nn.Embedding(num_items + 1, size, padding_idx=0)
        self.history = nn.GRU(size, size, batch_first=True)
        self.context = nn.Sequential(
            nn.Linear(size * 2, size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
        )
        self.output = nn.Linear(size, num_items + 1)

    def forward(
        self,
        histories: Tensor,
        history_mask: Tensor,
        candidates: Tensor,
        candidate_mask: Tensor,
        targets: Tensor,
        soft_weights: Tensor,
        hard_mask: Tensor,
        retrieval_scores: Tensor,
        titles: dict[int, str] | None = None,
        domains: list[str] | None = None,
    ) -> GeneratorOutput:
        _validate_right_padding(history_mask)
        history_embeddings = self.item_embeddings(histories)
        lengths = history_mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            history_embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, state = self.history(packed)
        history_context = state[-1]
        item_embeddings = self.item_embeddings(candidates)
        effective = soft_weights * hard_mask.to(soft_weights.dtype) * candidate_mask
        candidate_context = (item_embeddings * effective.unsqueeze(-1)).sum(dim=1)
        candidate_context = candidate_context / effective.sum(dim=1, keepdim=True).clamp_min(1.0)
        context = self.context(torch.cat([history_context, candidate_context], dim=-1))
        global_logits = self.output(context)
        candidate_scores = torch.gather(global_logits, 1, candidates)
        candidate_scores = candidate_scores.masked_fill(
            ~candidate_mask.bool(), torch.finfo(candidate_scores.dtype).min
        )
        loss = F.cross_entropy(global_logits, targets)
        return GeneratorOutput(loss=loss, candidate_scores=candidate_scores)

    def rank(
        self,
        histories: Tensor,
        history_mask: Tensor,
        candidates: Tensor,
        candidate_mask: Tensor,
        soft_weights: Tensor,
        hard_mask: Tensor,
        retrieval_scores: Tensor,
        titles: dict[int, str] | None = None,
        domains: list[str] | None = None,
    ) -> Tensor:
        targets = torch.zeros(histories.size(0), dtype=torch.long, device=histories.device)
        return self.forward(
            histories,
            history_mask,
            candidates,
            candidate_mask,
            targets,
            soft_weights,
            hard_mask,
            retrieval_scores,
            titles,
            domains,
        ).candidate_scores

    def save_adapter(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), destination)


class LlamaGenerator(nn.Module):
    templates = {
        "movielens": (
            "User history: {history}.\nCandidates:",
            "\nWhich movie is next?\nAnswer: ",
        ),
        "lastfm": (
            "User history: {history}.\nCandidates:",
            "\nWhich artist is next?\nAnswer: ",
        ),
        "steam": (
            "User history: {history}.\nCandidates:",
            "\nWhich game is next?\nAnswer: ",
        ),
        "goodreads": (
            "User history: {history}.\nCandidates:",
            "\nWhich book is next?\nAnswer: ",
        ),
        "synthetic": (
            "User history: {history}.\nCandidates:",
            "\nWhich item is next?\nAnswer: ",
        ),
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        source = str(config["model_id"])
        local_value = config.get("local_path")
        if local_value:
            local_path = Path(str(local_value))
            if local_path.is_dir() and (local_path / "config.json").is_file():
                source = str(local_path)
        revision = config.get("revision") if source == str(config["model_id"]) else None
        dtype = self._dtype(config.get("torch_dtype", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            revision=revision,
            use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for candidate span alignment")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            source,
            revision=revision,
            torch_dtype=dtype,
            attn_implementation=config.get("attn_implementation", "sdpa"),
            low_cpu_mem_usage=True,
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config["lora_rank"]),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            target_modules=list(config["lora_targets"]),
        )
        self.model = get_peft_model(base, lora)
        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        self.model.config.use_cache = False
        self.max_length = int(config.get("max_length", 512))
        self.max_answer_tokens = int(config.get("max_answer_tokens", 32))
        self.max_title_tokens = int(config.get("max_title_tokens", 12))
        self.max_history_items = int(config.get("max_history_items", 20))
        self.scoring_batch_size = int(config.get("scoring_batch_size", 4))
        self.ranking_mode = str(config.get("ranking_mode", "sequence_logprob"))
        self.include_candidates = bool(config.get("include_candidates", True))
        if self.max_length < 16 or self.max_answer_tokens < 1:
            raise ValueError("Invalid sequence length configuration")
        if self.scoring_batch_size < 1:
            raise ValueError("scoring_batch_size must be positive")

    @staticmethod
    def _dtype(value: str) -> torch.dtype:
        choices = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if value not in choices:
            raise ValueError(f"Unsupported dtype: {value}")
        return choices[value]

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _limited_title(self, title: str, limit: int) -> str:
        normalized = str(title).strip()
        if limit <= 0 or not normalized:
            return ""
        token_ids = self.tokenizer.encode(normalized, add_special_tokens=False)[:limit]
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

    def _item_text(self, item: int, title: str, title_limit: int) -> str:
        limited = self._limited_title(title, title_limit)
        prefix = f"[item={item}]"
        return prefix if not limited else f"{prefix} {limited}"

    def _render_prompt(
        self,
        history_ids: list[int],
        selected: list[tuple[int, Tensor]],
        titles: dict[int, str],
        domain: str,
        history_limit: int,
        title_limit: int,
    ) -> PromptSpec:
        template = self.templates.get(domain, self.templates["synthetic"])
        retained_history = history_ids[-history_limit:] if history_limit else []
        history_text = ", ".join(
            self._item_text(item, titles.get(item, f"item_{item}"), title_limit)
            for item in retained_history
        )
        if not history_text:
            history_text = "None"
        text = template[0].format(history=history_text)
        spans = []
        if getattr(self, "include_candidates", True):
            if not selected:
                text += "\nNone"
            for rank, (item, weight) in enumerate(selected, start=1):
                text += f"\n{rank}. "
                start = len(text)
                text += self._item_text(item, titles.get(item, f"item_{item}"), title_limit)
                spans.append((start, len(text), weight))
        else:
            text = text.removesuffix("\nCandidates:")
        text += template[1]
        return PromptSpec(text=text, spans=tuple(spans))

    def _prompt_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _prompt(
        self,
        history: Tensor,
        history_mask: Tensor,
        candidates: Tensor,
        candidate_mask: Tensor,
        soft_weights: Tensor,
        hard_mask: Tensor,
        retrieval_scores: Tensor,
        titles: dict[int, str],
        domain: str,
    ) -> PromptSpec:
        _validate_right_padding(history_mask.unsqueeze(0))
        history_ids = [int(item) for item in history[history_mask.bool()].tolist()]
        selected_mask = candidate_mask.bool() & hard_mask.bool()
        order = torch.argsort(
            retrieval_scores.masked_fill(~selected_mask, -torch.inf),
            descending=True,
            stable=True,
        )
        selected = [
            (int(candidates[index]), soft_weights[index])
            for index in order.tolist()
            if bool(selected_mask[index])
        ]
        history_limit = min(len(history_ids), self.max_history_items)
        minimum_history = 1 if history_ids else 0
        title_limit = self.max_title_tokens
        prompt_budget = self.max_length - self.max_answer_tokens - 1
        while True:
            specification = self._render_prompt(
                history_ids,
                selected,
                titles,
                domain,
                history_limit,
                title_limit,
            )
            if self._prompt_token_count(specification.text) <= prompt_budget:
                return specification
            if history_limit > minimum_history:
                history_limit -= 1
                continue
            if title_limit > 0:
                title_limit -= 1
                continue
            raise RuntimeError(
                "All hard-retrieved candidates cannot fit in the configured LLM context"
            )

    def _tokenized(self, text: str) -> tuple[Tensor, list[tuple[int, int]]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        token_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=self.device)
        offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
        return token_ids, offsets

    def _example(
        self,
        prompt: PromptSpec,
        item: int,
        titles: dict[int, str],
    ) -> tuple[Tensor, Tensor]:
        answer_text = self._item_text(
            item,
            titles.get(item, f"item_{item}"),
            self.max_title_tokens,
        )
        answer_start = len(prompt.text)
        token_ids, offsets = self._tokenized(prompt.text + answer_text)
        answer_positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > answer_start and end > start
        ][: self.max_answer_tokens]
        if not answer_positions:
            raise RuntimeError("Tokenizer produced no next-item answer tokens")
        last_position = answer_positions[-1]
        token_ids = token_ids[: last_position + 1]
        offsets = offsets[: last_position + 1]
        scales = []
        for start, end in offsets:
            scale = torch.ones((), device=self.device)
            for span_start, span_end, weight in prompt.spans:
                if end > span_start and start < span_end and end > start:
                    scale = weight.to(device=self.device, dtype=torch.float32)
                    break
            scales.append(scale)
        scale_tensor = torch.stack(scales)
        embeddings = self.model.get_input_embeddings()(token_ids)
        embeddings = embeddings * scale_tensor.to(embeddings.dtype).unsqueeze(1)
        labels = torch.full_like(token_ids, -100)
        labels[answer_positions] = token_ids[answer_positions]
        eos_id = int(self.tokenizer.eos_token_id)
        eos_tensor = torch.tensor([eos_id], dtype=torch.long, device=self.device)
        eos_embedding = self.model.get_input_embeddings()(eos_tensor)
        embeddings = torch.cat([embeddings, eos_embedding], dim=0)
        labels = torch.cat([labels, eos_tensor], dim=0)
        if embeddings.size(0) > self.max_length:
            raise RuntimeError("Encoded prompt and answer exceed the configured LLM context")
        return embeddings, labels

    def _pad(
        self,
        examples: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        embeddings = pad_sequence([item[0] for item in examples], batch_first=True)
        labels = pad_sequence([item[1] for item in examples], batch_first=True, padding_value=-100)
        lengths = torch.tensor([item[0].size(0) for item in examples], device=self.device)
        positions = torch.arange(embeddings.size(1), device=self.device).unsqueeze(0)
        attention = positions < lengths.unsqueeze(1)
        return embeddings, attention, labels

    def _sequence_nll(self, embeddings: Tensor, attention: Tensor, labels: Tensor) -> Tensor:
        output = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention,
            use_cache=False,
        )
        logits = output.logits[:, :-1].float()
        shifted_labels = labels[:, 1:]
        valid = shifted_labels.ne(-100)
        token_nll = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shifted_labels.clamp_min(0).reshape(-1),
            reduction="none",
        ).reshape_as(shifted_labels)
        return (token_nll * valid).sum(dim=1)

    def forward(
        self,
        histories: Tensor,
        history_mask: Tensor,
        candidates: Tensor,
        candidate_mask: Tensor,
        targets: Tensor,
        soft_weights: Tensor,
        hard_mask: Tensor,
        retrieval_scores: Tensor,
        titles: dict[int, str],
        domains: list[str],
    ) -> GeneratorOutput:
        examples = []
        for index in range(histories.size(0)):
            prompt = self._prompt(
                histories[index],
                history_mask[index],
                candidates[index],
                candidate_mask[index],
                soft_weights[index],
                hard_mask[index],
                retrieval_scores[index],
                titles,
                domains[index],
            )
            examples.append(self._example(prompt, int(targets[index]), titles))
        embeddings, attention, labels = self._pad(examples)
        loss = self._sequence_nll(embeddings, attention, labels).mean()
        return GeneratorOutput(loss=loss, candidate_scores=None)

    @torch.no_grad()
    def rank(
        self,
        histories: Tensor,
        history_mask: Tensor,
        candidates: Tensor,
        candidate_mask: Tensor,
        soft_weights: Tensor,
        hard_mask: Tensor,
        retrieval_scores: Tensor,
        titles: dict[int, str],
        domains: list[str],
    ) -> Tensor:
        scores = torch.full_like(retrieval_scores, -torch.inf)
        for batch_index in range(histories.size(0)):
            prompt = self._prompt(
                histories[batch_index],
                history_mask[batch_index],
                candidates[batch_index],
                candidate_mask[batch_index],
                soft_weights[batch_index],
                hard_mask[batch_index],
                retrieval_scores[batch_index],
                titles,
                domains[batch_index],
            )
            selected = candidate_mask[batch_index].bool() & hard_mask[batch_index].bool()
            indices = selected.nonzero(as_tuple=False).flatten().tolist()
            if self.ranking_mode == "retrieval":
                scores[batch_index, selected] = retrieval_scores[batch_index, selected]
                continue
            for start in range(0, len(indices), self.scoring_batch_size):
                chunk = indices[start : start + self.scoring_batch_size]
                examples = [
                    self._example(prompt, int(candidates[batch_index, index]), titles)
                    for index in chunk
                ]
                if not examples:
                    continue
                embeddings, attention, labels = self._pad(examples)
                sequence_scores = -self._sequence_nll(embeddings, attention, labels)
                for index, score in zip(chunk, sequence_scores, strict=True):
                    scores[batch_index, index] = score.to(scores.dtype)
        return scores

    def save_adapter(self, path: str | Path) -> None:
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


def build_generator(num_items: int, config: dict[str, Any]) -> nn.Module:
    backend = config.get("backend", "llama")
    if backend == "tiny":
        return TinyGenerator(num_items, config)
    if backend == "llama":
        return LlamaGenerator(config)
    raise ValueError(f"Unknown generator backend: {backend}")
