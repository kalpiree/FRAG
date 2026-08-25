from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from frag.modeling.generator import LlamaGenerator


class FakeFastTokenizer:
    is_fast = True
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __init__(self) -> None:
        self.character_ids: dict[str, int] = {}
        self.id_characters: dict[int, str] = {}
        self.tokenized_texts: list[str] = []
        self.last_input_ids: list[int] = []
        self.last_offsets: list[tuple[int, int]] = []

    def _character_id(self, value: str) -> int:
        if value not in self.character_ids:
            token_id = len(self.character_ids) + 3
            self.character_ids[value] = token_id
            self.id_characters[token_id] = value
        return self.character_ids[value]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        values = [self._character_id(value) for value in text]
        return [self.bos_token_id, *values] if add_special_tokens else values

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        values = []
        for token_id in token_ids:
            numeric = int(token_id)
            if skip_special_tokens and numeric in {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            }:
                continue
            values.append(self.id_characters.get(numeric, ""))
        return "".join(values)

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        input_ids = self.encode(text, add_special_tokens=add_special_tokens)
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            offsets = [(0, 0), *offsets]
        self.tokenized_texts.append(text)
        self.last_input_ids = input_ids
        self.last_offsets = offsets
        return {"input_ids": input_ids, "offset_mapping": offsets}


class FakeCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 512, hidden_size: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        nn.init.ones_(self.embeddings.weight)
        self.forward_batch_sizes: list[int] = []

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
    ) -> SimpleNamespace:
        self.forward_batch_sizes.append(inputs_embeds.size(0))
        shape = (*inputs_embeds.shape[:2], self.vocab_size)
        logits = inputs_embeds.new_zeros(shape)
        return SimpleNamespace(logits=logits)


def _generator(**overrides: object) -> LlamaGenerator:
    generator = LlamaGenerator.__new__(LlamaGenerator)
    nn.Module.__init__(generator)
    generator.tokenizer = FakeFastTokenizer()
    generator.model = FakeCausalLM()
    values = {
        "max_length": 512,
        "max_answer_tokens": 128,
        "max_title_tokens": 32,
        "max_history_items": 20,
        "scoring_batch_size": 2,
        "ranking_mode": "sequence_logprob",
    }
    values.update(overrides)
    for name, value in values.items():
        setattr(generator, name, value)
    return generator


def test_candidate_spans_use_whole_string_offsets_and_scale_gradients() -> None:
    generator = _generator()
    titles = {1: "History", 2: "Candidate", 3: "Target"}
    weight = torch.tensor(0.25, requires_grad=True)
    prompt = generator._render_prompt(
        [1],
        [(2, weight)],
        titles,
        "synthetic",
        history_limit=1,
        title_limit=32,
    )
    answer = generator._item_text(3, titles[3], generator.max_title_tokens)
    embeddings, _ = generator._example(prompt, 3, titles)
    tokenizer = generator.tokenizer
    assert tokenizer.tokenized_texts == [prompt.text + answer]
    span_start, span_end, _ = prompt.spans[0]
    positions = [
        index
        for index, (start, end) in enumerate(tokenizer.last_offsets)
        if end > span_start and start < span_end and end > start
    ]
    token_ids = torch.tensor(tokenizer.last_input_ids)
    unscaled = generator.model.get_input_embeddings()(token_ids)
    assert len(positions) == span_end - span_start
    assert torch.allclose(embeddings[positions], unscaled[positions] * weight)
    embeddings[positions].sum().backward()
    assert weight.grad is not None
    assert weight.grad.item() > 0.0


def test_answer_labels_include_unique_item_ids_and_eos() -> None:
    generator = _generator()
    prompt = generator._render_prompt([], [], {}, "synthetic", 0, 32)
    titles = {7: "Same title", 8: "Same title"}
    _, first_labels = generator._example(prompt, 7, titles)
    first = first_labels[first_labels.ne(-100)]
    _, second_labels = generator._example(prompt, 8, titles)
    second = second_labels[second_labels.ne(-100)]
    assert int(first[-1]) == generator.tokenizer.eos_token_id
    assert int(second[-1]) == generator.tokenizer.eos_token_id
    assert generator.tokenizer.decode(first[:-1]) == "[item=7] Same title"
    assert generator.tokenizer.decode(second[:-1]) == "[item=8] Same title"
    assert not torch.equal(first, second)


def test_context_overflow_raises_without_dropping_hard_candidates() -> None:
    generator = _generator(max_title_tokens=0, max_history_items=0, max_answer_tokens=8)
    weights = [torch.tensor(1.0) for _ in range(3)]
    selected = list(zip([101, 102, 103], weights, strict=True))
    one_candidate = generator._render_prompt([], selected[:1], {}, "synthetic", 0, 0)
    all_candidates = generator._render_prompt([], selected, {}, "synthetic", 0, 0)
    one_count = generator._prompt_token_count(one_candidate.text)
    all_count = generator._prompt_token_count(all_candidates.text)
    generator.max_length = one_count + generator.max_answer_tokens + 1
    assert all_count > one_count
    with pytest.raises(RuntimeError, match="All hard-retrieved candidates cannot fit"):
        generator._prompt(
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.bool),
            torch.tensor([101, 102, 103]),
            torch.ones(3, dtype=torch.bool),
            torch.ones(3),
            torch.ones(3, dtype=torch.bool),
            torch.tensor([3.0, 2.0, 1.0]),
            {},
            "synthetic",
        )


def test_sequence_nll_sums_token_losses_per_example() -> None:
    generator = _generator()
    embeddings = torch.zeros(2, 5, 4)
    attention = torch.ones(2, 5, dtype=torch.bool)
    labels = torch.tensor(
        [
            [-100, 3, 4, -100, -100],
            [-100, 5, 6, 7, -100],
        ]
    )
    values = generator._sequence_nll(embeddings, attention, labels)
    unit = math.log(generator.model.vocab_size)
    assert torch.allclose(values, torch.tensor([2.0 * unit, 3.0 * unit]))


def test_ranking_scores_candidates_in_configured_chunks() -> None:
    generator = _generator(scoring_batch_size=2)
    candidates = torch.tensor([[3, 4, 5, 6, 7, 8]])
    hard_mask = torch.tensor([[True, True, True, True, True, False]])
    scores = generator.rank(
        histories=torch.tensor([[1, 2]]),
        history_mask=torch.tensor([[True, True]]),
        candidates=candidates,
        candidate_mask=torch.ones_like(candidates, dtype=torch.bool),
        soft_weights=torch.ones_like(candidates, dtype=torch.float32),
        hard_mask=hard_mask,
        retrieval_scores=torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]),
        titles={index: f"Item {index}" for index in range(1, 9)},
        domains=["synthetic"],
    )
    assert generator.model.forward_batch_sizes == [2, 2, 1]
    assert torch.isfinite(scores[0, :5]).all()
    assert torch.isneginf(scores[0, 5])


def test_llama_prompt_rejects_noncontiguous_right_padding() -> None:
    generator = _generator()
    with pytest.raises(ValueError, match="contiguous right padding"):
        generator._prompt(
            torch.tensor([1, 0, 2]),
            torch.tensor([True, False, True]),
            torch.tensor([3]),
            torch.tensor([True]),
            torch.tensor([1.0]),
            torch.tensor([True]),
            torch.tensor([1.0]),
            {1: "One", 2: "Two", 3: "Three"},
            "synthetic",
        )
