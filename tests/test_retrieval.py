import pytest
import torch

from frag.modeling.fairness import (
    FairnessRiskState,
    candidate_mass_hinge_square,
    soft_group_exposure,
)
from frag.modeling.retriever import AdaptiveRetriever


def test_equation_11_weights() -> None:
    retriever = AdaptiveRetriever(
        target_shares=[0.5, 0.5],
        tau_init=0.3,
        gamma=0.5,
        lambda_fair=2.0,
    )
    scores = torch.tensor([[0.4, 0.4]])
    groups = torch.tensor([[0, 1]])
    risks = torch.tensor([[1.0, 3.0]])
    output = retriever(scores, groups, risks)
    expected_adjusted = torch.tensor([[-1.9, 2.1]])
    torch.testing.assert_close(output.fairness_adjusted_scores, expected_adjusted)
    torch.testing.assert_close(
        output.soft_weights,
        torch.sigmoid(expected_adjusted / 0.5),
    )


def test_half_boundary_is_inclusive() -> None:
    retriever = AdaptiveRetriever(
        target_shares=[0.5, 0.5],
        tau_init=0.3,
        gamma=0.1,
        lambda_fair=0.0,
    )
    scores = torch.tensor([[0.3, 0.299]])
    groups = torch.tensor([[0, 1]])
    output = retriever(scores, groups, torch.zeros(1, 2))
    assert output.soft_weights[0, 0] == 0.5
    assert output.hard_mask.tolist() == [[True, False]]


def test_candidate_mask_excludes_padding() -> None:
    retriever = AdaptiveRetriever(
        target_shares=[0.5, 0.5],
        tau_init=0.0,
        gamma=0.1,
        lambda_fair=0.0,
    )
    scores = torch.tensor([[1.0, 100.0]])
    groups = torch.tensor([[0, -1]])
    active = torch.tensor([[True, False]])
    output = retriever(scores, groups, torch.zeros(1, 2), active)
    assert output.soft_weights.tolist()[0][1] == 0.0
    assert output.hard_mask.tolist() == [[True, False]]
    assert torch.isneginf(output.fairness_adjusted_scores[0, 1])


def test_fixed_top_k_uses_fairness_adjusted_scores() -> None:
    retriever = AdaptiveRetriever(
        target_shares=[0.5, 0.5],
        tau_init=0.0,
        gamma=1.0,
        lambda_fair=1.0,
        selection_mode="fixed_top_k",
        fixed_k=2,
    )
    scores = torch.tensor([[0.9, 0.8, 0.7]])
    groups = torch.tensor([[0, 0, 1]])
    risks = torch.tensor([[0.0, 2.0]])
    output = retriever(scores, groups, risks)
    assert output.hard_mask.tolist() == [[True, False, True]]
    selected_order = output.ranked_indices[0][output.ranked_mask[0]].tolist()
    assert selected_order == [0, 2]


def test_cumulative_state_is_per_user_detached_and_chronological() -> None:
    state = FairnessRiskState(2, [0.5, 0.5], mode="cumulative")
    exposure = torch.tensor([[1.0, 0.0]], requires_grad=True)
    mass = torch.tensor([1.0], requires_grad=True)
    first = state.update([0], exposure, mass, timestamps=[0])
    torch.testing.assert_close(first, torch.tensor([[0.0, 0.5]]))
    assert not first.requires_grad
    torch.testing.assert_close(state.current([1]), torch.zeros(1, 2))
    second = state.update(
        [0],
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        timestamps=[1],
    )
    torch.testing.assert_close(second, torch.tensor([[0.0, 1.0]]))
    with pytest.raises(ValueError, match="timestamps"):
        state.update(
            [0],
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([1.0]),
            timestamps=[1],
        )


def test_one_step_state_does_not_accumulate() -> None:
    state = FairnessRiskState(1, [0.5, 0.5], mode="one_step")
    for timestamp in (0, 1):
        updated = state.update(
            [0],
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1.0]),
            timestamps=[timestamp],
        )
    torch.testing.assert_close(updated, torch.tensor([[0.0, 0.5]]))


def test_group_exposure_and_state_update_from_weights() -> None:
    weights = torch.tensor([[0.8, 0.2, 0.7]])
    groups = torch.tensor([[0, 1, 1]])
    active = torch.tensor([[True, True, False]])
    exposure, mass = soft_group_exposure(weights, groups, 2, active)
    torch.testing.assert_close(exposure, torch.tensor([[0.8, 0.2]]))
    torch.testing.assert_close(mass, torch.tensor([1.0]))
    state = FairnessRiskState(1, [0.5, 0.5])
    updated = state.update_from_weights([0], weights, groups, active, timestamps=[0])
    torch.testing.assert_close(updated, torch.tensor([[0.0, 0.3]]))


def test_equation_15_hinge_square() -> None:
    weights = torch.tensor([[1.0, 1.0, 0.5], [0.25, 0.25, 0.25]])
    losses = candidate_mass_hinge_square(weights, 2.0, reduction="none")
    torch.testing.assert_close(losses, torch.tensor([0.25, 0.0]))
    mean_loss = candidate_mass_hinge_square(weights, 2.0, reduction="mean")
    torch.testing.assert_close(mean_loss, torch.tensor(0.125))


def test_generator_signal_reaches_scores_and_tau() -> None:
    retriever = AdaptiveRetriever(
        target_shares=[0.5, 0.5],
        tau_init=0.3,
        gamma=0.2,
        lambda_fair=1.0,
    )
    scores = torch.tensor([[0.4, 0.5]], requires_grad=True)
    groups = torch.tensor([[0, 1]])
    risks = torch.tensor([[0.1, 0.4]], requires_grad=True)
    output = retriever(scores, groups, risks)
    loss = (output.soft_weights * torch.tensor([[1.0, -0.5]])).sum()
    loss.backward()
    assert scores.grad is not None
    assert torch.count_nonzero(scores.grad) == 2
    assert retriever.tau.grad is not None
    assert retriever.tau.grad.abs() > 0
    assert risks.grad is None
