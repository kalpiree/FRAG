import torch

from frag.modeling.frag import FRAG


def config() -> dict:
    return {
        "model": {
            "retriever": {
                "embedding_dim": 16,
                "history_layers": 1,
                "dropout": 0.0,
                "score_activation": "sigmoid",
            },
            "generator": {
                "backend": "tiny",
                "hidden_size": 16,
                "dropout": 0.0,
            },
        },
        "fairness": {
            "target_shares": [0.5, 0.5],
            "lambda": 0.5,
            "state_mode": "cumulative",
        },
        "retrieval": {
            "tau_value": 0.4,
            "gamma": 0.2,
            "kmax": 100.0,
            "fixed_k": None,
        },
        "training": {
            "joint_training": True,
            "eta": 0.1,
            "retriever_learning_rate": 0.001,
            "tau_learning_rate": 0.001,
            "learning_rate": 0.00002,
        },
    }


def test_generator_loss_reaches_scorer_and_tau() -> None:
    torch.manual_seed(7)
    item_groups = torch.tensor([0, 0, 1, 0, 1, 0, 1])
    titles = {index: f"item {index}" for index in range(1, 7)}
    model = FRAG(2, 6, item_groups, titles, config())
    output = model(
        user_ids=torch.tensor([0, 1]),
        timestamps=torch.tensor([1, 1]),
        histories=torch.tensor([[1, 2, 3], [2, 4, 6]]),
        history_mask=torch.ones(2, 3, dtype=torch.bool),
        candidates=torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]),
        candidate_mask=torch.ones(2, 6, dtype=torch.bool),
        targets=torch.tensor([4, 5]),
        domains=["synthetic", "synthetic"],
    )
    output.loss.backward()
    scorer_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.scorer.parameters()
        if parameter.grad is not None
    )
    assert scorer_gradient > 0.0
    assert model.retriever.tau.grad is not None
    assert model.retriever.tau.grad.abs().item() > 0.0
    assert model.risk.risk.grad is None


def test_state_is_resettable() -> None:
    item_groups = torch.tensor([0, 0, 1])
    model = FRAG(1, 2, item_groups, {1: "one", 2: "two"}, config())
    model.risk.risk.fill_(3.0)
    model.reset_risk()
    assert torch.equal(model.risk.risk, torch.zeros_like(model.risk.risk))


def test_separate_retriever_pretraining_can_optimize_tau() -> None:
    item_groups = torch.tensor([0, 0, 1, 0, 1])
    model = FRAG(1, 4, item_groups, {index: str(index) for index in range(1, 5)}, config())
    loss = model.pretrain_loss(
        histories=torch.tensor([[1, 2]]),
        history_mask=torch.tensor([[True, True]]),
        candidates=torch.tensor([[1, 2, 3, 4]]),
        candidate_mask=torch.ones(1, 4, dtype=torch.bool),
        targets=torch.tensor([4]),
        include_threshold=True,
    )
    loss.backward()
    assert model.retriever.tau.grad is not None
    assert model.retriever.tau.grad.abs().item() > 0.0
