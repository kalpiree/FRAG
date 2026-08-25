from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

from frag.config import load_config
from frag.runner import (
    MatrixJob,
    RunResult,
    aggregate_artifacts,
    apply_validation_mean,
    build_experiment_data,
    execute_matrix,
    expand_matrix,
    load_prepared,
    run_experiment,
    select_matrix_jobs,
    summarize_records,
)
from frag.training import SequenceExample, chronological_batches, validate_effective_batch


def _example(user: int, round_index: int) -> SequenceExample:
    return SequenceExample(
        user_index=user,
        user_id=f"u{user}",
        round=round_index,
        split="train",
        history=(1, 2),
        candidates=(1, 2, 3),
        target=3,
        domain="synthetic",
        target_key="i3",
    )


def _smoke_config(tmp_path: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    config = load_config([root / "configs/base.yaml", root / "configs/examples/smoke.yaml"])
    config = copy.deepcopy(config)
    config["project"]["output_root"] = str(tmp_path / "outputs")
    config["project"]["experiment"] = "test"
    config["dataset"]["processed_dir"] = str(tmp_path / "processed")
    config["dataset"]["synthetic_users"] = 3
    config["dataset"]["synthetic_items"] = 12
    config["dataset"]["synthetic_events_per_user"] = 9
    config["dataset"]["min_user_events"] = 5
    config["dataset"]["min_history"] = 2
    config["retrieval"]["tau_value"] = -1.0
    config["retrieval"]["kmax"] = 100.0
    config["training"]["retriever_pretrain_epochs"] = 1
    config["training"]["effective_batch_size"] = 4
    config["training"]["micro_batch_size"] = 2
    config["training"]["gradient_accumulation_steps"] = 2
    config["training"]["mixed_precision"] = "none"
    config["evaluation"]["cutoff"] = 2
    config["evaluation"]["cfd_horizon"] = 1
    config["runtime"]["save_optimizer"] = True
    return config


def test_temporal_batches_transpose_user_sequences() -> None:
    examples = [_example(0, 10), _example(0, 11), _example(1, 20), _example(1, 21)]
    batches = list(chronological_batches(examples, batch_size=2))
    assert len(batches) == 2
    assert batches[0].user_keys == ["u0", "u1"]
    assert batches[0].rounds.tolist() == [10, 20]
    assert batches[1].rounds.tolist() == [11, 21]


def test_effective_batch_validation() -> None:
    config = {
        "training": {
            "micro_batch_size": 2,
            "gradient_accumulation_steps": 3,
            "effective_batch_size": 6,
        }
    }
    assert validate_effective_batch(config) == (2, 3, 6)
    config["training"]["effective_batch_size"] = 5
    with pytest.raises(ValueError, match="effective_batch_size"):
        validate_effective_batch(config)


def test_exact_per_user_metrics_and_hard_cfd() -> None:
    records = []
    for user in ("u0", "u1"):
        records.extend(
            [
                {
                    "user_id": user,
                    "round": 0,
                    "target_id": 1,
                    "ranking": [1, 2],
                    "retrieved_ids": [1],
                },
                {
                    "user_id": user,
                    "round": 1,
                    "target_id": 2,
                    "ranking": [2, 1],
                    "retrieved_ids": [2],
                },
            ]
        )
    summary, per_user, cfd = summarize_records(
        records,
        {1: 0, 2: 1},
        [0.5, 0.5],
        2,
        2,
        "synthetic",
        "frag",
        0,
        "test",
    )
    assert summary["utility"] == {"recall": 1.0, "precision": 0.5, "mrr": 1.0, "ndcg": 1.0}
    assert summary["fairness"] == {"ed": 0.0, "wger": 1.0, "gc": 1.0}
    assert summary["cfd_average"] == [0.5, 0.0]
    assert len(per_user) == 2
    assert len(cfd) == 4


def test_full_smoke_writes_raw_external_ids_and_adapter_checkpoint(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    result = run_experiment(config, "full")
    train = Path(result.paths["train"])
    test = Path(result.paths["test"])
    assert (train / "checkpoint" / "retrieval.pt").is_file()
    assert (train / "checkpoint" / "generator").is_file()
    assert (train / "optimizer.pt").is_file()
    assert not list((train / "checkpoint").rglob("pytorch_model*.bin"))
    with (test / "predictions.jsonl").open("r", encoding="utf-8") as handle:
        record = json.loads(next(handle))
    assert record["target_key"].startswith("i")
    assert record["ranking_keys"]
    assert record["retrieved_keys"]


def test_prediction_artifact_can_be_disabled(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    config["runtime"]["save_predictions"] = False
    result = run_experiment(config, "full")
    assert not Path(result.paths["test"], "predictions.jsonl").exists()
    assert Path(result.paths["test"], "summary.json").is_file()


def test_full_run_scores_zero_exposure_without_crashing(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    config["retrieval"]["tau_value"] = 2.0
    result = run_experiment(config, "full")
    summary = result.summaries["validation"]
    assert summary["empty_ranking_count"] == summary["interaction_count"]
    assert summary["fairness"] == {"ed": 0.5, "wger": 0.0, "gc": 0.0}


def test_prepared_data_rejects_a_changed_preparation_config(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    load_prepared(config)
    config["dataset"]["train_ratio"] = 0.7
    with pytest.raises(ValueError, match="does not match"):
        load_prepared(config)


def test_external_import_uses_the_same_metric_contract(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    prepared = load_prepared(config)
    data = build_experiment_data(prepared, config)
    records = []
    for example in data.examples["test"]:
        retrieved = list(dict.fromkeys((example.target, *example.candidates)))
        records.append(
            {
                "user_id": example.user_id,
                "round": example.round,
                "target_id": example.target,
                "ranking": retrieved[:2],
                "retrieved_ids": retrieved,
            }
        )
    source = tmp_path / "external.jsonl"
    with source.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")
    config["model"]["method"] = "external_test"
    config["model"]["external"] = True
    result = run_experiment(config, "external", source)
    assert result.summaries["test"]["utility"]["recall"] == 1.0
    assert Path(result.paths["test"], "predictions.jsonl").is_file()


def test_external_import_rejects_a_changed_test_cohort(tmp_path: Path) -> None:
    config = _smoke_config(tmp_path)
    prepared = load_prepared(config)
    data = build_experiment_data(prepared, config)
    example = data.examples["test"][0]
    source = tmp_path / "external.jsonl"
    source.write_text(
        json.dumps(
            {
                "user_id": example.user_id,
                "round": example.round,
                "target_id": example.target,
                "ranking": [example.target],
                "retrieved_ids": [example.target],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config["model"]["external"] = True
    with pytest.raises(ValueError, match="test cohort"):
        run_experiment(config, "external", source)


def test_cfd_reports_empty_hard_set_as_undefined() -> None:
    records = [
        {
            "user_id": "u0",
            "round": 0,
            "target_id": 1,
            "ranking": [1],
            "retrieved_ids": [],
        }
    ]
    summary, _, rows = summarize_records(
        records,
        {1: 0},
        [1.0],
        1,
        1,
        "synthetic",
        "frag",
        0,
        "test",
    )
    assert summary["cfd_status"] == "undefined_zero_cumulative_hard_set"
    assert summary["cfd_user_count"] == 1
    assert summary["cfd_average"] is None
    assert rows == []


def test_cfd_allows_an_empty_round_after_positive_cumulative_exposure() -> None:
    records = [
        {
            "user_id": "u0",
            "round": 0,
            "target_id": 1,
            "ranking": [1],
            "retrieved_ids": [1],
        },
        {
            "user_id": "u0",
            "round": 1,
            "target_id": 1,
            "ranking": [1],
            "retrieved_ids": [],
        },
    ]
    summary, _, rows = summarize_records(
        records,
        {1: 0},
        [1.0],
        1,
        2,
        "synthetic",
        "frag",
        0,
        "test",
    )
    assert summary["cfd_status"] == "computed"
    assert summary["cfd_average"] == [0.0, 0.0]
    assert len(rows) == 2


def test_rq2_matrix_changes_one_sweep_axis_at_a_time() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = expand_matrix(
        root / "configs/experiments/rq2.yaml",
        root / "configs/base.yaml",
        root / "configs",
    )
    assert len(jobs) == 460
    assert sum(len(job.config["project"]["sweeps"]) for job in jobs) == 520
    assert sum(len(job.config["project"]["sweeps"]) == 4 for job in jobs) == 20
    assert all(job.sweep_path is not None for job in jobs)
    assert {job.method for job in jobs} == {"frag"}


def test_ablation_matrix_retains_distinct_method_labels() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = expand_matrix(
        root / "configs/experiments/rq3.yaml",
        root / "configs/base.yaml",
        root / "configs",
    )
    assert {job.config["model"]["method"] for job in jobs} == {
        "frag",
        "no_adaptive",
        "no_budget",
        "no_cumulative_state",
        "no_fairness",
        "no_joint",
    }


def test_rq4_matrix_applies_its_evaluation_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    jobs = expand_matrix(
        root / "configs/experiments/rq4.yaml",
        root / "configs/base.yaml",
        root / "configs",
    )
    assert all(job.config["evaluation"]["cfd_horizon"] == 20 for job in jobs)
    assert all(
        job.config["evaluation"]["exposure_source"] == "hard_retrieval"
        for job in jobs
    )


def test_matrix_job_selection() -> None:
    jobs = [MatrixJob(str(index), "d", "frag", index, {}, None, None) for index in range(3)]
    assert select_matrix_jobs(jobs, 1, None) == [jobs[1]]
    assert select_matrix_jobs(jobs, None, "2") == [jobs[2]]
    with pytest.raises(ValueError, match="between 0 and 2"):
        select_matrix_jobs(jobs, 3, None)
    with pytest.raises(ValueError, match="unknown job-id"):
        select_matrix_jobs(jobs, None, "missing")


def test_matrix_cli_documents_mutually_exclusive_array_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    namespace = runpy.run_path(str(root / "scripts/run_matrix.py"))
    parser = namespace["build_parser"]()
    with pytest.raises(SystemExit):
        parser.parse_args(["--experiment", "rq2.yaml", "--job-index", "0", "--job-id", "x"])
    help_text = parser.format_help()
    assert "zero-based index" in help_text
    assert "checkpoint and validation only" in help_text
    assert "prior validation mean for one selected" in help_text
    assert "fixed-K job" in help_text


def test_validation_mean_override_is_safe_and_nonmutating() -> None:
    fixed = MatrixJob(
        "fixed",
        "d",
        "fixed_relevance",
        0,
        {
            "model": {"external": False},
            "retrieval": {"fixed_k": "validation_mean"},
        },
    )
    selected = apply_validation_mean([fixed], 6.5)
    assert selected[0].config["retrieval"]["validation_mean"] == 6.5
    assert "validation_mean" not in fixed.config["retrieval"]
    with pytest.raises(ValueError, match="exactly one"):
        apply_validation_mean([fixed, fixed], 6.5)
    with pytest.raises(ValueError, match="positive and finite"):
        apply_validation_mean([fixed], float("nan"))
    adaptive = copy.deepcopy(fixed)
    adaptive.config["retrieval"]["fixed_k"] = None
    with pytest.raises(ValueError, match="fixed-K validation_mean"):
        apply_validation_mean([adaptive], 6.5)
    external = copy.deepcopy(fixed)
    external.config["model"]["external"] = True
    selected_external = apply_validation_mean([external], 6.5)
    assert selected_external[0].config["retrieval"]["validation_mean"] == 6.5


def test_execute_matrix_command_and_fixed_k_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    adaptive = MatrixJob(
        "adaptive",
        "d",
        "frag",
        0,
        {
            "project": {"experiment": "rq"},
            "model": {"method": "frag", "external": False},
            "retrieval": {"fixed_k": None},
        },
    )
    fixed = MatrixJob(
        "fixed",
        "d",
        "fixed_relevance",
        0,
        {
            "project": {"experiment": "rq"},
            "model": {"method": "fixed_relevance", "external": False},
            "retrieval": {"fixed_k": "validation_mean"},
        },
    )
    calls = []

    def run(config: dict, command: str, external_path: Path | None) -> RunResult:
        calls.append((config, command, external_path))
        summaries = (
            {"validation": {"mean_hard_retrieval_size": 7.0}}
            if config["model"]["method"] == "frag"
            else {}
        )
        return RunResult("d", config["model"]["method"], 0, command, {}, summaries)

    monkeypatch.setattr("frag.runner.run_experiment", run)
    with pytest.raises(ValueError, match="missing FRAG validation retrieval size"):
        execute_matrix([fixed], command="train")
    assert calls == []
    execute_matrix(apply_validation_mean([fixed], 8.5), command="train")
    assert calls[0][0]["retrieval"]["validation_mean"] == 8.5
    calls.clear()
    summary_fixed = copy.deepcopy(fixed)
    summary_fixed.config["retrieval"]["validation_summary"] = "validation.json"
    execute_matrix([summary_fixed], command="train")
    assert calls[0][0]["retrieval"]["validation_summary"] == "validation.json"
    calls.clear()
    execute_matrix([fixed, adaptive], command="train")
    assert [call[1] for call in calls] == ["train", "train"]
    assert calls[1][0]["retrieval"]["validation_mean"] == 7.0


def test_execute_matrix_rejects_external_train_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = MatrixJob(
        "external",
        "d",
        "base",
        0,
        {
            "project": {"experiment": "rq"},
            "model": {"method": "base", "external": True},
            "retrieval": {"fixed_k": None},
        },
    )
    called = False

    def run(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr("frag.runner.run_experiment", run)
    with pytest.raises(ValueError, match="require command full"):
        execute_matrix([external], command="train")
    assert not called


def test_aggregation_keeps_raw_records_and_paired_users(tmp_path: Path) -> None:
    for method, values in (("frag", [0.8, 0.9]), ("base", [0.5, 0.6])):
        path = tmp_path / method / "test"
        path.mkdir(parents=True)
        summary = {
            "dataset": "d",
            "method": method,
            "seed": 0,
            "split": "test",
            "utility": {
                "recall": values[0],
                "precision": values[0],
                "mrr": values[0],
                "ndcg": values[0],
            },
            "fairness": {"ed": 0.1, "wger": values[0], "gc": values[0]},
            "cfd_average": [0.5, 0.25],
        }
        with (path / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle)
        with (path / "per_user_metrics.jsonl").open("w", encoding="utf-8") as handle:
            for index, value in enumerate(values):
                row = {
                    "dataset": "d",
                    "method": method,
                    "seed": 0,
                    "split": "test",
                    "user_id": f"u{index}",
                    "recall": value,
                    "precision": value,
                    "mrr": value,
                    "ndcg": value,
                    "ed": 1.0 - value,
                    "wger": value,
                    "gc": value,
                }
                handle.write(json.dumps(row))
                handle.write("\n")
    result = aggregate_artifacts([tmp_path], baseline_method="base")
    assert result["aggregates"]
    assert result["cfd_aggregates"]
    assert result["paired_tests"]
    assert all("statistic" in row for row in result["paired_tests"])


def test_aggregation_expands_deduplicated_sweep_labels(tmp_path: Path) -> None:
    path = tmp_path / "frag" / "test"
    path.mkdir(parents=True)
    labels = [
        {"path": "fairness.lambda", "value": 1.0},
        {"path": "training.eta", "value": 0.1},
    ]
    summary = {
        "dataset": "d",
        "method": "frag",
        "seed": 0,
        "split": "test",
        "utility": {"recall": 0.5, "precision": 0.5, "mrr": 0.5, "ndcg": 0.5},
        "fairness": {"ed": 0.5, "wger": 0.5, "gc": 0.5},
        "cfd_average": [0.5],
        "sweeps": labels,
    }
    with (path / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle)
    result = aggregate_artifacts([tmp_path])
    aggregate_labels = {json.dumps(row["sweep"], sort_keys=True) for row in result["aggregates"]}
    cfd_labels = {json.dumps(row["sweep"], sort_keys=True) for row in result["cfd_aggregates"]}
    expected = {json.dumps(label, sort_keys=True) for label in labels}
    assert aggregate_labels == expected
    assert cfd_labels == expected
