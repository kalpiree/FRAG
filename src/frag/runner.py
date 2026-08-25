from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from frag.artifacts import RunArtifacts
from frag.baselines import load_baseline_records
from frag.config import config_hash, load_config, merge_dicts, set_nested
from frag.data.prepare import PreparedData, preparation_config, prepare_dataset
from frag.data.runtime import PreparedTables, SequenceDataset, prepared_tables
from frag.data.runtime import load_prepared as load_prepared_tables
from frag.evaluation.fairness import aggregate_fairness, per_user_exposure_fairness
from frag.evaluation.rq4 import average_cfd_trajectory, per_user_cfd_trajectory
from frag.evaluation.significance import mean_confidence_interval, paired_t_test_by_user
from frag.evaluation.utility import aggregate_utility, per_user_utility
from frag.modeling.frag import FRAG
from frag.runtime import seed_everything
from frag.training import SequenceExample, Trainer, load_components, resolve_device


@dataclass
class ExperimentData:
    num_users: int
    num_items: int
    item_groups: torch.Tensor
    item_group_map: dict[int, int]
    titles: dict[int, str]
    examples: dict[str, list[SequenceExample]]
    user_to_index: dict[str, int]
    item_to_index: dict[str, int]
    index_to_item: tuple[str | None, ...]
    manifest: dict[str, Any]


@dataclass
class RunResult:
    dataset: str
    method: str
    seed: int
    command: str
    paths: dict[str, str]
    summaries: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "command": self.command,
            "paths": self.paths,
            "summaries": self.summaries,
        }


@dataclass
class MatrixJob:
    job_id: str
    dataset: str
    method: str
    seed: int
    config: dict[str, Any]
    sweep_path: str | None = None
    sweep_value: Any = None

    def record(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dataset": self.dataset,
            "method": self.method,
            "seed": self.seed,
            "sweep_path": self.sweep_path,
            "sweep_value": self.sweep_value,
            "sweeps": self.config.get("project", {}).get("sweeps", []),
            "config_hash": config_hash(self.config),
        }


def load_prepared(config: dict[str, Any]) -> PreparedTables:
    destination = Path(config["dataset"]["processed_dir"])
    required = tuple(
        destination / name
        for name in ("interactions.parquet", "items.parquet", "groups.parquet", "manifest.json")
    )
    if all(path.is_file() for path in required):
        tables = load_prepared_tables(destination)
        expected = preparation_config(config)
        actual = tables.manifest.get("preparation_config")
        if actual != expected:
            raise ValueError(
                f"prepared dataset at {destination} does not match the current dataset config; "
                "run prepare_data again or select a different processed_dir"
            )
        return tables
    return prepared_tables(prepare_dataset(config, write=True))


def build_experiment_data(
    prepared: PreparedTables | PreparedData,
    config: dict[str, Any],
) -> ExperimentData:
    tables = prepared_tables(prepared)
    minimum_history = int(config["dataset"].get("min_history", 1))
    examples = {"train": [], "validation": [], "test": []}
    for split in examples:
        dataset = SequenceDataset(
            tables,
            split,
            minimum_history,
            config.get("candidate_pool", {}),
        )
        for index in range(len(dataset)):
            row = dataset[index]
            examples[split].append(
                SequenceExample(
                    user_index=int(row["user_id"]),
                    user_id=str(row["user_key"]),
                    round=int(row["timestamp"]),
                    split=split,
                    history=tuple(int(value) for value in row["history"]),
                    candidates=tuple(int(value) for value in row["candidates"]),
                    target=int(row["target"]),
                    domain=str(row["domain"]),
                    target_key=str(row["target_key"]),
                )
            )
        examples[split].sort(key=lambda value: (value.round, value.user_index))
    group_map = {index: int(tables.item_groups[index]) for index in range(1, tables.num_items + 1)}
    return ExperimentData(
        num_users=tables.num_users,
        num_items=tables.num_items,
        item_groups=tables.item_groups,
        item_group_map=group_map,
        titles=tables.titles,
        examples=examples,
        user_to_index=tables.user_to_index,
        item_to_index=tables.item_to_index,
        index_to_item=tables.index_to_item,
        manifest=tables.manifest,
    )


def summarize_records(
    records: Sequence[dict[str, Any]],
    item_groups: dict[int, int],
    target_shares: Sequence[float],
    cutoff: int,
    cfd_horizon: int,
    dataset: str,
    method: str,
    seed: int,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        raise ValueError("evaluation records are empty")
    ordered = sorted(records, key=lambda row: (str(row["user_id"]), int(row["round"])))
    rankings: dict[str, list[list[int]]] = {}
    targets: dict[str, list[int]] = {}
    hard_sets: dict[str, list[list[int]]] = {}
    for record in ordered:
        user = str(record["user_id"])
        rankings.setdefault(user, []).append([int(value) for value in record["ranking"]])
        targets.setdefault(user, []).append(int(record["target_id"]))
        hard_sets.setdefault(user, []).append([int(value) for value in record["retrieved_ids"]])
    target_map = {group: float(value) for group, value in enumerate(target_shares)}
    per_utility = per_user_utility(rankings, targets, k=cutoff)
    per_fairness = per_user_exposure_fairness(
        rankings,
        item_groups,
        target_map,
        cutoff=cutoff,
    )
    utility = aggregate_utility(per_utility)
    fairness = aggregate_fairness(per_fairness)
    per_user_rows = []
    for user in sorted(per_utility):
        row = {
            "dataset": dataset,
            "method": method,
            "seed": seed,
            "split": split,
            "user_id": user,
            **per_utility[user],
            "ed": per_fairness[user]["ed"],
            "wger": per_fairness[user]["wger"],
            "gc": per_fairness[user]["gc"],
            "interaction_count": per_fairness[user]["interaction_count"],
            "shares": per_fairness[user]["shares"],
        }
        per_user_rows.append(row)
    eligible = {user: rounds for user, rounds in hard_sets.items() if len(rounds) >= cfd_horizon}
    cfd_rows = []
    cfd_average: list[float] | None = None
    nonzero_initial_mass = all(rounds[0] for rounds in eligible.values())
    cfd_status = "insufficient_history"
    if eligible and nonzero_initial_mass:
        cfd_status = "computed"
        per_cfd = per_user_cfd_trajectory(
            eligible,
            item_groups,
            target_map,
            horizon=cfd_horizon,
        )
        cfd_average = average_cfd_trajectory(
            eligible,
            item_groups,
            target_map,
            horizon=cfd_horizon,
        )
        for user, values in sorted(per_cfd.items()):
            for round_index, value in enumerate(values, start=1):
                cfd_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "split": split,
                        "user_id": user,
                        "round": round_index,
                        "cfd": value,
                    }
                )
    elif eligible:
        cfd_status = "undefined_zero_cumulative_hard_set"
    summary = {
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "split": split,
        "cutoff": cutoff,
        "user_count": len(per_user_rows),
        "interaction_count": len(records),
        "utility": utility,
        "fairness": fairness,
        "mean_hard_retrieval_size": sum(len(row["retrieved_ids"]) for row in records)
        / len(records),
        "short_ranking_count": sum(len(row["ranking"]) < cutoff for row in records),
        "empty_ranking_count": sum(not row["ranking"] for row in records),
        "cfd_horizon": cfd_horizon,
        "cfd_average": cfd_average,
        "cfd_user_count": len(eligible),
        "cfd_status": cfd_status,
    }
    return summary, per_user_rows, cfd_rows


def write_evaluation_artifacts(
    config: dict[str, Any],
    split: str,
    records: Sequence[dict[str, Any]],
    data: ExperimentData,
) -> tuple[Path, dict[str, Any]]:
    dataset = str(config["dataset"]["name"])
    method = str(config["model"]["method"])
    seed = int(config["training"]["seed"])
    evaluation = config["evaluation"]
    summary, per_user, cfd = summarize_records(
        records,
        data.item_group_map,
        config["fairness"]["target_shares"],
        int(evaluation["cutoff"]),
        int(evaluation["cfd_horizon"]),
        dataset,
        method,
        seed,
        split,
    )
    sweeps = config.get("project", {}).get("sweeps", [])
    run_hash = config_hash(config)
    summary["config_hash"] = run_hash
    summary["sweeps"] = sweeps
    for row in per_user:
        row["config_hash"] = run_hash
        row["sweeps"] = sweeps
    for row in cfd:
        row["config_hash"] = run_hash
        row["sweeps"] = sweeps
    artifacts = RunArtifacts(config, split)
    if bool(config.get("runtime", {}).get("save_predictions", True)):
        artifacts.jsonl("predictions.jsonl", records)
    artifacts.jsonl("per_user_metrics.jsonl", per_user)
    artifacts.jsonl("cfd.jsonl", cfd)
    artifacts.json("summary.json", summary)
    return artifacts.path, summary


def _resolve_fixed_k(config: dict[str, Any], data: ExperimentData) -> None:
    value = config["retrieval"].get("fixed_k")
    if value == 0:
        config["retrieval"]["fixed_k"] = data.num_items
        return
    if value != "validation_mean":
        return
    explicit = config["retrieval"].get("validation_mean")
    if explicit is not None:
        config["retrieval"]["fixed_k"] = max(1, int(round(float(explicit))))
        return
    source = config["retrieval"].get("validation_summary")
    if source is None:
        raise ValueError("validation_mean fixed_k requires validation_mean or validation_summary")
    with Path(source).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    config["retrieval"]["fixed_k"] = max(
        1,
        int(round(float(summary["mean_hard_retrieval_size"]))),
    )


def _load_retriever_pretrained(model: FRAG, config: dict[str, Any], device: torch.device) -> None:
    source = config["model"]["retriever"].get("pretrained_path")
    if not source:
        return
    payload = torch.load(Path(source), map_location=device)
    state = payload.get("scorer", payload)
    model.scorer.load_state_dict(state)


def _checkpoint_path(config: dict[str, Any]) -> Path:
    explicit = config.get("runtime", {}).get("checkpoint")
    if explicit:
        return Path(explicit)
    return RunArtifacts(config, "train").path / "checkpoint"


def _external_path(config: dict[str, Any], external_path: str | Path | None) -> Path:
    value = external_path or config["model"].get("predictions_path")
    if value is None:
        raise ValueError("external methods require a predictions JSONL path")
    return Path(value)


def _validate_external_records(
    records: Sequence[dict[str, Any]],
    examples: Sequence[SequenceExample],
    data: ExperimentData,
    config: dict[str, Any],
) -> None:
    expected = {(example.user_id, example.round): example for example in examples}
    actual = {(str(record["user_id"]), int(record["round"])): record for record in records}
    if len(actual) != len(records):
        raise ValueError("external predictions contain duplicate user-round keys")
    missing = sorted(set(expected).difference(actual))
    extra = sorted(set(actual).difference(expected))
    if missing or extra:
        raise ValueError(
            f"external predictions do not match the test cohort: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    fixed_k = config["retrieval"].get("fixed_k")
    cutoff = int(config["evaluation"]["cutoff"])
    for key, example in expected.items():
        record = actual[key]
        if str(record.get("split", "test")) != "test":
            raise ValueError(f"external prediction {key} has a non-test split")
        if int(record["target_id"]) != example.target:
            raise ValueError(f"external prediction {key} has the wrong target")
        ranking = [int(value) for value in record["ranking"]]
        retrieved = [int(value) for value in record["retrieved_ids"]]
        if not ranking:
            raise ValueError(f"external prediction {key} has an empty ranking")
        if not retrieved:
            raise ValueError(f"external prediction {key} has an empty retrieved set")
        if len(ranking) != len(set(ranking)) or len(retrieved) != len(set(retrieved)):
            raise ValueError(f"external prediction {key} contains duplicate item IDs")
        known = set(range(1, data.num_items + 1))
        if not set(ranking).issubset(known) or not set(retrieved).issubset(known):
            raise ValueError(f"external prediction {key} contains an unknown item ID")
        pool = set(example.candidates)
        if not set(retrieved).issubset(pool):
            raise ValueError(f"external prediction {key} leaves the shared candidate pool")
        if not set(ranking[:cutoff]).issubset(set(retrieved)):
            raise ValueError(f"external prediction {key} ranks an unretrieved item")
        if isinstance(fixed_k, int):
            expected_size = min(fixed_k, len(example.candidates))
            if len(retrieved) != expected_size:
                raise ValueError(
                    f"external prediction {key} has retrieval size {len(retrieved)}; "
                    f"expected {expected_size}"
                )


def run_experiment(
    config: dict[str, Any],
    command: str = "full",
    external_path: str | Path | None = None,
) -> RunResult:
    if command not in {"train", "evaluate", "full", "external"}:
        raise ValueError("command must be train, evaluate, full, or external")
    resolved = copy.deepcopy(config)
    seed = int(resolved["training"]["seed"])
    seed_everything(seed, bool(resolved.get("runtime", {}).get("deterministic", True)))
    prepared = load_prepared(resolved)
    data = build_experiment_data(prepared, resolved)
    dataset = str(resolved["dataset"]["name"])
    method = str(resolved["model"]["method"])
    paths: dict[str, str] = {}
    summaries: dict[str, dict[str, Any]] = {}
    if bool(resolved["model"].get("external", False)) or command == "external":
        _resolve_fixed_k(resolved, data)
        records = load_baseline_records(_external_path(resolved, external_path))
        normalized = [
            {
                **record,
                "user_id": str(record["user_id"]),
                "target_id": int(record["target_id"]),
                "split": str(record.get("split", "test")),
            }
            for record in records
        ]
        _validate_external_records(normalized, data.examples["test"], data, resolved)
        path, summary = write_evaluation_artifacts(resolved, "test", normalized, data)
        paths["test"] = str(path)
        summaries["test"] = summary
        return RunResult(dataset, method, seed, command, paths, summaries)
    _resolve_fixed_k(resolved, data)
    device = resolve_device(resolved)
    model = FRAG(
        data.num_users,
        data.num_items,
        data.item_groups,
        data.titles,
        resolved,
    )
    trainer = Trainer(model, resolved, device)
    if command in {"train", "full"}:
        _load_retriever_pretrained(model, resolved, device)
        pretrain = trainer.pretrain_retriever(data.examples["train"])
        if not bool(resolved["training"].get("joint_training", True)):
            model.freeze_retrieval()
        training = trainer.fit(data.examples["train"])
        train_artifacts = RunArtifacts(resolved, "train")
        train_artifacts.json("pretraining.json", pretrain)
        train_artifacts.json("training.json", training)
        train_artifacts.json("data_manifest.json", data.manifest)
        checkpoint = train_artifacts.path / "checkpoint"
        model.save_components(checkpoint)
        if bool(resolved.get("runtime", {}).get("save_optimizer", False)):
            trainer.save_optimizer_state(train_artifacts.path / "optimizer.pt")
        paths["train"] = str(train_artifacts.path)
        validation_records = trainer.predict(
            data.examples["validation"],
            int(resolved["evaluation"]["cutoff"]),
            "validation",
            data.index_to_item,
        )
        validation_path, validation_summary = write_evaluation_artifacts(
            resolved,
            "validation",
            validation_records,
            data,
        )
        paths["validation"] = str(validation_path)
        summaries["validation"] = validation_summary
    if command == "evaluate":
        load_components(model, _checkpoint_path(resolved), device)
    if command in {"evaluate", "full"}:
        test_records = trainer.predict(
            data.examples["test"],
            int(resolved["evaluation"]["cutoff"]),
            "test",
            data.index_to_item,
        )
        test_path, test_summary = write_evaluation_artifacts(
            resolved,
            "test",
            test_records,
            data,
        )
        paths["test"] = str(test_path)
        summaries["test"] = test_summary
    return RunResult(dataset, method, seed, command, paths, summaries)


def _method_path(config_root: Path, method: str) -> Path:
    direct = config_root / "methods" / f"{method}.yaml"
    if direct.is_file():
        return direct
    ablation = config_root / "ablations" / f"{method}.yaml"
    if ablation.is_file():
        return ablation
    raise FileNotFoundError(f"no method or ablation config for {method}")


def _job(
    base: dict[str, Any],
    experiment: dict[str, Any],
    config_root: Path,
    dataset: str,
    method: str,
    seed: int,
    sweep_path: str | None = None,
    sweep_value: Any = None,
) -> MatrixJob:
    dataset_config = load_config([config_root / "datasets" / f"{dataset}.yaml"])
    method_config = load_config([_method_path(config_root, method)])
    config = merge_dicts(merge_dicts(base, dataset_config), method_config)
    config.setdefault("project", {})["experiment"] = str(experiment["name"])
    config["model"]["method"] = method
    config["training"]["seed"] = int(seed)
    if "horizon" in experiment:
        config["evaluation"]["cfd_horizon"] = int(experiment["horizon"])
    if "exposure_source" in experiment:
        source = str(experiment["exposure_source"])
        if source != "hard_retrieval":
            raise ValueError("exposure_source must be hard_retrieval")
        config["evaluation"]["exposure_source"] = source
    if sweep_path is not None:
        set_nested(config, sweep_path, sweep_value)
        config["project"]["sweeps"] = [{"path": sweep_path, "value": sweep_value}]
    identifier = config_hash(config, length=16)
    return MatrixJob(identifier, dataset, method, int(seed), config, sweep_path, sweep_value)


def expand_matrix(
    experiment_path: str | Path,
    base_path: str | Path = "configs/base.yaml",
    config_root: str | Path = "configs",
) -> list[MatrixJob]:
    with Path(experiment_path).open("r", encoding="utf-8") as handle:
        experiment = yaml.safe_load(handle)
    base = load_config([base_path])
    root = Path(config_root)
    datasets = list(experiment["datasets"])
    seeds = [int(value) for value in experiment.get("seeds", base["training"]["seeds"])]
    jobs = []
    if "sweeps" in experiment:
        method = str(experiment.get("method", "frag"))
        for dataset in datasets:
            for path, values in experiment["sweeps"].items():
                for value in values:
                    for seed in seeds:
                        jobs.append(
                            _job(
                                base,
                                experiment,
                                root,
                                str(dataset),
                                method,
                                seed,
                                str(path),
                                value,
                            )
                        )
    else:
        methods = [str(value) for value in experiment["methods"]]
        for dataset in datasets:
            for method in methods:
                for seed in seeds:
                    jobs.append(_job(base, experiment, root, str(dataset), method, seed))
    if "sweeps" in experiment:
        return _deduplicate_sweep_jobs(jobs)
    return jobs


def _semantic_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _semantic_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_semantic_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int | float):
        return float(value)
    return value


def _deduplicate_sweep_jobs(jobs: Sequence[MatrixJob]) -> list[MatrixJob]:
    grouped: dict[str, list[MatrixJob]] = {}
    for job in jobs:
        semantic = copy.deepcopy(job.config)
        semantic.get("project", {}).pop("sweeps", None)
        key = json.dumps(_semantic_value(semantic), sort_keys=True, separators=(",", ":"))
        grouped.setdefault(key, []).append(job)
    unique = []
    for matching in grouped.values():
        first = matching[0]
        config = copy.deepcopy(first.config)
        labels = [
            {"path": job.sweep_path, "value": job.sweep_value}
            for job in matching
            if job.sweep_path is not None
        ]
        config["project"]["sweeps"] = labels
        unique.append(
            MatrixJob(
                config_hash(config, length=16),
                first.dataset,
                first.method,
                first.seed,
                config,
                first.sweep_path,
                first.sweep_value,
            )
        )
    return unique


def select_matrix_jobs(
    jobs: Sequence[MatrixJob],
    job_index: int | None = None,
    job_id: str | None = None,
) -> list[MatrixJob]:
    if job_index is not None and job_id is not None:
        raise ValueError("job-index and job-id are mutually exclusive")
    if job_index is not None:
        if job_index < 0 or job_index >= len(jobs):
            raise ValueError(f"job-index must be between 0 and {len(jobs) - 1}")
        return [jobs[job_index]]
    if job_id is not None:
        selected = [job for job in jobs if job.job_id == job_id]
        if not selected:
            raise ValueError(f"unknown job-id: {job_id}")
        if len(selected) > 1:
            raise ValueError(f"ambiguous job-id: {job_id}")
        return selected
    return list(jobs)


def apply_validation_mean(
    jobs: Sequence[MatrixJob],
    value: float,
) -> list[MatrixJob]:
    if len(jobs) != 1:
        raise ValueError("validation-mean requires exactly one selected job")
    mean = float(value)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("validation-mean must be positive and finite")
    job = jobs[0]
    if job.config["retrieval"].get("fixed_k") != "validation_mean":
        raise ValueError("validation-mean requires a fixed-K validation_mean job")
    selected = copy.deepcopy(job)
    selected.config["retrieval"]["validation_mean"] = mean
    return [selected]


def execute_matrix(
    jobs: Sequence[MatrixJob],
    external_root: str | Path | None = None,
    command: str = "full",
) -> list[RunResult]:
    if command not in {"train", "full"}:
        raise ValueError("matrix command must be train or full")
    if command != "full" and any(bool(job.config["model"].get("external", False)) for job in jobs):
        raise ValueError("external matrix jobs require command full")
    results = []
    validation_means: dict[tuple[str, int], float] = {}
    ordered = sorted(
        jobs,
        key=lambda job: (
            job.dataset,
            job.seed,
            job.config["retrieval"].get("fixed_k") == "validation_mean",
            job.method,
            job.job_id,
        ),
    )
    for job in ordered:
        config = copy.deepcopy(job.config)
        if config["retrieval"].get("fixed_k") == "validation_mean":
            retrieval = config["retrieval"]
            configured = (
                retrieval.get("validation_mean") is not None
                or retrieval.get("validation_summary") is not None
            )
            if not configured:
                key = (job.dataset, job.seed)
                if key not in validation_means:
                    raise ValueError(f"missing FRAG validation retrieval size for {key}")
                retrieval["validation_mean"] = validation_means[key]
        external_path = None
        if bool(config["model"].get("external", False)):
            if external_root is None:
                raise ValueError("external_root is required for external matrix jobs")
            external_path = (
                Path(external_root)
                / str(config["project"]["experiment"])
                / job.dataset
                / job.method
                / f"seed={job.seed}.jsonl"
            )
        job_command = "full" if bool(config["model"].get("external", False)) else command
        result = run_experiment(config, job_command, external_path)
        results.append(result)
        if job.method == "frag" and "validation" in result.summaries:
            validation_means[(job.dataset, job.seed)] = float(
                result.summaries["validation"]["mean_hard_retrieval_size"]
            )
    return results


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _sweep_labels(record: dict[str, Any]) -> list[dict[str, Any] | None]:
    labels = record.get("sweeps")
    if labels is None:
        legacy = record.get("sweep")
        return [legacy]
    if not isinstance(labels, list):
        raise TypeError("sweeps must be a list")
    return labels or [None]


def aggregate_artifacts(
    roots: Iterable[str | Path],
    baseline_method: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    summary_paths = []
    for value in roots:
        path = Path(value)
        if path.is_file() and path.name == "summary.json":
            summary_paths.append(path)
        elif path.is_dir():
            summary_paths.extend(path.rglob("summary.json"))
    summaries = []
    per_user = []
    for path in sorted(set(summary_paths)):
        with path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append(summary)
        metric_path = path.with_name("per_user_metrics.jsonl")
        if metric_path.is_file():
            per_user.extend(_read_jsonl(metric_path))
    metric_names = ("recall", "precision", "mrr", "ndcg", "ed", "wger", "gc")
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for summary in summaries:
        for sweep in _sweep_labels(summary):
            sweep_key = json.dumps(sweep, sort_keys=True, separators=(",", ":"))
            key = (summary["dataset"], summary["method"], summary["split"], sweep_key)
            groups.setdefault(key, []).append(summary)
    aggregates = []
    for (dataset, method, split, sweep_key), rows in sorted(groups.items()):
        for metric in metric_names:
            section = "utility" if metric in {"recall", "precision", "mrr", "ndcg"} else "fairness"
            values = [float(row[section][metric]) for row in rows]
            record = {
                "dataset": dataset,
                "method": method,
                "split": split,
                "metric": metric,
                "n_seeds": len(values),
                "mean": sum(values) / len(values),
                "values": values,
                "sweep": json.loads(sweep_key),
            }
            if len(values) >= 2:
                record["confidence_interval"] = mean_confidence_interval(values)
            aggregates.append(record)
    cfd_aggregates = []
    for (dataset, method, split, sweep_key), rows in sorted(groups.items()):
        trajectories = [row.get("cfd_average") for row in rows]
        trajectories = [list(map(float, value)) for value in trajectories if value is not None]
        if not trajectories:
            continue
        horizon = len(trajectories[0])
        if any(len(value) != horizon for value in trajectories):
            raise ValueError("CFD trajectories must share a common horizon")
        for round_index in range(horizon):
            values = [value[round_index] for value in trajectories]
            record = {
                "dataset": dataset,
                "method": method,
                "split": split,
                "round": round_index + 1,
                "n_seeds": len(values),
                "mean": sum(values) / len(values),
                "values": values,
                "sweep": json.loads(sweep_key),
            }
            if len(values) >= 2:
                record["confidence_interval"] = mean_confidence_interval(values)
            cfd_aggregates.append(record)
    paired = []
    if baseline_method is not None:
        indexed: dict[tuple[str, str, str, str, str, str], list[tuple[int, float]]] = {}
        for row in per_user:
            for sweep in _sweep_labels(row):
                sweep_key = json.dumps(sweep, sort_keys=True, separators=(",", ":"))
                for metric in metric_names:
                    key = (
                        str(row["dataset"]),
                        str(row["method"]),
                        str(row["split"]),
                        sweep_key,
                        metric,
                        str(row["user_id"]),
                    )
                    indexed.setdefault(key, []).append((int(row["seed"]), float(row[metric])))
        combinations = sorted(
            {
                (
                    str(row["dataset"]),
                    str(row["method"]),
                    str(row["split"]),
                    json.dumps(sweep, sort_keys=True, separators=(",", ":")),
                )
                for row in per_user
                for sweep in _sweep_labels(row)
                if str(row["method"]) != baseline_method
            }
        )
        for dataset, method, split, sweep_key in combinations:
            for metric in metric_names:
                first = {
                    key[5]: sum(value for _, value in values) / len(values)
                    for key, values in indexed.items()
                    if key[:5] == (dataset, method, split, sweep_key, metric)
                }
                second = {
                    key[5]: sum(value for _, value in values) / len(values)
                    for key, values in indexed.items()
                    if key[:5] == (dataset, baseline_method, split, sweep_key, metric)
                }
                common = sorted(set(first) & set(second))
                if len(common) < 2:
                    continue
                test = paired_t_test_by_user(
                    {user: first[user] for user in common},
                    {user: second[user] for user in common},
                )
                paired.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "baseline_method": baseline_method,
                        "split": split,
                        "metric": metric,
                        "sweep": json.loads(sweep_key),
                        "seed_aggregation": "per_user_mean",
                        **test,
                    }
                )
    return {
        "aggregates": aggregates,
        "cfd_aggregates": cfd_aggregates,
        "paired_tests": paired,
    }
