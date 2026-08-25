from __future__ import annotations

import argparse
import json
from pathlib import Path

from frag.runner import apply_validation_mean, execute_matrix, expand_matrix, select_matrix_jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Expand or execute a reproducible experiment matrix."
    )
    parser.add_argument("--experiment", required=True, help="Experiment matrix YAML file.")
    parser.add_argument("--base", default="configs/base.yaml", help="Base configuration YAML.")
    parser.add_argument("--config-root", default="configs", help="Configuration directory.")
    parser.add_argument("--manifest", help="Write selected job records as JSONL.")
    parser.add_argument("--external-root", help="Root containing imported baseline predictions.")
    parser.add_argument("--execute", action="store_true", help="Execute selected jobs.")
    parser.add_argument("--limit", type=int, help="Keep at most this many selected jobs.")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--job-index",
        type=int,
        help="Select one job by zero-based index in the expanded matrix.",
    )
    selector.add_argument(
        "--job-id",
        help="Select one job by the stable identifier printed in the manifest.",
    )
    parser.add_argument(
        "--command",
        choices=("train", "full"),
        default="full",
        help="Use train for checkpoint and validation only, or full to include test evaluation.",
    )
    parser.add_argument(
        "--validation-mean",
        type=float,
        help="Supply FRAG's prior validation mean for one selected fixed-K job.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _write(path: str | None, rows: list[dict]) -> None:
    if path is None:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    jobs = expand_matrix(args.experiment, args.base, args.config_root)
    try:
        jobs = select_matrix_jobs(jobs, args.job_index, args.job_id)
    except ValueError as error:
        parser.error(str(error))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("limit must be positive")
        jobs = jobs[: args.limit]
    if args.validation_mean is not None:
        if args.job_index is None and args.job_id is None:
            parser.error("validation-mean requires job-index or job-id")
        try:
            jobs = apply_validation_mean(jobs, args.validation_mean)
        except ValueError as error:
            parser.error(str(error))
    records = [job.record() for job in jobs]
    _write(args.manifest, records)
    if args.execute:
        results = execute_matrix(jobs, args.external_root, args.command)
        for result in results:
            print(json.dumps(result.as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
