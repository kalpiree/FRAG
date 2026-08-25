from __future__ import annotations

import argparse

from frag.baselines import load_baseline_records, save_baseline_contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    args = parser.parse_args()
    records = load_baseline_records(args.source)
    save_baseline_contract(args.destination, records)
    print(f"validated_records={len(records)}")


if __name__ == "__main__":
    main()
