from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import torch


def status(name: str, passed: bool, detail: str) -> bool:
    marker = "ok" if passed else "missing"
    print(f"{marker}: {name}: {detail}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "production"), default="smoke")
    args = parser.parse_args()
    production = args.mode == "production"
    passed = True
    passed &= status("python", sys.version_info >= (3, 10), sys.version.split()[0])
    for package in ["numpy", "pandas", "pyarrow", "yaml", "scipy", "torch"]:
        found = importlib.util.find_spec(package) is not None
        passed &= status(package, found, "available" if found else "not installed")
    for package in ["transformers", "peft", "accelerate", "huggingface_hub"]:
        found = importlib.util.find_spec(package) is not None
        available = status(package, found, "available" if found else "needed for LLaMA runs")
        if production:
            passed &= available
    cuda = torch.cuda.is_available()
    cuda_ready = status("cuda", cuda, torch.version.cuda or "not available")
    if production:
        passed &= cuda_ready
    if cuda:
        suitable_gpu = False
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory = props.total_memory / 1024**3
            suitable = status(
                f"gpu_{index}",
                memory >= 40,
                f"{props.name}, {memory:.1f} GiB",
            )
            suitable_gpu |= suitable
        if production:
            passed &= suitable_gpu
    space = shutil.disk_usage(Path.cwd())
    free = space.free / 1024**3
    minimum_free = 25 if production else 1
    passed &= status(
        "workspace_storage",
        free >= minimum_free,
        f"{free:.1f} GiB free",
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
