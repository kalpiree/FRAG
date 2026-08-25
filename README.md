# FRAG

FRAG is a fairness-aware retrieval-augmented framework for LLM-based sequential recommendation.

## What Is In This Repository

- `src/frag/`: FRAG models, retrieval, training, and evaluation
- `scripts/`: runnable entry points for setup, experiments, and aggregation
- `configs/datasets/`: MovieLens, LastFM, Steam, and GoodReads settings
- `configs/methods/`: method configurations
- `configs/ablations/`: ablation settings
- `configs/experiments/`: RQ1-RQ4 experiment matrices
- `tests/`: unit and end-to-end tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Download the Model

```bash
huggingface-cli login
python scripts/download_model.py --config configs/base.yaml
```

## Prepare the Data

Download the datasets:

```bash
python scripts/download_data.py --dataset movielens
python scripts/download_data.py --dataset lastfm
python scripts/download_data.py --dataset steam
python scripts/download_data.py --dataset goodreads --variant spoiler
```

Prepare chronological train, validation, and test splits:

```bash
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/movielens.yaml
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/lastfm.yaml
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/steam.yaml
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/goodreads.yaml
```

## Check the Environment

```bash
python scripts/check_environment.py --mode production
```

## Run a Small End-to-End Example

```bash
python scripts/run_smoke.py
```

## Train and Evaluate FRAG

```bash
python scripts/run.py full \
  --config configs/base.yaml \
  --config configs/datasets/movielens.yaml \
  --config configs/methods/frag.yaml
```

## Run the Main Experiments

```bash
python scripts/run_matrix.py \
  --experiment configs/experiments/rq2.yaml \
  --job-index 0 \
  --execute
```

Create a complete job manifest:

```bash
python scripts/run_matrix.py \
  --experiment configs/experiments/rq1.yaml \
  --manifest outputs/rq1_jobs.jsonl
```

## Included Experiment Families

- `configs/experiments/rq1.yaml`: main comparison
- `configs/experiments/rq2.yaml`: parameter sensitivity
- `configs/experiments/rq3.yaml`: ablation study
- `configs/experiments/rq4.yaml`: cumulative retrieval fairness

## Reproduction Guide

[docs/reproduce.md](docs/reproduce.md)
