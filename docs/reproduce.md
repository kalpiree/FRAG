# Reproducing the Experiments

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Model

```bash
huggingface-cli login
python scripts/download_model.py --config configs/base.yaml
```

The default configuration uses `meta-llama/Meta-Llama-3-8B`.

## 3. Data

```bash
python scripts/download_data.py --dataset movielens
python scripts/download_data.py --dataset lastfm
python scripts/download_data.py --dataset steam
python scripts/download_data.py --dataset goodreads --variant spoiler
```

```bash
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/movielens.yaml
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/lastfm.yaml
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/steam.yaml
python scripts/prepare_data.py --config configs/base.yaml --config configs/datasets/goodreads.yaml
```

## 4. Check the Setup

```bash
python scripts/check_environment.py --mode production
```

## 5. Small End-to-End Run

```bash
python scripts/run_smoke.py
```

## 6. Train and Evaluate FRAG

MovieLens:

```bash
python scripts/run.py train \
  --config configs/base.yaml \
  --config configs/datasets/movielens.yaml \
  --config configs/methods/frag.yaml

python scripts/run.py evaluate \
  --config configs/base.yaml \
  --config configs/datasets/movielens.yaml \
  --config configs/methods/frag.yaml
```

Replace the dataset configuration with one of:

- `configs/datasets/movielens.yaml`
- `configs/datasets/lastfm.yaml`
- `configs/datasets/steam.yaml`
- `configs/datasets/goodreads.yaml`

## 7. Experiment Matrices

Create a job manifest:

```bash
python scripts/run_matrix.py \
  --experiment configs/experiments/rq1.yaml \
  --manifest outputs/rq1_jobs.jsonl
```

Run one job:

```bash
python scripts/run_matrix.py \
  --experiment configs/experiments/rq1.yaml \
  --job-index 0 \
  --execute
```

Run one parameter-sensitivity job:

```bash
python scripts/run_matrix.py \
  --experiment configs/experiments/rq2.yaml \
  --job-index 0 \
  --execute
```

Available matrices:

- `configs/experiments/rq1.yaml`
- `configs/experiments/rq2.yaml`
- `configs/experiments/rq3.yaml`
- `configs/experiments/rq4.yaml`

## 8. Aggregate Runs

```bash
python scripts/aggregate.py outputs --output outputs/aggregate.json
```
