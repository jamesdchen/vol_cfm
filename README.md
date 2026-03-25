# vol-cfm

Conditional Flow Matching for generating synthetic intraday realized volatility paths.

## Overview

This project trains a neural ODE-based generative model (OT-CFM) that learns to produce realistic 48-period intraday RV decompositions, conditioned on recent daily realized volatility. It downscales a single daily RV forecast into a full intraday volatility curve.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Data

The model reads 30-minute RV bars from parquet files in `data/all30min/`. These are preprocessed into daily aggregates and intraday proportion targets.

## Usage

### Train

```bash
python scripts/train.py
python scripts/train.py --epochs 100 --batch-size 512 --lr 5e-4
```

### Generate synthetic paths

```bash
python scripts/generate.py --checkpoint checkpoints/best.pt
python scripts/generate.py --checkpoint checkpoints/best.pt --split test --num-samples-per-day 5
```

### Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt
python scripts/evaluate.py --checkpoint checkpoints/best.pt --output-dir eval_results --num-samples 2000
```

### Export dataset

```bash
python scripts/export_dataset.py --harxhar-path data/all30min
```

All scripts support `--verbose` and `--quiet` flags for log verbosity control.

## SLURM

```bash
sbatch slurm/train.slurm
sbatch slurm/generate.slurm
```

## Architecture

```
cfm/
├── config.py          # CFMConfig dataclass (all hyperparameters)
├── logging.py         # Structured logging setup
├── export.py          # Export generated RV to harxhar-compatible parquet
├── data/
│   ├── loading.py     # Parquet loading, daily aggregation, CFM pair construction
│   ├── transforms.py  # Scaling and proportion normalization
│   └── dataset.py     # PyTorch Dataset and DataLoader builders
├── model/
│   ├── vector_field.py    # ConditionalVectorField MLP with time embedding
│   ├── flow_matching.py   # OT-CFM loss (Hungarian algorithm pairing)
│   └── sampler.py         # ODE-based sampling (dopri5 or Euler fallback)
├── training/
│   ├── trainer.py     # Training loop with checkpointing
│   └── scheduler.py   # Cosine warmup LR scheduler
└── evaluation/
    ├── metrics.py     # Daily consistency, diurnal pattern, energy distance, KS, ACF
    └── visualize.py   # Sample paths, diurnal patterns, marginals, ACF plots
```

## Testing

```bash
python -m pytest tests/ -v
```
