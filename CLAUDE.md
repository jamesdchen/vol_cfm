# vol_cfm — Conditional Flow Matching for Intraday RV

## Architecture
```
cfm/
├── data/          Data loading, dataset, masking, transforms
├── model/         ConditionalVectorField, OT-CFM loss, ODE sampler
├── training/      CFMTrainer, CosineWarmupScheduler
├── evaluation/    Metrics + visualization
├── cli.py         Shared CLI utilities
└── config.py      CFMConfig dataclass
```

## Pipeline (3-stage, managed by claude-hpc)

| Stage | Type | Script | Depends on |
|-------|------|--------|------------|
| train | single | `python scripts/train.py` | — |
| generate | array (10 chunks) | `python scripts/generate.py` | train |
| evaluate | single | `python scripts/evaluate.py` | generate |

Stages are defined in `project.yaml` and submitted via claude-hpc `/submit`.

## Scripts

```bash
python scripts/train.py --help        # Train CFM model
python scripts/generate.py --help     # Generate synthetic RV paths
python scripts/evaluate.py --help     # Evaluate with metrics + plots
python scripts/export_dataset.py      # Export dataset utility
```

## HPC Configuration

- **Cluster:** Discovery (USC), SLURM scheduler
- **Account:** pollok_1603
- **Username:** jc_905
- **Remote path:** /home1/jc_905/vol_cfm
- **Conda env:** project-cucuringu
- **GPU constraint:** a100|a40|v100|l40s

## Data Layout

| Directory | Contents |
|-----------|----------|
| `data/all30min/` | Source parquet files (vwstock_stats, ewstock_stats, core_stats) |
| `checkpoints/` | Model checkpoints (best.pt, epoch_N.pt, final.pt) |
| `samples/` | Generated RV paths (seed_N/chunk_*.pt) |
| `eval_results/` | Metrics JSON + visualization PNGs |

## Development

```bash
ruff check .
ruff format .
mypy cfm/ scripts/ --ignore-missing-imports
pytest tests/ -m "not slow and not gpu" --tb=short
```
