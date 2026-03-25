#!/bin/bash
#SBATCH --job-name=cfm_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|a40|v100|l40s"
#SBATCH --partition=gpu
#SBATCH --account=pollok_1603
#SBATCH --time=0:30:00
#SBATCH --chdir=/home1/jc_905/vol-cfm
#SBATCH --output=logs/generate_%A_%a.out
#SBATCH --error=logs/generate_%A_%a.err

source /home1/jc_905/.bashrc
conda activate project-cucuringu

mkdir -p logs

# Seed = base seed + array task ID (defaults to task 0 if not an array job)
SEED=$(( ${BASE_SEED:-42} + ${SLURM_ARRAY_TASK_ID:-0} ))

python scripts/generate.py \
    --checkpoint ${CHECKPOINT:?"CHECKPOINT env var required"} \
    --output-dir ${OUTPUT_DIR:-samples} \
    --num-samples-per-day ${NUM_SAMPLES_PER_DAY:-1} \
    --split ${SPLIT:-test} \
    --seed ${SEED}
