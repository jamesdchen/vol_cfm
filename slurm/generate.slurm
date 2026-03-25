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
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

source /home1/jc_905/.bashrc
conda activate project-cucuringu

mkdir -p logs

python scripts/generate.py --checkpoint ${CHECKPOINT:?"CHECKPOINT env var required"}
