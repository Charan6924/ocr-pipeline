#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --account=csds312
#SBATCH --partition=markov_cpu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=8
#SBATCH --time=13-08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
cd /mnt/vstor/courses/csds312/cxv166/OCR
mkdir -p logs
uv run preprocess.py

echo "End time: $(date)"