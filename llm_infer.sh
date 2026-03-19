#!/bin/bash
#SBATCH --job-name=got_ocr_infer
#SBATCH --account=csds312
#SBATCH --partition=markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --constraint=gpu2h100
#SBATCH --output=logs/infer_%A_%a.out
#SBATCH --error=logs/infer_%A_%a.err
#SBATCH --array=1-5

cd /mnt/vstor/courses/csds312/cxv166/OCR
set -a
source .env
set +a
mkdir -p logs
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

SOURCE_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" sources.txt)
echo "Processing: $SOURCE_DIR"

unset VIRTUAL_ENV
SOURCE_DIR="$SOURCE_DIR" uv run llm_inference.py

echo "End: $(date)"