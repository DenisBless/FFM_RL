#!/usr/bin/env bash
#SBATCH --job-name=ffm-flowmap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
# SLURM array over a job list of scripts/flowmap_jobs.sh: array task i runs
# line i. Submit from this folder with the environment active (the job
# inherits it), adding the partition / account flags of your cluster:
#
#   mkdir -p logs
#   bash scripts/flowmap_jobs.sh 2 > jobs_k2.txt
#   sbatch --array=1-$(wc -l < jobs_k2.txt) scripts/slurm_array.sh jobs_k2.txt
#
# DMC runs take 15-35 min and the humanoid runs 1.5-3.5 h on a V100; every run
# fits on a 16 GB V100. Resubmitting the same list skips finished runs.
set -euo pipefail
LIST=${1:?job list (scripts/flowmap_jobs.sh)}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
[ -n "$LINE" ] || { echo "no line ${SLURM_ARRAY_TASK_ID} in $LIST" >&2; exit 1; }
echo "$LINE"
eval "$LINE"
