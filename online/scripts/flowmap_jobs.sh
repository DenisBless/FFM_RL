#!/usr/bin/env bash
# The job list of the flow-map experiments: one scripts/run_flowmap.sh command
# per line for every task of the suite (20 DMC + 4 humanoid) and seed.
#
# Usage: bash scripts/flowmap_jobs.sh <map_steps> [first_seed [last_seed]]   (seeds 0-9)
#
#   bash scripts/flowmap_jobs.sh 2 > jobs_k2.txt     # two-step sampler, 240 runs
#   bash scripts/flowmap_jobs.sh 1 > jobs_k1.txt     # one-step sampler, 240 runs
#
# Run a list from this folder with the environment active, e.g.
#   bash jobs_k2.txt                                                  # one GPU, in turn
#   parallel -j 4 'CUDA_VISIBLE_DEVICES=$(({%} - 1)) {}' < jobs_k2.txt  # 4 GPUs (GNU parallel)
#   sbatch --array=1-240 scripts/slurm_array.sh jobs_k2.txt          # SLURM
# The humanoid lines checkpoint every 2 evaluation chunks (RESUME_EVERY=2):
# they take 1.5-3.5 h on a V100, and a relaunched line resumes where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/tasks.sh
K=${1:?map_steps (1 or 2)}; FIRST=${2:-0}; LAST=${3:-9}

for seed in $(seq "$FIRST" "$LAST"); do
  for task in "${DMC_TASKS[@]}"; do
    echo "bash scripts/run_flowmap.sh $K $task $seed"
  done
  for task in "${HUMANOID_TASKS[@]}"; do
    echo "RESUME_EVERY=2 bash scripts/run_flowmap.sh $K $task $seed"
  done
done
