#!/usr/bin/env bash
# One flow-map FFM run of the paper protocol: 55M environment steps, 21
# evaluations of the deterministic central action. DMC tasks run at discount
# 0.99, the humanoid joystick tasks at 0.97 (scripts/tasks.sh).
#
# Usage: bash scripts/run_flowmap.sh <map_steps> <task> <seed> [hydra overrides...]
#   map_steps 2: the two-step sampler, 1: the one-step sampler
#
# Output: outputs/flowmap_k<map_steps>/<task>__s<seed>/ (metrics.npz, the hydra
# config and log; set OUT_ROOT to replace outputs/). A run whose metrics.npz
# exists is skipped, so a job list can simply be relaunched.
#   RESUME_EVERY=N  checkpoint every N of the 21 evaluation chunks and resume an
#                   interrupted run from its directory (preemptible clusters)
#   WANDB_MODE      wandb mode (default online; offline / disabled)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/tasks.sh
K=${1:?map_steps (1 or 2)}; TASK=${2:?task}; SEED=${3:?seed}; shift 3
[[ "$K" =~ ^[1-9][0-9]*$ ]] || { echo "map_steps must be a positive integer (got $K)" >&2; exit 1; }
ARGS=$(task_args "$TASK")

OUT=${OUT_ROOT:-outputs}/flowmap_k${K}/${TASK}__s${SEED}
if [ -f "$OUT/metrics.npz" ]; then echo "already done: $OUT"; exit 0; fi

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python src/jaxrl/flowmap_ffm.py env.name="$TASK" seed="$SEED" $ARGS \
  hyperparameters.diffusion.flowmap.map_steps="$K" \
  resume_every="${RESUME_EVERY:-0}" wandb.mode="${WANDB_MODE:-online}" \
  hydra.run.dir="$OUT" "$@"

# the trainer logs exceptions and exits normally: success is the metrics file
test -f "$OUT/metrics.npz" || { echo "FAILED: no $OUT/metrics.npz" >&2; exit 1; }
