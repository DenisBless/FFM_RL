#!/usr/bin/env bash
# Behaviour-cloning pretraining of the reference flow policy, 300K steps.
# The BC policy is task-independent within a domain (the datasets share
# observations and actions across tasks), so it is trained once on task 1 and
# reused by every task of the domain.
#
# Usage: scripts/run_bc.sh <domain> <seed>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/domains.sh
DOMAIN=${1:?domain}; SEED=${2:?seed}; domain_check "$DOMAIN"

ENV=${ENV_PREFIX[$DOMAIN]}-singletask-task1-v0
CHUNK=""; [ "${FAMILY[$DOMAIN]}" = manip ] && CHUNK="--agent.action_chunking=True"
WIDTH=$(echo "${EXTRA[$DOMAIN]}" | grep -o -- "--agent.actor_hidden_dims=[^ ]* --agent.value_hidden_dims=[^ ]* --agent.actor_layer_norm=True" || true)
DD=$(echo "${EXTRA[$DOMAIN]}" | grep -o -- "--ogbench_dataset_dir=[^ ]*" || true)

eval python main.py --env_name "$ENV" --agent agents/ffm.py --bc_only=True $CHUNK $WIDTH $DD \
  --offline_steps 300000 --online_steps 0 \
  --eval_interval 0 --log_interval 5000 --save_interval 300000 --save_last_checkpoint \
  --seed "$SEED" --run_group bc --tags bc
