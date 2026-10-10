#!/usr/bin/env bash
# FFM offline-to-online: 1M offline steps on the dataset, then 500K online
# environment steps, evaluating every 50K steps over 50 episodes. Initialises
# the flow policy from the domain's BC checkpoint (scripts/run_bc.sh).
#
# Usage: scripts/run_ffm.sh <domain> <task 1-5> <seed>
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/domains.sh
DOMAIN=${1:?domain}; TASK=${2:?task 1-5}; SEED=${3:?seed}; domain_check "$DOMAIN"

ENV=${ENV_PREFIX[$DOMAIN]}-singletask-task${TASK}-v0
BCENV=${ENV_PREFIX[$DOMAIN]}-singletask-task1-v0
BC=$(ls exp/ffm/bc/${BCENV}/bc_ffm_*_s${SEED}_*/params_300000.pkl 2>/dev/null | head -1)
if [ -z "$BC" ]; then
  echo "no BC checkpoint for ${DOMAIN} seed ${SEED}; run: scripts/run_bc.sh ${DOMAIN} ${SEED}" >&2; exit 1
fi

eval python main.py --env_name "$ENV" --agent agents/ffm.py ${REGIME[$DOMAIN]} ${EXTRA[$DOMAIN]} \
  --kl_budget_online=8.0 \
  --offline_steps 1000000 --online_steps 500000 \
  --eval_interval 50000 --eval_episodes 50 --log_interval 5000 \
  --save_interval 250000 --save_last_checkpoint \
  --seed "$SEED" --run_group ffm --tags ffm \
  --pretrained_actor_path "$BC"
