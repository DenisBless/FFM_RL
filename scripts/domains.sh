#!/usr/bin/env bash
# Per-domain recipe used for every number in the paper. Source this file.
#
# Two regimes, selected by domain family:
#   Regime A (locomotion)    tau_q=30, fixed Fisher normaliser (i_fisher=1),
#                            KL trust region on (lam_scale=1, eps_KL=1 offline, 8 online),
#                            no action chunking
#   Regime B (manipulation)  tau_q=1, adaptive Fisher normaliser, trust region off
#                            offline (lam_scale=0), action chunking with horizon 5
# Everything else is the agent default in agents/ffm.py.
#
# Domain-specific extras: humanoidmaze uses discount 0.999 / rho 0; the large
# datasets (antmaze-giant, cube-triple, puzzle-4x4: 10M; cube-quadruple: 100M)
# use 1024-wide networks with actor layer norm and are read from a directory of
# .npz shards (--ogbench_dataset_dir).

DATA=${OGBENCH_DATA:-$HOME/.ogbench/data}

W1024="--agent.actor_hidden_dims='(1024,1024,1024,1024)' --agent.value_hidden_dims='(1024,1024,1024,1024)' --agent.actor_layer_norm=True"
REGIME_A="--agent.tau_q=30"
REGIME_B="--agent.action_chunking=True --agent.tau_q=1 --agent.adaptive_fisher=True --agent.lam_scale=0"

declare -A ENV_PREFIX FAMILY REGIME EXTRA
ENV_PREFIX[antmaze-large]=antmaze-large-navigate;        FAMILY[antmaze-large]=loco;  EXTRA[antmaze-large]=""
ENV_PREFIX[antmaze-giant]=antmaze-giant-navigate;        FAMILY[antmaze-giant]=loco;  EXTRA[antmaze-giant]="$W1024 --ogbench_dataset_dir=$DATA/antmaze-giant-navigate-10m-v0"
ENV_PREFIX[humanoidmaze-medium]=humanoidmaze-medium-navigate; FAMILY[humanoidmaze-medium]=loco; EXTRA[humanoidmaze-medium]="--agent.discount=0.999 --agent.rho=0.0"
ENV_PREFIX[humanoidmaze-large]=humanoidmaze-large-navigate;   FAMILY[humanoidmaze-large]=loco;  EXTRA[humanoidmaze-large]="--agent.discount=0.999 --agent.rho=0.0"
ENV_PREFIX[cube-double]=cube-double-play;                FAMILY[cube-double]=manip;   EXTRA[cube-double]=""
ENV_PREFIX[cube-triple]=cube-triple-play;                FAMILY[cube-triple]=manip;   EXTRA[cube-triple]="$W1024 --ogbench_dataset_dir=$DATA/cube-triple-play-10m-v0"
ENV_PREFIX[cube-quadruple]=cube-quadruple-play;          FAMILY[cube-quadruple]=manip; EXTRA[cube-quadruple]="$W1024 --ogbench_dataset_dir=$DATA/cube-quadruple-play-100m-v0"
ENV_PREFIX[scene]=scene-play;                            FAMILY[scene]=manip;         EXTRA[scene]=""
ENV_PREFIX[puzzle-3x3]=puzzle-3x3-play;                  FAMILY[puzzle-3x3]=manip;    EXTRA[puzzle-3x3]=""
ENV_PREFIX[puzzle-4x4]=puzzle-4x4-play;                  FAMILY[puzzle-4x4]=manip;    EXTRA[puzzle-4x4]="$W1024 --ogbench_dataset_dir=$DATA/puzzle-4x4-play-10m-v0"
for d in "${!FAMILY[@]}"; do
  if [ "${FAMILY[$d]}" = loco ]; then REGIME[$d]=$REGIME_A; else REGIME[$d]=$REGIME_B; fi
done

DOMAINS="antmaze-large antmaze-giant humanoidmaze-medium humanoidmaze-large cube-double cube-triple cube-quadruple scene puzzle-3x3 puzzle-4x4"
SEEDS_PAPER="10001 20002 30003 40004 50005 60006 70007 80008"

domain_check() {
  if [ -z "${ENV_PREFIX[$1]:-}" ]; then
    echo "unknown domain '$1'. Choose one of: $DOMAINS" >&2; exit 1
  fi
}
