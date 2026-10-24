#!/usr/bin/env bash
# The online benchmark suite and its per-family settings. Source this file.
#
#   MuJoCo Playground DMC (20 tasks)   env=mjx_dmc (config default), discount 0.99
#   MJX humanoid joystick (4 tasks)    env=mjx_humanoid, discount 0.97
#
# Everything else is the trainer's config default.

DMC_TASKS=(
  PendulumSwingup AcrobotSwingup AcrobotSwingupSparse BallInCup
  CartpoleBalance CartpoleBalanceSparse CartpoleSwingup CartpoleSwingupSparse
  CheetahRun FingerSpin FingerTurnEasy FingerTurnHard FishSwim HopperHop
  HopperStand ReacherEasy ReacherHard WalkerRun WalkerWalk WalkerStand
)
HUMANOID_TASKS=(
  G1JoystickFlatTerrain G1JoystickRoughTerrain
  T1JoystickFlatTerrain T1JoystickRoughTerrain
)
HUMANOID_ARGS="env=mjx_humanoid hyperparameters.gamma=0.97"

# task_args <task>: the hydra overrides of the task's family (fails on an
# unknown task name)
task_args() {
  local t
  for t in "${DMC_TASKS[@]}"; do
    [ "$t" = "$1" ] && return 0
  done
  for t in "${HUMANOID_TASKS[@]}"; do
    [ "$t" = "$1" ] && { echo "$HUMANOID_ARGS"; return 0; }
  done
  echo "unknown task: $1 (see scripts/tasks.sh)" >&2
  return 1
}
