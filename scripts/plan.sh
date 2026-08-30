#!/bin/bash
# Planning eval (CEM + receding-horizon MPC) of a trained checkpoint.
#
# Cluster-agnostic: this just runs `python plan.py ...`. Prerequisites are the same
# as scripts/train.sh (lpwm env active, DATASET_DIR set).
#
# Usage:
#   scripts/plan.sh <plan_config> <model_name> <epoch> [n_evals] [max_iter]
#     <plan_config> : plan_lewm.yaml   (from-scratch JEPA / LpWM & LeWM; env-agnostic, loads from ckpt)
#                     plan_pusht.yaml / plan_wall.yaml   (DINO-WM concat baselines)
#     <model_name>  : run-dir name under $CKPT_BASE/outputs/  (matches train.sh's RUN_NAME)
#     <epoch>       : latest | <int>
# Env-var overrides: SEED (plan.py seed=), GOAL_H (planning horizon), CKPT_BASE
# (default ./runs), PLAN_OUTPUT_DIR (persistent Hydra output), WANDB_ENTITY and
# WANDB_PROJECT.
set -euo pipefail
CONFIG=${1:?usage: plan.sh <plan_config> <model_name> <epoch> [n_evals] [max_iter]}
MODEL_NAME=${2:?need model_name}; EPOCH=${3:?need epoch}
NEVALS=${4:-50}; MAXITER=${5:-10}

REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${DATASET_DIR:?set DATASET_DIR to the dataset root (contains pusht_noise/ and wall_single/)}"
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}

export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}   # headless pygame rendering (PushT)

EXTRA=()
[ -n "${SEED:-}" ] && EXTRA+=("seed=${SEED}")
[ -n "${GOAL_H:-}" ] && EXTRA+=("goal_H=${GOAL_H}")
[ -n "${PLAN_OUTPUT_DIR:-}" ] && EXTRA+=(
  "hydra.run.dir=${PLAN_OUTPUT_DIR}"
  "hydra.job.chdir=true"
)
[ -n "${WANDB_ENTITY:-}" ] && EXTRA+=("+wandb_entity=${WANDB_ENTITY}")
[ -n "${WANDB_PROJECT:-}" ] && EXTRA+=("+wandb_project=${WANDB_PROJECT}")
EXTRA+=("+wandb_run_name=plan_${MODEL_NAME}_seed${SEED:-99}")

cd "${REPO}"
python plan.py --config-name "${CONFIG}" \
    ckpt_base_path="${CKPT_BASE}" model_name="${MODEL_NAME}" model_epoch="${EPOCH}" \
    n_evals="${NEVALS}" planner.max_iter="${MAXITER}" "${EXTRA[@]}"
