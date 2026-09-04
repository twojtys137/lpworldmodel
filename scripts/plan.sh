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
# Env-var overrides: SEED, GOAL_H, CKPT_BASE, PLAN_OUTPUT_DIR, WANDB_ENTITY,
# WANDB_PROJECT, WANDB_RUN_NAME, EVALUATION_MODE (plan|gt_replay),
# OBJECTIVE_ALPHA, GOAL_SOURCE, GOAL_FILE_PATH, CEM_EVAL_EVERY,
# CEM_OPT_STEPS, and CEM_NUM_SAMPLES.
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
[ -n "${EVALUATION_MODE:-}" ] && EXTRA+=("evaluation_mode=${EVALUATION_MODE}")
[ -n "${OBJECTIVE_ALPHA:-}" ] && EXTRA+=("objective.alpha=${OBJECTIVE_ALPHA}")
[ -n "${OBJECTIVE_BASIS:-}" ] && EXTRA+=(
  "+objective.basis_path=${OBJECTIVE_BASIS}"
  "+objective.basis_key=${OBJECTIVE_BASIS_KEY:-goal_residual}"
  "+objective.rank=${OBJECTIVE_RANK:?set OBJECTIVE_RANK with OBJECTIVE_BASIS}"
)
[ -n "${GOAL_SOURCE:-}" ] && EXTRA+=("goal_source=${GOAL_SOURCE}")
[ -n "${GOAL_FILE_PATH:-}" ] && EXTRA+=("goal_file_path=${GOAL_FILE_PATH}")
[ -n "${EXCLUDED_TRAJ_IDS:-}" ] && EXTRA+=("+excluded_traj_ids=${EXCLUDED_TRAJ_IDS}")
[ -n "${CEM_EVAL_EVERY:-}" ] && EXTRA+=("planner.sub_planner.eval_every=${CEM_EVAL_EVERY}")
[ -n "${CEM_OPT_STEPS:-}" ] && EXTRA+=("planner.sub_planner.opt_steps=${CEM_OPT_STEPS}")
[ -n "${CEM_NUM_SAMPLES:-}" ] && EXTRA+=("planner.sub_planner.num_samples=${CEM_NUM_SAMPLES}")
[ -n "${CEM_BATCH_SIZE:-}" ] && EXTRA+=("+planner.sub_planner.candidate_batch_size=${CEM_BATCH_SIZE}")
[ -n "${CEM_CACHE_ENCODING:-}" ] && EXTRA+=("+planner.sub_planner.cache_initial_encoding=${CEM_CACHE_ENCODING}")
[ -n "${PLAN_OUTPUT_DIR:-}" ] && EXTRA+=(
  "hydra.run.dir=${PLAN_OUTPUT_DIR}"
  "hydra.job.chdir=true"
)
[ -n "${WANDB_ENTITY:-}" ] && EXTRA+=("+wandb_entity=${WANDB_ENTITY}")
[ -n "${WANDB_PROJECT:-}" ] && EXTRA+=("+wandb_project=${WANDB_PROJECT}")
EXTRA+=("+wandb_run_name=${WANDB_RUN_NAME:-plan_${MODEL_NAME}_seed${SEED:-99}}")

cd "${REPO}"
python plan.py --config-name "${CONFIG}" \
    ckpt_base_path="${CKPT_BASE}" model_name="${MODEL_NAME}" model_epoch="${EPOCH}" \
    n_evals="${NEVALS}" planner.max_iter="${MAXITER}" "${EXTRA[@]}"
