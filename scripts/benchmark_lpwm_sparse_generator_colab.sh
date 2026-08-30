#!/bin/bash
# Fair LpWM vs sparse-generator benchmark for Colab/MyDrive.
#
# Dry-run is the default.  The script provides three persistent stages:
#   STAGE=train    train deterministic run names and register them in a manifest
#   STAGE=plan     run paired PushT MPC+CEM evaluation on the same goals
#   STAGE=collect  join local/W&B/planning metrics into JSON, CSV and Markdown
#   STAGE=all      run all three stages sequentially
#
# Profiles:
#   smoke   8 rollouts, 1 epoch, cheap shape/IO validation
#   screen  50 rollouts, 2 epochs, 256 RDMReg projections
#   full    full PushT dataset, 2 epochs, 8192 projections; paper LpWM batch=64
#
# Model IDs:
#   dense_dense   dense signed patch state + dense generator
#   dense_sparse  dense signed patch state + sparse generator (our hypothesis)
#   sparse_dense  LpWM-style sparse patch state + dense generator
#   sparse_sparse LpWM-style sparse patch state + sparse generator
#   ltv / sparse_ltv
#   lewm / lpwm   literal CLS+D384+Deep-AdaLN paper controls
#
# Typical Colab sequence:
#   bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#   RUN=1 PROFILE=screen STAGE=train SEEDS=0 bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#   RUN=1 PROFILE=screen STAGE=train SEEDS="0 1 2" bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#   RUN=1 PROFILE=screen STAGE=plan  SEEDS="0 1 2" bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#
# Final confirmatory grid (expensive; dry-run it first):
#   MODELS="lpwm lewm dense_dense dense_sparse sparse_dense sparse_sparse ltv sparse_ltv"
#   PROFILE=full STAGE=train MODELS="$MODELS" SEEDS="0 1 2" bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#   RUN=1 PROFILE=full STAGE=train MODELS="$MODELS" SEEDS="0 1 2" bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#   RUN=1 PROFILE=full STAGE=plan  MODELS="$MODELS" SEEDS="0 1 2" bash scripts/benchmark_lpwm_sparse_generator_colab.sh
#
# The exact LpWM/LeWM reproduction intentionally retains the paper recipe.  Set
# PAPER_BATCH_SIZE=16 if you additionally want a batch-matched cross-system run.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_PATCH=${TRAIN_PATCH:-${REPO}/scripts/train_sparse_generator_colab.sh}
TRAIN_PAPER=${TRAIN_PAPER:-${REPO}/scripts/train.sh}
PLAN=${PLAN:-${REPO}/scripts/plan.sh}
COLLECTOR=${COLLECTOR:-${REPO}/scripts/collect_lpwm_benchmark.py}

PROFILE=${PROFILE:-screen}
STAGE=${STAGE:-train}
RUN=${RUN:-0}
SEEDS=${SEEDS:-0}
MODELS=${MODELS:-"dense_dense dense_sparse sparse_dense sparse_sparse"}
BENCHMARK_ID=${BENCHMARK_ID:-fair_lpwm_v1}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
BENCHMARK_DIR=${BENCHMARK_DIR:-${CKPT_BASE}/benchmarks/${BENCHMARK_ID}}
PLAN_ROOT=${PLAN_ROOT:-${CKPT_BASE}/planning/${BENCHMARK_ID}/${PROFILE}}

WANDB_ENTITY=${WANDB_ENTITY:-twojtys137-tw}
WANDB_PROJECT=${WANDB_PROJECT:-lpwm-sparse-generator}
WANDB_MODE=${WANDB_MODE:-online}
REQUIRE_WANDB_ONLINE=${REQUIRE_WANDB_ONLINE:-1}
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE REQUIRE_WANDB_ONLINE

NUM_WORKERS=${NUM_WORKERS:-2}
RESUME=${RESUME:-0}
REPLAN=${REPLAN:-0}
PLAN_SEED=${PLAN_SEED:-99}
GOAL_H=${GOAL_H:-5}
MAXITER=${MAXITER:-10}

check_planning_dependencies() {
  if ! python3 -c 'import submitit; import hydra_plugins.hydra_submitit_launcher' 2>/dev/null; then
    echo "Missing planning dependencies: submitit and/or hydra-submitit-launcher." >&2
    echo "Rerun the Colab dependency-install cell, then retry the planning stage." >&2
    return 2
  fi
}

case "${PROFILE}" in
  smoke)
    default_epochs=1
    default_patch_batch=4
    default_paper_batch=4
    default_rollouts=8
    default_projections=64
    default_nevals=4
    ;;
  screen)
    default_epochs=2
    default_patch_batch=16
    default_paper_batch=16
    default_rollouts=50
    default_projections=256
    default_nevals=50
    ;;
  full)
    default_epochs=2
    default_patch_batch=16
    default_paper_batch=64
    default_rollouts=all
    default_projections=8192
    default_nevals=50
    ;;
  *)
    echo "PROFILE must be smoke, screen, or full" >&2
    exit 2
    ;;
esac

EPOCHS=${EPOCHS:-${default_epochs}}
PATCH_BATCH_SIZE=${PATCH_BATCH_SIZE:-${default_patch_batch}}
PAPER_BATCH_SIZE=${PAPER_BATCH_SIZE:-${default_paper_batch}}
N_ROLLOUT=${N_ROLLOUT:-${default_rollouts}}
NUM_PROJECTIONS=${NUM_PROJECTIONS:-${default_projections}}
NEVALS=${NEVALS:-${default_nevals}}

case "${STAGE}" in
  train|plan|collect|all) ;;
  *)
    echo "STAGE must be train, plan, collect, or all" >&2
    exit 2
    ;;
esac

if [ "${RUN}" = "1" ] && { [ "${STAGE}" = "plan" ] || [ "${STAGE}" = "all" ]; }; then
  check_planning_dependencies
fi

if [ "${RUN}" = "1" ] && [ "${STAGE}" != "collect" ]; then
  : "${DATASET_DIR:?set DATASET_DIR to the root containing pusht_noise}"
  if [ ! -d "${DATASET_DIR}/pusht_noise" ]; then
    echo "Missing ${DATASET_DIR}/pusht_noise" >&2
    exit 2
  fi
  if [ "${REQUIRE_WANDB_ONLINE}" = "1" ] && [ "${WANDB_MODE}" != "online" ]; then
    echo "W&B online mode is required; authenticate or explicitly set REQUIRE_WANDB_ONLINE=0" >&2
    exit 2
  fi
fi

set_model_meta() {
  local model=$1
  case "${model}" in
    dense_dense)
      META_ARCH=patch_generator
      META_PREDICTOR=dense_generator
      META_LINK=identity
      META_TARGET_P=2
      META_STATE_SPARSE=false
      META_LAW_SPARSE=false
      META_EDGE_TOPK=64
      META_PAPER=0
      ;;
    dense_sparse)
      META_ARCH=patch_generator
      META_PREDICTOR=sparse_generator
      META_LINK=identity
      META_TARGET_P=2
      META_STATE_SPARSE=false
      META_LAW_SPARSE=true
      META_EDGE_TOPK=8
      META_PAPER=0
      ;;
    sparse_dense)
      META_ARCH=patch_generator
      META_PREDICTOR=dense_generator
      META_LINK=reprelu
      META_TARGET_P=1
      META_STATE_SPARSE=true
      META_LAW_SPARSE=false
      META_EDGE_TOPK=64
      META_PAPER=0
      ;;
    sparse_sparse)
      META_ARCH=patch_generator
      META_PREDICTOR=sparse_generator
      META_LINK=reprelu
      META_TARGET_P=1
      META_STATE_SPARSE=true
      META_LAW_SPARSE=true
      META_EDGE_TOPK=8
      META_PAPER=0
      ;;
    ltv)
      META_ARCH=patch_ltv
      META_PREDICTOR=ltv
      META_LINK=identity
      META_TARGET_P=2
      META_STATE_SPARSE=false
      META_LAW_SPARSE=false
      META_EDGE_TOPK=64
      META_PAPER=0
      ;;
    sparse_ltv)
      META_ARCH=patch_ltv
      META_PREDICTOR=sparse_ltv
      META_LINK=identity
      META_TARGET_P=2
      META_STATE_SPARSE=false
      META_LAW_SPARSE=true
      META_EDGE_TOPK=64
      META_PAPER=0
      ;;
    lewm)
      META_ARCH=cls_adaln_d384
      META_PREDICTOR=ar_adaln
      META_LINK=identity
      META_TARGET_P=2
      META_STATE_SPARSE=false
      META_LAW_SPARSE=na
      META_EDGE_TOPK=na
      META_PAPER=1
      ;;
    lpwm)
      META_ARCH=cls_adaln_d384
      META_PREDICTOR=ar_adaln
      META_LINK=reprelu
      META_TARGET_P=1
      META_STATE_SPARSE=true
      META_LAW_SPARSE=na
      META_EDGE_TOPK=na
      META_PAPER=1
      ;;
    *)
      echo "Unknown model '${model}'. Valid: dense_dense dense_sparse sparse_dense sparse_sparse ltv sparse_ltv lewm lpwm" >&2
      exit 2
      ;;
  esac
}

run_logged() {
  local log_file=$1
  shift
  local command_status tee_status status
  local -a pipeline_status
  mkdir -p "$(dirname "${log_file}")"

  {
    printf '\n[launcher] started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '[launcher] command:'
    printf ' %q' "$@"
    printf '\n'
  } | tee -a "${log_file}"

  set +e
  "$@" 2>&1 | tee -a "${log_file}"
  pipeline_status=("${PIPESTATUS[@]}")
  command_status=${pipeline_status[0]}
  tee_status=${pipeline_status[1]}
  status=${command_status}
  if [ "${status}" -eq 0 ] && [ "${tee_status}" -ne 0 ]; then
    status=${tee_status}
  fi
  printf '[launcher] finished=%s status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" | tee -a "${log_file}"
  set -e
  return "${status}"
}

log_launcher_error() {
  local log_file=$1
  shift
  printf '[launcher] ERROR: %s\n' "$*" | tee -a "${log_file}" >&2
}

register_run() {
  local model=$1
  local seed=$2
  local run_name=$3
  local planning_dir=$4
  python3 "${COLLECTOR}" register \
    --benchmark-dir "${BENCHMARK_DIR}" \
    --benchmark-id "${BENCHMARK_ID}" \
    --profile "${PROFILE}" \
    --model "${model}" \
    --architecture "${META_ARCH}" \
    --state-sparse "${META_STATE_SPARSE}" \
    --law-sparse "${META_LAW_SPARSE}" \
    --run-name "${run_name}" \
    --seed "${seed}" \
    --planning-dir "${planning_dir}"
}

train_one() {
  local model=$1
  local seed=$2
  set_model_meta "${model}"
  local run_name="${BENCHMARK_ID}_${PROFILE}_${model}_seed${seed}"
  local run_dir="${CKPT_BASE}/outputs/${run_name}"
  local planning_dir="${PLAN_ROOT}/${run_name}"
  local checkpoint="${run_dir}/checkpoints/model_latest.pth"
  local launcher_log="${run_dir}/launcher.log"

  if [ "${RUN}" != "1" ]; then
    printf '[dry-run][train] model=%-14s seed=%s profile=%s run=%s\n' \
      "${model}" "${seed}" "${PROFILE}" "${run_name}"
    return
  fi

  mkdir -p "${run_dir}"
  run_logged "${launcher_log}" \
    register_run "${model}" "${seed}" "${run_name}" "${planning_dir}"
  if [ -f "${checkpoint}" ] && [ "${RESUME}" != "1" ]; then
    echo "[skip][train] completed checkpoint: ${checkpoint}" | tee -a "${launcher_log}"
    return
  fi

  echo "[train] ${model}, seed=${seed}, profile=${PROFILE} -> ${run_dir}"
  if [ "${META_PAPER}" = "1" ]; then
    local paper_env=(
      CKPT_BASE="${CKPT_BASE}"
      RUN_NAME="${run_name}"
      SEED="${seed}"
      PREDICTOR=ar_adaln
      PROJ_DIM=384
      MUP=1
      MUP_LR=1e-4
      REG_WEIGHT=0.5
      REGULARIZER=rdmreg
      NUM_PROJECTIONS="${NUM_PROJECTIONS}"
      SAVE_EVERY=1
      WANDB_PROJECT="${WANDB_PROJECT}"
    )
    if [ "${N_ROLLOUT}" != "all" ]; then
      paper_env+=(N_ROLLOUT="${N_ROLLOUT}")
    fi
    if [ "${model}" = "lpwm" ]; then
      paper_env+=(MU=0)
    fi
    run_logged "${launcher_log}" env "${paper_env[@]}" "${TRAIN_PAPER}" \
      pusht 5 3 "${EPOCHS}" "${PAPER_BATCH_SIZE}" \
      "${META_LINK}" cls "${META_TARGET_P}" b "${NUM_WORKERS}"
  else
    run_logged "${launcher_log}" env \
      CKPT_BASE="${CKPT_BASE}" \
      RUN_NAME="${run_name}" \
      SEED="${seed}" \
      PREDICTOR="${META_PREDICTOR}" \
      STATE_LINK="${META_LINK}" \
      TARGET_P="${META_TARGET_P}" \
      MU=0 \
      AGG=btp \
      REG_WEIGHT=0.5 \
      EPOCHS="${EPOCHS}" \
      BATCH_SIZE="${PATCH_BATCH_SIZE}" \
      N_ROLLOUT="${N_ROLLOUT}" \
      NUM_PROJECTIONS="${NUM_PROJECTIONS}" \
      NUM_WORKERS="${NUM_WORKERS}" \
      NUM_LAWS=8 \
      LAW_RANK=32 \
      LAW_TOPK=2 \
      EDGE_TOPK="${META_EDGE_TOPK}" \
      LTV_TOPK=2 \
      QUERY_CHUNK_SIZE=16 \
      RESUME="${RESUME}" \
      SMOKE=0 \
      "${TRAIN_PATCH}"
  fi

  if [ ! -s "${run_dir}/metrics.jsonl" ]; then
    log_launcher_error "${launcher_log}" \
      "training returned success but ${run_dir}/metrics.jsonl is missing or empty"
    return 4
  fi
  if [ ! -f "${checkpoint}" ]; then
    log_launcher_error "${launcher_log}" \
      "training returned success but checkpoint is missing: ${checkpoint}"
    return 4
  fi
  echo "[launcher] verified metrics and checkpoint for ${run_name}" | tee -a "${launcher_log}"
}

plan_one() {
  local model=$1
  local seed=$2
  set_model_meta "${model}"
  local run_name="${BENCHMARK_ID}_${PROFILE}_${model}_seed${seed}"
  local checkpoint="${CKPT_BASE}/outputs/${run_name}/checkpoints/model_latest.pth"
  local planning_dir="${PLAN_ROOT}/${run_name}"
  local launcher_log="${planning_dir}/launcher.log"

  if [ "${RUN}" != "1" ]; then
    printf '[dry-run][plan]  model=%-14s seed=%s evals=%s plan_seed=%s run=%s\n' \
      "${model}" "${seed}" "${NEVALS}" "${PLAN_SEED}" "${run_name}"
    return
  fi
  if [ ! -f "${checkpoint}" ]; then
    mkdir -p "${planning_dir}"
    log_launcher_error "${launcher_log}" "Missing checkpoint for planning: ${checkpoint}"
    exit 3
  fi
  if [ -f "${planning_dir}/logs.json" ] \
      && grep -q 'final_eval/success_rate' "${planning_dir}/logs.json" \
      && [ "${REPLAN}" != "1" ]; then
    register_run "${model}" "${seed}" "${run_name}" "${planning_dir}"
    echo "[skip][plan] final evaluation exists: ${planning_dir}/logs.json" | tee -a "${launcher_log}"
    return
  fi
  if [ -e "${planning_dir}" ] && [ "${REPLAN}" = "1" ]; then
    planning_dir="${planning_dir}_retry_$(date +%Y%m%d-%H%M%S)"
    launcher_log="${planning_dir}/launcher.log"
  fi
  mkdir -p "${planning_dir}"
  run_logged "${launcher_log}" \
    register_run "${model}" "${seed}" "${run_name}" "${planning_dir}"

  echo "[plan] ${model}, train_seed=${seed}, paired_plan_seed=${PLAN_SEED} -> ${planning_dir}"
  run_logged "${launcher_log}" env \
    CKPT_BASE="${CKPT_BASE}" \
    PLAN_OUTPUT_DIR="${planning_dir}" \
    SEED="${PLAN_SEED}" \
    GOAL_H="${GOAL_H}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    "${PLAN}" plan_lewm.yaml "${run_name}" latest "${NEVALS}" "${MAXITER}"

  if [ ! -s "${planning_dir}/logs.json" ] \
      || ! grep -q 'final_eval/success_rate' "${planning_dir}/logs.json"; then
    log_launcher_error "${launcher_log}" \
      "planning returned success without final_eval/success_rate in ${planning_dir}/logs.json"
    return 4
  fi
  echo "[launcher] verified final planning metrics for ${run_name}" | tee -a "${launcher_log}"
}

train_matrix() {
  local seed model
  for seed in ${SEEDS}; do
    for model in ${MODELS}; do
      train_one "${model}" "${seed}"
    done
  done
}

plan_matrix() {
  local seed model
  for seed in ${SEEDS}; do
    for model in ${MODELS}; do
      plan_one "${model}" "${seed}"
    done
  done
}

collect_results() {
  local args=(
    collect
    --benchmark-dir "${BENCHMARK_DIR}"
    --ckpt-base "${CKPT_BASE}"
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-project "${WANDB_PROJECT}"
  )
  if [ "${WANDB_MODE}" = "online" ]; then
    args+=(--use-wandb)
  fi
  python3 "${COLLECTOR}" "${args[@]}"
}

echo "Benchmark: ${BENCHMARK_ID}; profile=${PROFILE}; stage=${STAGE}; run=${RUN}"
echo "Models: ${MODELS}; seeds: ${SEEDS}"
echo "Persistent root: ${CKPT_BASE}"
echo "Train: epochs=${EPOCHS}; patch_batch=${PATCH_BATCH_SIZE}; paper_batch=${PAPER_BATCH_SIZE}; rollouts=${N_ROLLOUT}; projections=${NUM_PROJECTIONS}"
echo "Plan: n_evals=${NEVALS}; seed=${PLAN_SEED}; goal_H=${GOAL_H}; max_iter=${MAXITER}"

case "${STAGE}" in
  train)
    train_matrix
    ;;
  plan)
    plan_matrix
    ;;
  collect)
    collect_results
    ;;
  all)
    train_matrix
    plan_matrix
    collect_results
    ;;
esac

if [ "${RUN}" != "1" ] && [ "${STAGE}" != "collect" ]; then
  echo "Dry-run only. Add RUN=1 after checking the matrix above."
fi
