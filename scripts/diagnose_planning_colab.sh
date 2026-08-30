#!/bin/bash
# Cheap, paired PushT planning diagnostic for Colab/MyDrive.
#
# 1. Replays exact dataset actions and requires oracle success >= 0.99.
# 2. Reuses the exact saved targets for every model.
# 3. Runs 10 goals x 3 MPC iterations for alpha in {0, 1}.
#
# Dry-run is the default:
#   bash scripts/diagnose_planning_colab.sh
# Execute after inspecting the matrix:
#   RUN=1 bash scripts/diagnose_planning_colab.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
PLAN=${PLAN:-${REPO}/scripts/plan.sh}
COLLECTOR=${COLLECTOR:-${REPO}/scripts/collect_planning_diagnostic.py}

RUN=${RUN:-0}
BENCHMARK_ID=${BENCHMARK_ID:-fair_lpwm_v1}
PROFILE=${PROFILE:-screen}
MODELS=${MODELS:-"dense_dense dense_sparse sparse_dense sparse_sparse"}
TRAIN_SEED=${TRAIN_SEED:-0}
ORACLE_MODEL=${ORACLE_MODEL:-dense_dense}
PLAN_SEED=${PLAN_SEED:-99}
NEVALS=${NEVALS:-10}
GOAL_H=${GOAL_H:-5}
MAXITER=${MAXITER:-3}
ALPHAS=${ALPHAS:-"0 1"}
CEM_EVAL_EVERY=${CEM_EVAL_EVERY:-10}
ORACLE_MIN_SUCCESS=${ORACLE_MIN_SUCCESS:-0.99}

CKPT_BASE=${CKPT_BASE:-${REPO}/runs}
DIAG_NAME=${DIAG_NAME:-screen_seed${TRAIN_SEED}_planseed${PLAN_SEED}_n${NEVALS}_h${GOAL_H}}
DIAG_ROOT=${DIAG_ROOT:-${CKPT_BASE}/planning/${BENCHMARK_ID}/diagnostic/${DIAG_NAME}}
RESULT_DIR=${RESULT_DIR:-${CKPT_BASE}/benchmarks/${BENCHMARK_ID}/planning_diagnostic/${DIAG_NAME}}

WANDB_ENTITY=${WANDB_ENTITY:-twojtys137-tw}
WANDB_PROJECT=${WANDB_PROJECT:-lpwm-sparse-generator}
WANDB_MODE=${WANDB_MODE:-online}
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE

run_name() {
  printf '%s_%s_%s_seed%s' "${BENCHMARK_ID}" "${PROFILE}" "$1" "${TRAIN_SEED}"
}

run_logged() {
  local log_file=$1
  shift
  local command_status tee_status status
  local -a pipeline_status
  mkdir -p "$(dirname "${log_file}")"
  {
    printf '\n[diagnostic] started=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf '[diagnostic] command:'
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
  printf '[diagnostic] finished=%s status=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${status}" | tee -a "${log_file}"
  set -e
  return "${status}"
}

check_runtime() {
  : "${DATASET_DIR:?set DATASET_DIR to the root containing pusht_noise}"
  if [ ! -d "${DATASET_DIR}/pusht_noise" ]; then
    echo "Missing ${DATASET_DIR}/pusht_noise" >&2
    exit 2
  fi
  python3 - <<'PY'
import hydra_plugins.hydra_submitit_launcher  # noqa: F401
import pymunk
import submitit  # noqa: F401

space = pymunk.Space()
if not hasattr(space, "add_collision_handler"):
    raise SystemExit(
        f"Pymunk {pymunk.version} is incompatible; install pymunk==6.11.1."
    )
space.add_collision_handler(0, 0)
print(f"Planning diagnostic preflight: Pymunk {pymunk.version}")
PY
}

checkpoint_for() {
  printf '%s/outputs/%s/checkpoints/model_latest.pth' "${CKPT_BASE}" "$(run_name "$1")"
}

register_run() {
  local run_dir=$1
  local kind=$2
  local model=$3
  local alpha=${4:-}
  local args=(
    register
    --run-dir "${run_dir}"
    --kind "${kind}"
    --model "${model}"
    --train-seed "${TRAIN_SEED}"
    --plan-seed "${PLAN_SEED}"
    --n-evals "${NEVALS}"
    --goal-h "${GOAL_H}"
    --max-iter "${MAXITER}"
  )
  if [ -n "${alpha}" ]; then
    args+=(--alpha "${alpha}")
  fi
  python3 "${COLLECTOR}" "${args[@]}"
}

fresh_attempt_dir() {
  local base=$1
  local completion_key=$2
  local candidate completed=""
  for candidate in "${base}" "${base}"_retry_*; do
    if [ -s "${candidate}/logs.json" ] \
        && grep -q "${completion_key}" "${candidate}/logs.json"; then
      completed=${candidate}
    fi
  done
  if [ -n "${completed}" ]; then
    printf '%s' "${completed}"
  elif [ -e "${base}" ]; then
    printf '%s_retry_%s' "${base}" "$(date +%Y%m%d-%H%M%S)"
  else
    printf '%s' "${base}"
  fi
}

ORACLE_ACTIVE_DIR=""
run_oracle() {
  local model=${ORACLE_MODEL}
  local model_run
  model_run=$(run_name "${model}")
  local base_dir="${DIAG_ROOT}/oracle_${model}_seed${TRAIN_SEED}"

  if [ "${RUN}" != "1" ]; then
    echo "[dry-run][oracle] model=${model} evals=${NEVALS} seed=${PLAN_SEED} -> ${base_dir}"
    ORACLE_ACTIVE_DIR=${base_dir}
    return
  fi
  if [ ! -f "$(checkpoint_for "${model}")" ]; then
    echo "Missing oracle checkpoint: $(checkpoint_for "${model}")" >&2
    exit 3
  fi

  ORACLE_ACTIVE_DIR=$(fresh_attempt_dir "${base_dir}" "oracle_eval/success_rate")
  register_run "${ORACLE_ACTIVE_DIR}" oracle "${model}"
  if [ -s "${ORACLE_ACTIVE_DIR}/logs.json" ] \
      && grep -q 'oracle_eval/success_rate' "${ORACLE_ACTIVE_DIR}/logs.json"; then
    echo "[skip][oracle] completed: ${ORACLE_ACTIVE_DIR}/logs.json"
  else
    run_logged "${ORACLE_ACTIVE_DIR}/launcher.log" env \
      CKPT_BASE="${CKPT_BASE}" \
      PLAN_OUTPUT_DIR="${ORACLE_ACTIVE_DIR}" \
      SEED="${PLAN_SEED}" \
      GOAL_H="${GOAL_H}" \
      GOAL_SOURCE=dset \
      EVALUATION_MODE=gt_replay \
      OBJECTIVE_ALPHA=0 \
      WANDB_RUN_NAME="diagnostic_oracle_${model_run}" \
      WANDB_ENTITY="${WANDB_ENTITY}" \
      WANDB_PROJECT="${WANDB_PROJECT}" \
      WANDB_MODE="${WANDB_MODE}" \
      "${PLAN}" plan_lewm.yaml "${model_run}" latest "${NEVALS}" 0
  fi

  if [ ! -s "${ORACLE_ACTIVE_DIR}/plan_targets.pkl" ]; then
    echo "Oracle did not persist paired targets: ${ORACLE_ACTIVE_DIR}/plan_targets.pkl" >&2
    exit 4
  fi
  python3 "${COLLECTOR}" check-oracle \
    --logs "${ORACLE_ACTIVE_DIR}/logs.json" \
    --min-success "${ORACLE_MIN_SUCCESS}"
}

run_planner() {
  local model=$1
  local alpha=$2
  local alpha_tag=${alpha//./p}
  local model_run
  model_run=$(run_name "${model}")
  local base_dir="${DIAG_ROOT}/${model}_seed${TRAIN_SEED}_alpha${alpha_tag}"

  if [ "${RUN}" != "1" ]; then
    echo "[dry-run][plan] model=${model} alpha=${alpha} evals=${NEVALS} max_iter=${MAXITER} paired_targets=oracle"
    return
  fi
  if [ ! -f "$(checkpoint_for "${model}")" ]; then
    echo "Missing planner checkpoint: $(checkpoint_for "${model}")" >&2
    exit 3
  fi

  local run_dir
  run_dir=$(fresh_attempt_dir "${base_dir}" "final_eval/success_rate")
  register_run "${run_dir}" plan "${model}" "${alpha}"
  if [ -s "${run_dir}/logs.json" ] \
      && grep -q 'final_eval/success_rate' "${run_dir}/logs.json"; then
    echo "[skip][plan] completed: ${run_dir}/logs.json"
    return
  fi

  run_logged "${run_dir}/launcher.log" env \
    CKPT_BASE="${CKPT_BASE}" \
    PLAN_OUTPUT_DIR="${run_dir}" \
    SEED="${PLAN_SEED}" \
    GOAL_H="${GOAL_H}" \
    GOAL_SOURCE=file \
    GOAL_FILE_PATH="${ORACLE_ACTIVE_DIR}/plan_targets.pkl" \
    EVALUATION_MODE=plan \
    OBJECTIVE_ALPHA="${alpha}" \
    CEM_EVAL_EVERY="${CEM_EVAL_EVERY}" \
    WANDB_RUN_NAME="diagnostic_${model_run}_alpha${alpha_tag}" \
    WANDB_ENTITY="${WANDB_ENTITY}" \
    WANDB_PROJECT="${WANDB_PROJECT}" \
    WANDB_MODE="${WANDB_MODE}" \
    "${PLAN}" plan_lewm.yaml "${model_run}" latest "${NEVALS}" "${MAXITER}"

  if [ ! -s "${run_dir}/logs.json" ] \
      || ! grep -q 'final_eval/success_rate' "${run_dir}/logs.json"; then
    echo "Planner did not persist final metrics: ${run_dir}/logs.json" >&2
    exit 4
  fi
}

echo "Planning diagnostic: ${BENCHMARK_ID}/${DIAG_NAME}; run=${RUN}"
echo "Models: ${MODELS}; train_seed=${TRAIN_SEED}; alphas: ${ALPHAS}"
echo "Paired targets: n_evals=${NEVALS}; plan_seed=${PLAN_SEED}; goal_H=${GOAL_H}"
echo "MPC: max_iter=${MAXITER}; CEM eval telemetry every ${CEM_EVAL_EVERY} steps"
echo "Persistent root: ${DIAG_ROOT}"

if [ "${RUN}" = "1" ]; then
  check_runtime
fi

run_oracle
for model in ${MODELS}; do
  for alpha in ${ALPHAS}; do
    run_planner "${model}" "${alpha}"
  done
done

if [ "${RUN}" = "1" ]; then
  python3 "${COLLECTOR}" collect --root "${DIAG_ROOT}" --output-dir "${RESULT_DIR}"
  echo "Diagnostic summary: ${RESULT_DIR}/diagnostic_results.md"
else
  echo "Dry-run only. Add RUN=1 to execute; the oracle gate stops the matrix on protocol failure."
fi
