#!/bin/bash
# Controlled dense-state ablation. Dry-run by default; set RUN=1 to execute.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN="${HERE}/train_sparse_generator_colab.sh"
VARIANTS=${VARIANTS:-"ltv sparse_ltv dense_generator sparse_generator"}
EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
N_ROLLOUT=${N_ROLLOUT:-50}
NUM_PROJECTIONS=${NUM_PROJECTIONS:-256}
NUM_WORKERS=${NUM_WORKERS:-2}
ENV_NAME=${ENV_NAME:-pusht}

if [ "${SMOKE:-0}" = "1" ]; then
  EPOCHS=1
  BATCH_SIZE=4
  N_ROLLOUT=8
  NUM_PROJECTIONS=64
fi

for predictor in ${VARIANTS}; do
  case "${predictor}" in
    ltv)
      routing_overrides=()
      ;;
    sparse_ltv)
      routing_overrides=(LTV_TOPK="${LTV_TOPK:-2}")
      ;;
    sparse_generator)
      routing_overrides=(
        LAW_TOPK="${LAW_TOPK:-2}"
        EDGE_TOPK="${EDGE_TOPK:-8}"
        NUM_LAWS="${NUM_LAWS:-8}"
        LAW_RANK="${LAW_RANK:-32}"
        QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-16}"
      )
      ;;
    dense_generator)
      routing_overrides=(
        EDGE_TOPK="${DENSE_EDGE_TOPK:-64}"
        NUM_LAWS="${NUM_LAWS:-8}"
        LAW_RANK="${LAW_RANK:-32}"
        QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-16}"
      )
      ;;
    *)
      echo "unknown predictor: ${predictor}" >&2
      exit 1
      ;;
  esac

  run_name="dense_patch64_${predictor}_${ENV_NAME}_seed${SEED:-0}"
  if [ "${SMOKE:-0}" = "1" ]; then
    run_name=${run_name}_smoke
  fi

  if [ "${RUN:-0}" != "1" ]; then
    printf 'PREDICTOR=%s RUN_NAME=%s epochs=%s batch=%s rollouts=%s projections=%s\n' \
      "${predictor}" "${run_name}" "${EPOCHS}" "${BATCH_SIZE}" "${N_ROLLOUT}" "${NUM_PROJECTIONS}"
    continue
  fi

  env \
    PREDICTOR="${predictor}" RUN_NAME="${run_name}" \
    NUM_PROJECTIONS="${NUM_PROJECTIONS}" N_ROLLOUT="${N_ROLLOUT}" \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" NUM_WORKERS="${NUM_WORKERS}" \
    SMOKE=0 \
    WANDB_MODE="${WANDB_MODE:-offline}" \
    "${routing_overrides[@]}" \
    "${TRAIN}"
done
