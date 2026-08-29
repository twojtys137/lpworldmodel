#!/bin/bash
# Colab-scale LpWM screening run for dense patch state + sparse shared generator.
# Required: DATASET_DIR contains the original pusht_noise/ or wall_single/ tree.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${DATASET_DIR:?set DATASET_DIR to the root created by download_lpwmdatasets.py}"

ENV_NAME=${ENV_NAME:-pusht}
case "${ENV_NAME}" in
  pusht)
    dataset_subdir=pusht_noise
    NUM_HIST=${NUM_HIST:-3}
    FRAMESKIP=${FRAMESKIP:-5}
    ;;
  wall)
    dataset_subdir=wall_single
    NUM_HIST=${NUM_HIST:-1}
    FRAMESKIP=${FRAMESKIP:-5}
    ;;
  *)
    echo "ENV_NAME must be pusht or wall" >&2
    exit 1
    ;;
esac
if [ ! -d "${DATASET_DIR}/${dataset_subdir}" ]; then
  echo "Missing ${DATASET_DIR}/${dataset_subdir}; run scripts/download_lpwmdatasets.py first" >&2
  exit 1
fi

EPOCHS=${EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-16}
N_ROLLOUT=${N_ROLLOUT:-50}
NUM_WORKERS=${NUM_WORKERS:-2}
NUM_PROJECTIONS=${NUM_PROJECTIONS:-256}
PREDICTOR=${PREDICTOR:-sparse_generator}
RUN_NAME=${RUN_NAME:-${PREDICTOR}_${ENV_NAME}_seed${SEED:-0}}
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}

if [ "${SMOKE:-0}" = "1" ]; then
  EPOCHS=1
  BATCH_SIZE=4
  N_ROLLOUT=8
  NUM_PROJECTIONS=64
  RUN_NAME=${RUN_NAME}_smoke
fi
RUN_DIR=${CKPT_BASE}/outputs/${RUN_NAME}
if [ -f "${RUN_DIR}/checkpoints/model_latest.pth" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "Checkpoint already exists: ${RUN_DIR}/checkpoints/model_latest.pth" >&2
  echo "Use a new RUN_NAME, or set RESUME=1 to train for EPOCHS additional epochs." >&2
  exit 1
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_ENTITY=${WANDB_ENTITY:-twojtys137-tw}
export WANDB_PROJECT=${WANDB_PROJECT:-lpwm-sparse-generator}
if [ "${REQUIRE_WANDB_ONLINE:-0}" = "1" ] && [ "${WANDB_MODE}" != "online" ]; then
  echo "W&B online mode is required; authenticate and set WANDB_MODE=online" >&2
  exit 1
fi
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
if [ -z "${MASTER_PORT:-}" ]; then
  MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
fi
export MASTER_PORT

case "${PREDICTOR}" in
  sparse_generator)
    predictor_overrides=(
      "predictor.num_laws=${NUM_LAWS:-8}"
      "predictor.law_rank=${LAW_RANK:-32}"
      "predictor.law_topk=${LAW_TOPK:-2}"
      "predictor.edge_topk=${EDGE_TOPK:-8}"
      "predictor.query_chunk_size=${QUERY_CHUNK_SIZE:-16}"
      "predictor.gate_balance_weight=${GATE_BALANCE_WEIGHT:-0.0}"
      "predictor.base_mode=${BASE_MODE:-identity}"
    )
    ;;
  dense_generator)
    num_laws=${NUM_LAWS:-8}
    predictor_overrides=(
      "predictor.num_laws=${num_laws}"
      "predictor.law_rank=${LAW_RANK:-32}"
      "predictor.law_topk=${num_laws}"
      "predictor.edge_topk=${EDGE_TOPK:-64}"
      "predictor.query_chunk_size=${QUERY_CHUNK_SIZE:-16}"
      "predictor.base_mode=${BASE_MODE:-identity}"
    )
    ;;
  sparse_ltv)
    predictor_overrides=("predictor.topk=${LTV_TOPK:-2}")
    ;;
  ltv)
    predictor_overrides=()
    ;;
  *)
    echo "PREDICTOR must be ltv, sparse_ltv, dense_generator, or sparse_generator" >&2
    exit 1
    ;;
esac

cd "${REPO}"
echo "Environment: ${ENV_NAME}; data: ${DATASET_DIR}/${dataset_subdir}"
echo "Persistent run directory: ${RUN_DIR}"
echo "W&B: ${WANDB_MODE} (${WANDB_ENTITY}/${WANDB_PROJECT})"
python train.py --config-name train_sparse_generator.yaml \
  env="${ENV_NAME}" frameskip="${FRAMESKIP}" num_hist="${NUM_HIST}" \
  predictor="${PREDICTOR}" \
  training.epochs="${EPOCHS}" training.batch_size="${BATCH_SIZE}" \
  training.seed="${SEED:-0}" env.num_workers="${NUM_WORKERS}" \
  env.dataset.n_rollout="${N_ROLLOUT}" \
  regularizer.num_projections="${NUM_PROJECTIONS}" \
  "${predictor_overrides[@]}" \
  ckpt_base_path="${CKPT_BASE}" hydra.run.dir="${RUN_DIR}" \
  hydra.job.chdir=true
