#!/bin/bash
# Single training run of the from-scratch JEPA world model (PushT / Wall).
#
# Cluster-agnostic: this just runs `python train.py ...` on the current machine.
# Wrap it in whatever scheduler you use, or run it directly. Prerequisites:
#   - activate the `lpwm` conda env         (see README.md > Installation)
#   - export DATASET_DIR=/path/to/data      (folder containing pusht_noise/ and wall_single/)
#   - optional: `wandb login`, or `export WANDB_MODE=offline` to skip logging
# PushT/Wall are pure-Python (pymunk/pygame, numpy) -- no simulator install needed.
#
# Usage:
#   scripts/train.sh <env> <frameskip> <num_hist> <epochs> <batch> <link> <feature> [target_p] [agg] [num_workers]
#     <env>     : pusht | wall
#     <link>    : reprelu (sparse LpWM) | identity (dense LeWM)
#     <feature> : cls | patch (256 tokens) | patch64 (8x8 tokens, Colab-friendly)
#     [target_p]: 1 (sparse rectified-Laplace) | 2 (dense Gaussian);  [agg]: b (default) | btp | bp | bt
#
# Method-knob env-var overrides (all optional):
#   PREDICTOR (...|sparse_ltv|sparse_generator), PROJ_DIM (latent dim D),
#   MU (sparsity), MUP=1 MUP_LR=1e-4, REG_WEIGHT, LAMB_VAR, LAMB_COV, VAR_SPACE, REGULARIZER
#   (rdmreg|sigreg|none), TRAIN_ENCODER, SEED, RUN_NAME, WANDB_PROJECT, SAVE_EVERY, DEBUG=1,
#   and CKPT_BASE (where run dirs are written; default ./runs).
set -euo pipefail
ENV=${1:?usage: train.sh <env> <frameskip> <num_hist> <epochs> <batch> <link> <feature> [target_p] [agg] [num_workers]}
FRAMESKIP=${2:?need frameskip}; NUM_HIST=${3:?need num_hist}; EPOCHS=${4:?need epochs}
BATCH=${5:?need batch}; LINK=${6:?need link: reprelu|identity}; FEATURE=${7:?need feature: cls|patch|patch64}
case "${LINK}" in reprelu) DEFP=1.0;; identity) DEFP=2.0;; *) DEFP=1.0;; esac
TARGET_P=${8:-${DEFP}}; AGG=${9:-b}; NUM_WORKERS=${10:-20}
case "${FEATURE}" in
  cls) ENCODER=vit_scratch;;
  patch) ENCODER=vit_scratch_patch;;
  patch64) ENCODER=vit_scratch_patch64;;
  *) echo "feature must be cls|patch|patch64" >&2; exit 1;;
esac
ENCODER=${ENCODER_OVERRIDE:-${ENCODER}}

REPO=$(cd "$(dirname "$0")/.." && pwd)
: "${DATASET_DIR:?set DATASET_DIR to the dataset root (contains pusht_noise/ and wall_single/)}"
CKPT_BASE=${CKPT_BASE:-${REPO}/runs}

# headless pygame rendering (PushT) + single-process torch (generic; not cluster-specific)
export SDL_VIDEODRIVER=${SDL_VIDEODRIVER:-dummy}
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')

EXTRA=""; TAG=""
add(){ EXTRA="${EXTRA} $1"; }
[ -n "${LR:-}" ]         && { add "training.encoder_lr=${LR} training.predictor_lr=${LR} training.action_encoder_lr=${LR}"; TAG="${TAG}_lr${LR}"; }
[ -n "${REG_WEIGHT:-}" ] && { add "reg_weight=${REG_WEIGHT}"; TAG="${TAG}_rw${REG_WEIGHT}"; }
[ -n "${LAMB_VAR:-}" ]   && { add "lamb_var=${LAMB_VAR}"; TAG="${TAG}_lv${LAMB_VAR}"; }
[ -n "${LAMB_COV:-}" ]   && { add "lamb_cov=${LAMB_COV}"; TAG="${TAG}_lc${LAMB_COV}"; }
[ -n "${VAR_SPACE:-}" ]  && add "var_space=${VAR_SPACE}"
[ -n "${PREDICTOR:-}" ]  && { add "predictor=${PREDICTOR}"; TAG="${TAG}_${PREDICTOR}"; }
[ -n "${MU:-}" ]         && { add "mu=${MU}"; TAG="${TAG}_mu${MU}"; }
[ "${MUP:-0}" = "1" ]    && { add "mup=true"; TAG="${TAG}_mup"; }
[ -n "${MUP_LR:-}" ]     && { add "training.mup_lr=${MUP_LR}"; TAG="${TAG}_mlr${MUP_LR}"; }
[ -n "${SEED:-}" ]       && { add "training.seed=${SEED}"; TAG="${TAG}_seed${SEED}"; }
[ -n "${PROJ_DIM:-}" ]   && { add "encoder.proj_dim=${PROJ_DIM} action_emb_dim=${PROJ_DIM}"; TAG="${TAG}_pd${PROJ_DIM}"; }
[ -n "${ENCODER_DIM:-}" ] && add "embed_dim=${ENCODER_DIM}"
[ -n "${NUM_PROJECTIONS:-}" ] && add "regularizer.num_projections=${NUM_PROJECTIONS}"
[ -n "${N_ROLLOUT:-}" ] && add "env.dataset.n_rollout=${N_ROLLOUT}"
[ -n "${LAW_TOPK:-}" ] && add "predictor.law_topk=${LAW_TOPK}"
[ -n "${EDGE_TOPK:-}" ] && add "predictor.edge_topk=${EDGE_TOPK}"
[ -n "${NUM_LAWS:-}" ] && add "predictor.num_laws=${NUM_LAWS}"
[ -n "${LAW_RANK:-}" ] && add "predictor.law_rank=${LAW_RANK}"
[ -n "${QUERY_CHUNK_SIZE:-}" ] && add "predictor.query_chunk_size=${QUERY_CHUNK_SIZE}"
[ -n "${GATE_BALANCE_WEIGHT:-}" ] && add "predictor.gate_balance_weight=${GATE_BALANCE_WEIGHT}"
[ -n "${BASE_MODE:-}" ] && add "predictor.base_mode=${BASE_MODE}"
[ -n "${LTV_TOPK:-}" ] && add "predictor.topk=${LTV_TOPK}"
[ -n "${SAVE_EVERY:-}" ] && add "training.save_every_x_epoch=${SAVE_EVERY}"
[ -n "${TRAIN_ENCODER:-}" ] && add "model.train_encoder=${TRAIN_ENCODER}"
[ -n "${WANDB_PROJECT:-}" ] && add "wandb_project=${WANDB_PROJECT}"
[ "${DEBUG:-0}" = "1" ]  && add "debug=True"
REGULARIZER=${REGULARIZER:-rdmreg}

STAMP=$(date +%Y%m%d-%H%M%S); RAND=$(python3 -c 'import secrets; print(secrets.token_hex(3))')
RUNDIR=${CKPT_BASE}/outputs/lpwm_${LINK}_${FEATURE}_${ENV}_p${TARGET_P}_${AGG}${TAG}_${STAMP}_${RAND}
[ -n "${RUN_NAME:-}" ] && RUNDIR=${CKPT_BASE}/outputs/${RUN_NAME}

cd "${REPO}"
python train.py --config-name train_rdmreg.yaml \
    env="${ENV}" frameskip="${FRAMESKIP}" num_hist="${NUM_HIST}" \
    encoder="${ENCODER}" link="${LINK}" regularizer="${REGULARIZER}" \
    target_p="${TARGET_P}" agg="${AGG}" \
    training.epochs="${EPOCHS}" training.batch_size="${BATCH}" env.num_workers="${NUM_WORKERS}" \
    ckpt_base_path="${CKPT_BASE}" hydra.run.dir="${RUNDIR}" hydra.job.chdir=true ${EXTRA}
