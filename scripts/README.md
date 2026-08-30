# `scripts/` — portable run scripts

Plain shell wrappers around the two Python entry points (`train.py`, `plan.py`). They run
directly on the current machine — no scheduler or container assumed. On a cluster, wrap each
invocation in your own job submission.

## Prerequisites

1. Activate the environment (see the top-level [`README.md`](../README.md#installation)):
   ```bash
   conda activate lpwm
   ```
2. Download the exact original data and point `DATASET_DIR` at its root:
   ```bash
   python scripts/download_lpwmdatasets.py --dataset pusht_noise --output-dir /path/to/data
   # Use --dataset wall_single or --dataset all when needed.
   export DATASET_DIR=/path/to/data
   ```
3. Optional: `wandb login` to log runs, or `export WANDB_MODE=offline` to skip. The
   sparse-generator config accepts `WANDB_ENTITY` and `WANDB_PROJECT`.

PushT (`pymunk`/`pygame`) and Wall (`numpy`) are pure-Python — no simulator install is needed, and
the scripts set `SDL_VIDEODRIVER=dummy` for headless pygame rendering automatically.

Checkpoints/run dirs are written under `CKPT_BASE` (default `./runs`); override with `export CKPT_BASE=...`.

## Scripts

| script | what it does |
|---|---|
| `download_lpwmdatasets.py` | resumes, verifies, and extracts the original PushT/Wall OSF archives |
| `train.sh` | one training run of the from-scratch JEPA world model |
| `plan.sh`  | CEM + MPC planning eval of a trained checkpoint |
| `reproduce_pusht.sh` | drives the full PushT sparsity-vs-linearity grid (train + eval per cell) |
| `train_sparse_generator_colab.sh` | budgeted dense-patch / sparse-law PushT experiment |
| `sweep_sparse_generator_colab.sh` | dry-run/execute the controlled dense-state ablations |
| `benchmark_lpwm_sparse_generator_colab.sh` | fair 2x2 + literal LpWM/LeWM training, paired planning and result collection |
| `collect_lpwm_benchmark.py` | builds seed-level and aggregate JSON/CSV/Markdown benchmark tables |

Each script's header comment lists its positional args and env-var knobs.

## Examples

```bash
# sparse LpWM (mu=0), Deep-AdaLN(k) predictor, D=384, on PushT, 2 epochs:
PREDICTOR=ar_adaln PROJ_DIM=384 MU=0 MUP=1 MUP_LR=1e-4 REG_WEIGHT=0.5 RUN_NAME=my_lpwm \
  scripts/train.sh pusht 5 3 2 64 reprelu cls 1 b

# plan with the trained checkpoint:
scripts/plan.sh plan_lewm.yaml my_lpwm latest 50 10

# preview the full reproduction grid (prints the per-cell commands), then run it:
bash scripts/reproduce_pusht.sh
RUN=1 bash scripts/reproduce_pusht.sh

# dense signed patch state; exact top-k sparsity only in the dynamics generator:
SMOKE=1 bash scripts/train_sparse_generator_colab.sh
bash scripts/train_sparse_generator_colab.sh

# same architecture on the original Wall data:
ENV_NAME=wall SMOKE=1 bash scripts/train_sparse_generator_colab.sh

# inspect, then execute the controlled predictor/generator ablation:
bash scripts/sweep_sparse_generator_colab.sh
RUN=1 SMOKE=1 bash scripts/sweep_sparse_generator_colab.sh
```

## Fair LpWM comparison on Colab

The fair benchmark separates two questions:

1. a controlled patch-field 2x2 (`dense|sparse` state x `dense|sparse` generator),
   where architecture, parameter bank, data and optimization are fixed; and
2. literal CLS+D384 Deep-AdaLN LpWM/LeWM controls, evaluated by the same downstream
   PushT planner. Raw latent errors should only be compared within a matched block;
   planning success is the cross-architecture endpoint.

The launcher is a dry-run unless `RUN=1` is explicit. It uses deterministic run
names, skips completed checkpoints, refuses implicit continuation, persists planning
outputs under `CKPT_BASE`, and records every cell in a benchmark manifest.

```bash
# Inspect the four controlled cells, then run seed 0 screening.
PROFILE=screen STAGE=train SEEDS=0 bash scripts/benchmark_lpwm_sparse_generator_colab.sh
RUN=1 PROFILE=screen STAGE=train SEEDS=0 bash scripts/benchmark_lpwm_sparse_generator_colab.sh

# Promote the controlled cells to three seeds and evaluate identical planning goals.
RUN=1 PROFILE=screen STAGE=train SEEDS="0 1 2" \
  bash scripts/benchmark_lpwm_sparse_generator_colab.sh
RUN=1 PROFILE=screen STAGE=plan SEEDS="0 1 2" \
  bash scripts/benchmark_lpwm_sparse_generator_colab.sh

# Final full-data comparison, including exact paper controls and strong LTV controls.
export MODELS="lpwm lewm dense_dense dense_sparse sparse_dense sparse_sparse ltv sparse_ltv"
PROFILE=full STAGE=train MODELS="${MODELS}" SEEDS="0 1 2" \
  bash scripts/benchmark_lpwm_sparse_generator_colab.sh  # dry-run
RUN=1 PROFILE=full STAGE=train MODELS="${MODELS}" SEEDS="0 1 2" \
  bash scripts/benchmark_lpwm_sparse_generator_colab.sh
RUN=1 PROFILE=full STAGE=plan MODELS="${MODELS}" SEEDS="0 1 2" \
  bash scripts/benchmark_lpwm_sparse_generator_colab.sh

# Refresh local results from local logs and, when online, W&B.
PROFILE=full STAGE=collect bash scripts/benchmark_lpwm_sparse_generator_colab.sh
```

The final command writes `benchmark_results.{json,csv,md}` plus aggregate
`benchmark_aggregate.{json,csv}` under
`${CKPT_BASE}/benchmarks/${BENCHMARK_ID:-fair_lpwm_v1}`. Aggregates report mean and
95% Student-t intervals across completed training seeds. Planning uses one fixed
`PLAN_SEED` (99 by default), so every model sees the same evaluation goals.
