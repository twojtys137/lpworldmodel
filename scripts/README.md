# `scripts/` — portable run scripts

Plain shell wrappers around the two Python entry points (`train.py`, `plan.py`). They run
directly on the current machine — no scheduler or container assumed. On a cluster, wrap each
invocation in your own job submission.

## Prerequisites

1. Activate the environment (see the top-level [`README.md`](../README.md#installation)):
   ```bash
   conda activate lpwm
   ```
2. Point `DATASET_DIR` at the dataset root (the folder holding `pusht_noise/` and `wall_single/`):
   ```bash
   export DATASET_DIR=/path/to/data
   ```
3. Optional: `wandb login` to log runs, or `export WANDB_MODE=offline` to skip.

PushT (`pymunk`/`pygame`) and Wall (`numpy`) are pure-Python — no simulator install is needed, and
the scripts set `SDL_VIDEODRIVER=dummy` for headless pygame rendering automatically.

Checkpoints/run dirs are written under `CKPT_BASE` (default `./runs`); override with `export CKPT_BASE=...`.

## Scripts

| script | what it does |
|---|---|
| `train.sh` | one training run of the from-scratch JEPA world model |
| `plan.sh`  | CEM + MPC planning eval of a trained checkpoint |
| `reproduce_pusht.sh` | drives the full PushT sparsity-vs-linearity grid (train + eval per cell) |
| `train_sparse_generator_colab.sh` | budgeted dense-patch / sparse-law PushT experiment |
| `sweep_sparse_generator_colab.sh` | dry-run/execute the controlled dense-state ablations |

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

# inspect, then execute the controlled predictor/generator ablation:
bash scripts/sweep_sparse_generator_colab.sh
RUN=1 SMOKE=1 bash scripts/sweep_sparse_generator_colab.sh
```
