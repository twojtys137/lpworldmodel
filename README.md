# LpWorldModel

<p align="left">
  <a href="https://arxiv.org/abs/2608.22764"><img src="https://img.shields.io/badge/arXiv-2608.22764-b31b1b.svg" /></a>
  <a href="https://wm-booth.org/"><img src="https://img.shields.io/badge/WM%40Booth-Accepted-4C6EF5.svg" /></a>
</p>

This repository contains the code for **LpWM: A Case for Sparse Representations in World Models** ([arXiv](https://arxiv.org/abs/2608.22764)).

## Overview

**LpWorldModel (LpWM)** is an action-conditioned Joint-Embedding Predictive Architecture (JEPA) world model that learns **sparse, non-negative representations** using **RDMReg** (Rectified Distribution-Matching Regularization).

RDMReg matches per-timestep feature distributions to **Rectified Generalized Gaussian** targets. The underlying Generalized Gaussian distributions before rectification are maximum-entropy under expected $\ell_p$-norm constraints, with $p \leq 1$ yielding sparse targets.

<div align="center">
  <img src="assets/fig1_lpworldmodel.png" width="100%"/>
</div>


## Repository Layout

```
train.py / plan.py      # Hydra entry points (train a world model / plan with it)
conf/                   # Hydra configs: env, encoder, predictor, regularizer, link, planner, ...
models/                 # VWorldModel + the JEPA modules (RDMReg, Link, predictor ladder), ViT encoder, muP
env/{wall,pusht}/       # environments
datasets/               # wall / pusht trajectory datasets
planning/               # CEM / GD / MPC planners + evaluator
scripts/                # portable run scripts (train / plan / reproduction grid)  ← see scripts/README.md
lpwm_swm/               # Piecewise / OGBench-Cube on stable-worldmodel (SEPARATE env)  ← see lpwm_swm/README.md
```

## Installation

To reproduce Section 3 of our paper on Wall and PushT, use the following installation guide. 

```bash
git clone https://github.com/YilunKuang/lpworldmodel
cd lpworldmodel
conda env create -f environment.yaml
conda activate lpwm
```

The `lpwm_swm/` component for Piecewise and OGBench-Cube environments for Section 4 of our paper is built on [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel/tree/main) and has its **own separate environment**. See [`lpwm_swm/README.md`](lpwm_swm/README.md#installation) for details. 

## Datasets

The Wall and PushT datasets are taken from [DINO-WM](https://github.com/gaoyuezhou/dino_wm), available
[here](https://osf.io/bmw48/?view_only=a56a296ce3b24cceaf408383a175ce28). Set the environment variable for the path to the dataset. 
```bash
export DATASET_DIR=/path/to/data
```
Expected structure (this repo uses only `pusht_noise` and `wall_single`):
```
data
├── pusht_noise
└── wall_single
```

## Core LpWM Code

The core LpWM implementation lives in:

- [`models/infojepa_modules.py`](models/infojepa_modules.py) — **RDMReg** (the sliced-Wasserstein
  distribution-matching regularizer), the shared **`Link`** `h(·)` (reprelu / relu / identity), and the
  **predictor types** (`ARPredictor` = Deep-/Shallow-AdaLN; `LinearDynamicsPredictor` = the (MLP) * LTV / LTI variants).
- [`models/visual_world_model.py`](models/visual_world_model.py) — the end-to-end JEPA path
  (`_forward_adaln` / `_rollout_adaln`).

The Piecewise / OGBench-Cube variant carries its own copies under [`lpwm_swm/`](lpwm_swm/) (RDMReg in `loss.py` , link function and predictors in `module.py`, world model in `model.py`).

## Quickstart

The [`scripts/`](scripts/README.md) folder contains instructions for how to run the entry point files `train.py` and `plan.py`. See `scripts/README.md` for more details. 

### Experimental branch: dense state, sparse generator

This fork also includes a slot-free experiment in which patch representations remain
dense and signed while exact top-k sparsity is applied only to shared low-rank dynamics
laws and token-to-token relations. See [`docs/sparse_generator.md`](docs/sparse_generator.md)
for the architecture, controls, diagnostics, ablations, and Colab workflow.

```bash
export DATASET_DIR=/path/to/data
SMOKE=1 bash scripts/train_sparse_generator_colab.sh
```

In the following section, we show examples of how to run the code with python commands. Before that, activate the conda environment and set `DATASET_DIR` environment.

### Training 

To train LpWM and LeWM with standard Transformer predictor with AdaLN action-conditioning at D=384, use the following commands.

```bash
# --- LpWM (sparse: reprelu link, p=1, mu=0), Deep-AdaLN(k), D=384 ---
# PushT
python train.py --config-name train_rdmreg.yaml \
  env=pusht frameskip=5 num_hist=3 encoder=vit_scratch \
  link=reprelu target_p=1 regularizer=rdmreg mu=0 agg=b \
  predictor=ar_adaln encoder.proj_dim=384 action_emb_dim=384 \
  mup=true training.mup_lr=1e-4 reg_weight=0.5 \
  training.epochs=2 training.batch_size=64 env.num_workers=20 \
  ckpt_base_path=./runs hydra.run.dir=./runs/outputs/lpwm_adaln_pusht_d384
# Wall
python train.py --config-name train_rdmreg.yaml \
  env=wall frameskip=5 num_hist=1 encoder=vit_scratch \
  link=reprelu target_p=1 regularizer=rdmreg mu=0 agg=b \
  predictor=ar_adaln encoder.proj_dim=384 action_emb_dim=384 \
  mup=true training.mup_lr=1e-4 reg_weight=0.5 \
  training.epochs=20 training.batch_size=128 env.num_workers=20 \
  ckpt_base_path=./runs hydra.run.dir=./runs/outputs/lpwm_adaln_wall_d384

# --- LeWM (dense Gaussian: identity link, p=2), Deep-AdaLN(k), D=384 ---
# PushT
python train.py --config-name train_rdmreg.yaml \
  env=pusht frameskip=5 num_hist=3 encoder=vit_scratch \
  link=identity target_p=2 regularizer=rdmreg agg=b \
  predictor=ar_adaln encoder.proj_dim=384 action_emb_dim=384 \
  mup=true training.mup_lr=1e-4 reg_weight=0.5 \
  training.epochs=2 training.batch_size=64 env.num_workers=20 \
  ckpt_base_path=./runs hydra.run.dir=./runs/outputs/lewm_adaln_pusht_d384
# Wall
python train.py --config-name train_rdmreg.yaml \
  env=wall frameskip=5 num_hist=1 encoder=vit_scratch \
  link=identity target_p=2 regularizer=rdmreg agg=b \
  predictor=ar_adaln encoder.proj_dim=384 action_emb_dim=384 \
  mup=true training.mup_lr=1e-4 reg_weight=0.5 \
  training.epochs=20 training.batch_size=128 env.num_workers=20 \
  ckpt_base_path=./runs hydra.run.dir=./runs/outputs/lewm_adaln_wall_d384
```

We note that we use the RDMReg formulation to match to isotropic Gaussian for fair comparison, i.e. all one-dimensional distribution matching under projections are performed using 2-Wasserstein distance instead of Epps-Pulley. You can also use the SIGReg implemetation under `models/infojepa_modules.py`, but it would require extra hyperparameter tuning.

For controlled experiment without introducing confounds, it's recommended to use 2-Wasserstein distance for generic distribution matching. Gaussian admits easy-to-evaluate closed form characteristic function, and hence Epps-Pulley should be in general more efficient, though it's specific to Gaussian.

### Planning (CEM + MPC)

Use the following code snippet for planning
```bash
python plan.py --config-name plan_lewm.yaml \
  ckpt_base_path=./runs model_name=lpwm_adaln_pusht_d384 model_epoch=latest \
  n_evals=50 planner.max_iter=10 goal_H=5
```

### Analysis of LpWM vs. LeWM across different predictor types and feature dimensions

[`scripts/reproduce_pusht.sh`](scripts/reproduce_pusht.sh) reproduces Fig 1 b) in the paper.

```bash
bash scripts/reproduce_pusht.sh              # dry-run: print every cell's train + eval command
RUN=1 bash scripts/reproduce_pusht.sh        # run the full grid (train + eval per cell)

# narrow the sweep with ARM_LIST / PRED_LIST / PD_LIST:
ARM_LIST=sparse PRED_LIST="ar_adaln ar_adaln_d1" PD_LIST="384 768" RUN=1 bash scripts/reproduce_pusht.sh
```

## Acknowledgements

This codebase is built upon [dino_wm](https://github.com/gaoyuezhou/dino_wm), [le-wm](https://github.com/lucas-maes/le-wm), and [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel/tree/main). We thank the authors for releasing their code under the MIT license.

## Citation

Please cite our work if you find it helpful:

```
@misc{kuang2026lpwmcasesparserepresentations,
      title={LpWM: A Case for Sparse Representations in World Models}, 
      author={Yilun Kuang and Yash Dagade and Quentin Le Lidec and Lucas Maes and Randall Balestriero and Yann LeCun},
      year={2026},
      eprint={2608.22764},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2608.22764}, 
}
```
