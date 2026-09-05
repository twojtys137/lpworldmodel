# Native PushT baselines

These runs execute pinned official repositories in separate Python environments.
They do not relabel our sparse-generator models as LeWM or LpWM.

## What 10 epochs means

The [LeWM paper, Appendix E](https://arxiv.org/html/2603.19312v1#A5) explicitly trains
PushT for **10 epochs** on the full published dataset (described as 20,000 episodes,
mean length 196). Its generic YAML says 100; we deliberately override it to 10 to
follow the task-specific paper recipe. Batch size is 128, ViT-Tiny CLS192 is trained
from scratch, SIGReg has weight 0.09,1024 projections and 17 knots, and AdamW uses
lr 5e-5 and weight decay 1e-3. The native scheduler, bf 16 and clipping 1.0 remain enabled.
History 3 and frameskip 5 are unchanged. Reducing microbatch size plus gradient
accumulation is not assumed equivalent: SIGReg and BatchNorm depend on batch statistics.

The [official LpWM README](https://github.com/YilunKuang/lpworldmodel/tree/bdd812d9432cccda8c350086006401b436f91982)
uses 2 epochs, batch 64, CLS384, Deep-AdaLN, RDMReg with 8192 projections, Reprelu,
p 1,mu 0,agg=b,muP lr 1e-4 and regularizer weight 0.5. Our 10-epoch run is a **training
duration ablation** of that recipe. Evaluate saved epochs 2,5,10 without resuming.
The LpWM paper's Gaussian RDM comparison is `lewm_rdm_control`, not native SIGReg LeWM.

The [DINO-WM paper, Tables 11–12](https://arxiv.org/html/2411.04983v2#A1.SS9)
lists 18,500 PushT trajectories, batch 32 and 100 epochs. It uses frozen DINOv2 ViT-S/14
patch tokens. Its paper predictor lr 5e-5 differs from released YAML5e-4. A10-epoch
run would be an explicitly shortened experiment, not an exact reproduction.
DINO-WM training is therefore not automatically launched by this notebook.
An optional native evaluation command accepts an already acquired official checkpoint
under `OUTPUT/dinowm/outputs/RUN_NAME`, including its original `hydra.yaml`.

## Pins and compatibility

| Component | Revision |
|---|---|
| LeWM | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` |
| LpWM | `bdd812d9432cccda8c350086006401b436f91982` |
| DINO-WM | `0a9492fa12044b852ae9e001cc74604b79c8bb0c` |
| stable-worldmodel dependency | `abdced49809d5eae38e24b27dc7b635c502c4812` |
| Official LeWM HF model | `3970e07a65a74097a492f8954b073ec984afb09b` |
| Official LeWM HF dataset | `655cd446b9929369d7d406001da85c15d1457850` |

The stable-worldmodel commit is an explicit compatibility snapshot contemporary
with the LeWM repository, not a claim about the authors' original environment.
Transformers 4.57.6 preserves the checkpoint's encoder key layout. Native LeWM uses
Pymunk 7.0.1; legacy LpWM/DINO-WM use 6.11.1. Separate venvs prevent cross-contamination.
The installer creates an isolated Python 3.12 environment via uv, regardless of the
Colab kernel version. It installs PyTorch 2.11.0 / torchvision 0.26.0 (CUDA 12.8)
and NumPy 1.26.4 without inheriting Colab packages. NumPy 1.26 does not support
Python 3.13. An incompatible previous venv is retained with a timestamped suffix.
Package resolution, import checks and the complete installer output are recorded
in `environment/<method>-install.log`; resolved dependencies are saved by `pip freeze`.
This is a compatibility environment, not a fully locked historical environment.
CLI processes explicitly use Matplotlib's `Agg` backend so they do not inherit
Colab's notebook-only `matplotlib_inline` backend. The install smoke check imports
the actual stable-pretraining backbone submodule and Lightning, including their
lazy plotting dependencies, before checkpoint preparation.
The Linux decord 0.6.0 wheel contains a stale CPython 3.6 tag, which can make
`pip check` fail after a successful install. The launcher accepts only that exact
diagnostic and known Linux x86_64 wheel metadata, then requires an actual H.264
video decoding and indexed batch-read test. All other dependency errors remain fatal;
the package metadata is not rewritten. See the [upstream report](https://github.com/dmlc/decord/issues/366).
See [NumPy support](https://numpy.org/devdocs/release/1.26.4-notes.html),
[PyTorch package pairing](https://pytorch.org/get-started/previous-versions/) and
[uv Python installation](https://docs.astral.sh/uv/guides/install-python/).

LeWM's README manual `_object.ckpt` conversion no longer matches its actual eval.py.
The importer retains the published tensor values and retargets Hydra imports to
the pinned repository's `jepa.JEPA` and `module` classes. It requires `strict=True`
and a CPU encoding check before saving. No missing weights or randomly initialized
submodules are accepted.

The official data artifact is a **zstd-compressed HDF5 file**, not a tar archive.
It is downloaded from `quentinll/lewm-pusht` and streamed into
`pusht_expert_train.h5`. The native data loader is directed at HDF5 instead of the
newer YAML's Lance path; observations, actions and slicing are handled by upstream.
The script inventories actual episode/frame counts from the pinned official artifact.
It records any discrepancy with the paper's reported 20,000 episodes; it does not
assume the released artifact and the paper contain identical data quantities.
This is an explicit storage-format adaptation, not a claim of byte-identical Lance data.
Allow disk space for both the 13.1GB download and its expanded HDF5 file. Only the
checkpoint/log root is persistent; datasets remain on the Colab local disk.

## Protocols must remain separate

Native LeWM uses `swm/PushT-v1`, seed 42,50 evaluations, goals 25 raw steps ahead,
50 raw action budget, horizon 5 blocks of 5 actions, CEM300 candidates×30 iterations,
top 30 and a final visual embedding cost. Its `success_rate` is a **percentage 0–100**.
The 10-episode published-checkpoint run is a sanity check, not the final result.

Native LpWM uses the legacy DINO-WM PushT stack, seed 99,50 evaluations, goal_H5,
CEM300×30 and max_iter 10. With 5 actions per replan and frameskip 5 this allows
250 raw actions. Do not compare this number directly with LeWM's 50-action result.
DINO-WM's published `plan_pusht.yaml` leaves max_iter unlimited and uses visual plus
proprioceptive objective alpha 1. Our optional evaluation uses a finite configurable
cap and is labelled accordingly. A shared-protocol comparison is a separate experiment.

## Execution

Open `notebooks/native_worldmodels_colab.ipynb`. Run setup/download/checkpoint loading
first. Then run a native published-LeWM evaluation. If it remains 0%, the training
launcher stops: this points to an evaluation/dependency/data problem that extra
epochs cannot resolve. Once it succeeds, set `PHASE='lewm_train'` for native LeWM10ep.
LpWM is a separate optional stage after verifying the full legacy dataset.

LeWM training now uses `lewm_session.py` to invoke the unchanged pinned `train.py`.
It supplies session-boundary callbacks and an explicit full-state resume path to
the native `stable_pretraining.Manager`. `max_epochs=10` always stays fixed, so
the learning-rate scheduler is not rebuilt for a shorter training horizon.
`SESSION_EPOCHS=4` stops after at most four additional epochs (use fewer if setup
has already consumed much of the session). Planning evaluation is a separate
`PHASE='lewm_eval'` stage. Epoch counts alone are not a wall-clock guarantee.

`SPT_CACHE_DIR` is directed to `OUTPUT/lewm/training-state/RUN_NAME`. Native
epoch-end `last.ckpt` files therefore persist on Drive, alongside `progress.json`
with the explicit checkpoint path. Continue with the same run name, total epoch
target and seed, a new `SESSION_ID` for budget accounting, and
`RESUME_CHECKPOINT` set to that full checkpoint. We restore optimizer, scheduler
and Lightning loop state via `Manager(weights_only=False)`; model-only `.pt`
exports are rejected. New checkpoints also retain Python/NumPy/Torch and loader
generator states. Legacy checkpoints lack this extra RNG snapshot; neither case
claims bitwise equivalence for GPU kernels, prefetched data or worker RNG streams.

Before the session wrapper is used, the notebook runs `check_lewm_resume.py`:
a CPU Lightning integration test comparing three continuous epochs to one plus
two resumed epochs for model weights, AdamW state, scheduler state and counters.
This tests the resume mechanism, not native LeWM learning quality or GPU behavior.

For an already-running older launcher, **wait for a completed epoch and full
checkpoint save, then stop training without deleting the runtime**. Update only
the launcher checkout, and run `preserve-checkpoints` before ending that runtime.
It copies `.ckpt` files from the old local stable-pretraining cache (and the legacy
Lightning output directory) to `OUTPUT/lewm/recovered-checkpoints`, printing an
inventory. Select the checkpoint corresponding to the current run for resume;
its run name, seed and total epoch target are validated before loading.
Run `scripts/select_lewm_checkpoint.py --output OUTPUT --run-name RUN_NAME
--min-epoch 5` with the native venv to select the latest completed epoch from that
inventory. Original LeWM passes instantiated objects to Manager, so these older
checkpoints can lack the recipe in `hyper_parameters`. The selector then requires
the original `OUTPUT/lewm/checkpoints/RUN_NAME/config.yaml` and a matching W&B ID
inside the checkpoint. It validates the config's run name, seed and total epochs,
and saves a recipe sidecar bound to the checkpoint's SHA-256 for later resume.
It leaves checkpoint bytes intact and records the selection in
`OUTPUT/lewm/resume_selected.json`. New session-wrapper runs explicitly include
the three recipe fields in their Lightning hyperparameters.
Updating files does not retrofit checkpoint handling into a running process.
LpWM remains fresh-run-only; the legacy LpWM resume path has not been validated.

The budget guard counts **estimated wrapped-command CU only**, using the current
rate entered from Colab. Setup, download and idle GPU time also consume units and
are not measured by this ledger. Set TOTAL_CU to your remaining amount, not the
original 1800 if some has already been spent. Existing ledgers retain their fixed total; the notebook also checks the entered current balance. Train and evaluate sequentially.

Local validation of this launcher covers command manifests and syntax. Full
upstream dependencies, published-checkpoint loading and GPU planning must still
be validated in Colab; no native success rate has been produced in this workspace.
