# Theory experiments on Colab

[Open the notebook](https://colab.research.google.com/github/twojtys137/lpworldmodel/blob/experiment/theory-colab-1800/notebooks/theory_experiments_colab.ipynb)

The notebook defaults to a pilot on the existing
`fair_lpwm_v1_screen_{dense_dense,dense_sparse,sparse_dense,sparse_sparse}_seed0`
checkpoints. It does not retrain or overwrite them. The new campaign is `theory_v1`.

## What is implemented

| Question | Experiment | What the result can establish |
|---|---|---|
| A: does the routed graph describe local dependence? | Actual central differences at two scales; active-law edge union; support-switch counts | Sampled, one-step locality at the probed states. Not a global Jacobian/Lipschitz bound. |
| B: can a lower-rank goal cost retain action quality? | Uncentered goal-residual basis versus centered state PCA and random orthogonal basis; finite candidate regret; paired projected MPC | Terminal-cost compression, its finite-set error, and simulator performance. The full dynamics still execute. |
| C: does the model capture action order? | Swap two action blocks; compare the predicted terminal difference with a paired simulator difference | Empirical order sensitivity. Opposite actions are not treated as inverse flows, especially at contact. |
| Is the model better than doing little? | Ground-truth-action rollout versus persistence, zero-action sensitivity, paired zero/random control and CEM | A useful baseline before large training comparisons. |

The router and reparameterized ReLU have **surrogate training derivatives**.
Those derivatives are not the ordinary derivatives of their executed functions.
Experiment A uses finite differences of forward values and separately reports
changes of routing support. A single routing graph cannot certify a uniform
locality theorem across switching boundaries.

For B, let `r_a` be a terminal residual for candidate action sequence `a`, with
`P` patches and `D` channels, and let `V` have orthonormal columns. We use

```
J(a)   = ||r_a||_F^2 / (P D)
J_V(a) = ||r_a V||_F^2 / (P D)
```

The denominator remains **D**, not the retained rank. Orthogonality gives
`0 <= J(a)-J_V(a) = ||r_a (I-VV^T)||_F^2/(P D)`. On an evaluated finite candidate
set, selecting the minimum projected cost has full-cost regret at most the
maximum discarded energy on that set: add and subtract `J_V` at both minimizers
and use its minimality and the nonnegative discarded energy. The code verifies
this certificate. It is not a bound over unseen actions or simulator costs.
It also does not prove that the nonlinear predictor itself admits a reduced
state evolution. That stronger direction requires further mathematics.

## Data separation and evaluation

- Probe seed 991 selects 16 **distinct validation trajectories**: 8 calibration,
  8 test. No held-out test residual enters a fitted basis. Sixteen candidates
  include dataset actions, physically zero actions, a two-block swap and 13
  perturbations of the dataset sequence.
- The notebook passes `--context-frames 1` to match the current MPC's observed
  context, even if training used three frames. A diagnostic with three observed
  frames is a separate ablation; success with that richer context must not be
  presented as validation of the planner's single-image initialization.
- MPC goals are sampled from trajectories outside **both** probe splits when
  projections are used. All variants replay the exact same saved targets and
  simulator seeds. Checkpoint and basis SHA-256 hashes are checked on resume.
- The source trajectory and offset, environment information, and simulator
  seeds accompany every target. The new sampling implementation reads sequence
  lengths and only decodes the selected segment, rather than scanning all videos.
  Consequently, old planning scores must be re-evaluated on the new paired files.
- Alpha is zero: the present AdaLN models expose visual codes only. Repeating
  alpha=0 and alpha=1 would duplicate the same objective for these checkpoints.
- Zero/random baselines receive the same total action budget as MPC. Success is
  credited at the same MPC boundaries, including success followed by departure.
  The oracle only checks replay consistency; it does not establish model quality.
- The pilot uses 10 goals, 3 MPC iterations, 300 CEM candidates and 10 CEM steps.
  Later evaluations use 50 goals, 10 MPC iterations and 30 CEM steps.
- A goal bootstrap is included as a descriptive diagnostic for a **single**
  trained seed. Final inference must use trajectory clusters and training seeds.
  An all-zero/all-one bootstrap has a degenerate interval and is not evidence
  of zero uncertainty. Do not claim statistical superiority from the pilot.

## Budget and stages

| Stage | Command caps, CU | Schedule |
|---|---:|---|
| Pilot | 120 | 4 probes × 8, baseline planning 40, conditional projection planning 48 |
| Baselines | 480 | Full-data training 300, paired evaluation 180 |
| Factorial | 600 | Remaining 2×2 cells 300, 4 probes × 10, projected MPC matrix 260 |
| Replication | 300 | Seeds 1/2 training 180, paired evaluation 2 × 60 |
| Reserve | 300 | Unallocated; not automatically launched |

These are allocation caps, **not runtime predictions**. Colab compute-unit
consumption depends on the active hardware/rate; see the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html).
Enter the current displayed CU/hour rate. Wrapped command wall time is logged
and converted to an estimate. The wrapper terminates a command process group
near its cap; it does not read Google billing or stop idle GPU billing.
Setup, downloads, other runtimes and changing rates are outside its accounting.
The notebook disconnects the runtime after a successfully completed stage by
default; after an error, inspect the log and stop the runtime yourself.

Use one runtime per ledger. A local lock prevents concurrent launches in that
runtime; it does not coordinate separate Drive clients. A runtime that vanishes
without closing its ledger entry leaves the entire reservation charged. Existing
successful labels are skipped; changed commands under a reused label are rejected.
Interrupted training is not a completed cell of the scientific comparison.
Inspect persisted checkpoints and use a new attempt/campaign identifier; do not
silently compare unequal epoch counts or automatically add epochs with RESUME=1.

Training uses full data, two epochs, batch 16, 2048 RDM projections and the same
learning-rate settings as the existing launcher. These settings are a controlled
campaign, not a paper reproduction. In particular, the existing launcher's
`lewm` label uses **Gaussian RDM**, not LeWM's literal SIGReg. True pretrained
LeWM/DINO-WM comparisons require their checkpoint paths and matched evaluation
adapters. This package currently supports the repository's AdaLN LpWM/LeWM-style
checkpoints; it does not claim to evaluate unseen external checkpoints.

The primary 2×2 comparison matches data, epochs, batch size and regularization.
It does not equate actual FLOP or wall time. Sparse top-k support alone does not
establish computational speedup. Measure throughput and memory explicitly.
The replication phase is preselected as dense state + dense dynamics versus
dense state + sparse dynamics, seeds 1 and 2, supplementing seed 0. Compression
replication, more seeds and longer horizons are follow-up allocations, not hidden
promises inside the 1800-unit schedule.

## CEM execution efficiency

The campaign enables `cache_initial_encoding=true` and `candidate_batch_size=32`.
The initial observation code is reused across CEM action candidates; their
action encodings, rollouts and costs are still evaluated individually. Candidate
sampling is performed before chunking, so changing chunk size preserves the
candidate pool. Planning loads the world model in evaluation mode, and CEM runs
without autograd. Old callers retain the uncached/unbatched defaults.

The CPU equivalence test compares selected CEM actions with and without both
optimizations and counts encoded images (50 versus 4 in its small fixture).
This is not a measured Colab speedup. GPU arithmetic can differ slightly across
batch sizes; keep chunk size fixed across the scientific comparison.

## Outputs and status

- `manifest.json`, `samples.json`: source hashes, settings and split provenance.
- `progress.json`: each completed probe sample, surviving later interruption.
- `bases.npz`, `heldout_residuals.npz`, `compression.csv`: fitted bases and held-out costs.
- `results.json`: rollout, locality and optional simulator-order measurements.
- Paired planning: `campaign.json`, `target_provenance.json`, `results.md`, per-goal
  success lists, `paired_deltas.json`, individual Hydra output folders and logs.
- `budget.json` and `command_logs/`: estimated spend and process exit status.
- `sessions/`: environment freeze and exact code commit for each notebook session.

Local validation includes synthetic mathematical checks, the real PushT simulator
replay, checkpoint loading through the real `VWorldModel`, projection checks,
action alignment, split exclusions, CEM equivalence and command timeout behavior.
Small CPU reference outputs live in `experiments/results/`.
**No trained PushT checkpoint was evaluated on Colab when this package was created.**

## Local commands

```bash
python -m pytest -q tests
python -m experiments.theory_checks --output /tmp/lpwm-theory.json
python -m experiments.frozen_probe --help
python -m experiments.paired_eval --help
```

For Colab, open the notebook, grant Drive/secret access, check the checkpoint
inventory, enter the live CU/hour rate, and run `PHASE='pilot'`. Inspect model
action sensitivity and its advantage over persistence/zero/random controls
before interpreting any compression result as progress in planning.
