"""Paired oracle/zero/random/full/projected MPC evaluation, with resume checks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time

from experiments.budget import save
from experiments.frozen_probe import sha256
from experiments.probe_core import paired_bootstrap


ROOT = Path(__file__).resolve().parents[1]


def final_metrics(path, prefix):
    if not path.exists():
        return None
    matches = [row for line in path.read_text().splitlines() if line.strip()
               for row in [json.loads(line)] if f"{prefix}/success_rate" in row]
    return matches[-1] if matches else None


def execute(args, label, model, mode, target_file, extra=None):
    folder = Path(args.output).resolve() / label
    prefix = "oracle_eval" if mode == "gt_replay" else "final_eval"
    result = final_metrics(folder / "logs.json", prefix)
    if result is None:
        if folder.exists():
            raise RuntimeError(f"Incomplete attempt at {folder}; use a fresh output directory")
        folder.mkdir(parents=True)
        env = os.environ.copy()
        # Do not inherit stale projection, target or evaluation overrides.
        for key in ("OBJECTIVE_BASIS", "OBJECTIVE_BASIS_KEY", "OBJECTIVE_RANK",
                    "GOAL_FILE_PATH", "EXCLUDED_TRAJ_IDS"):
            env.pop(key, None)
        env.update({"CKPT_BASE": str(Path(args.ckpt_base).resolve()),
                    "PLAN_OUTPUT_DIR": str(folder), "EVALUATION_MODE": mode,
                    "SEED": str(args.seed), "GOAL_H": str(args.goal_h),
                    "OBJECTIVE_ALPHA": "0", "GOAL_SOURCE": "dset" if target_file is None else "file",
                    "CEM_OPT_STEPS": str(args.cem_steps), "CEM_NUM_SAMPLES": str(args.cem_samples),
                    "CEM_EVAL_EVERY": str(args.cem_steps),
                    "CEM_BATCH_SIZE": str(args.cem_batch_size), "CEM_CACHE_ENCODING": "true",
                    "WANDB_RUN_NAME": f"theory_{label}"})
        if target_file is not None:
            env["GOAL_FILE_PATH"] = str(target_file)
        env.update(extra or {})
        command = ["bash", str(ROOT/"scripts/plan.sh"), "plan_lewm.yaml", model, "latest",
                   str(args.n_evals), str(args.max_iter)]
        print(f"Running {label}: {mode}", flush=True)
        began = time.monotonic()
        with (folder/"launcher.log").open("w") as log:
            completed = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save(folder/"run.json", {"model": model, "mode": mode, "label": label,
                                "elapsed_seconds": time.monotonic()-began, "returncode": completed.returncode})
        if completed.returncode:
            raise RuntimeError(f"{label} failed ({completed.returncode}); inspect {folder/'launcher.log'}")
        result = final_metrics(folder/"logs.json", prefix)
    if result is None or len(result.get(f"{prefix}/successes", [])) != args.n_evals:
        raise RuntimeError(f"Incomplete per-goal metrics: {folder}")
    return {"label": label, "model": model, "mode": mode, "folder": str(folder),
            "success_rate": result[f"{prefix}/success_rate"],
            "successes": result[f"{prefix}/successes"],
            "mean_state_dist": result.get(f"{prefix}/mean_state_dist")}


def run(args):
    if args.n_evals < 1 or args.max_iter < 1 or args.goal_h < 1 or args.cem_steps < 1 or args.cem_samples < 30:
        raise ValueError("Positive evaluation sizes and >=30 CEM candidates are required")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    excluded, projections = set(), {}
    checkpoint_hashes = {}
    for model in args.models:
        if Path(model).name != model:
            raise ValueError("Model names must be direct children of outputs/")
        checkpoint = Path(args.ckpt_base)/"outputs"/model/"checkpoints/model_latest.pth"
        checkpoint_hashes[model] = sha256(checkpoint)
        if args.basis_root:
            probe = Path(args.basis_root).resolve()/model
            metadata = json.loads((probe/"manifest.json").read_text())
            if metadata["checkpoint_sha256"] != checkpoint_hashes[model]:
                raise ValueError(f"Projection fitted to a different checkpoint: {model}")
            samples = json.loads((probe/"samples.json").read_text())
            # Keep MPC goals disjoint from BOTH projection calibration and probe testing.
            excluded.update(sample["trajectory"] for sample in samples)
            projections[model] = probe/"bases.npz"
    identity = {"args": vars(args), "checkpoint_sha256": checkpoint_hashes,
                "excluded_trajectories": sorted(excluded),
                "basis_sha256": {name: sha256(path) for name, path in projections.items()}}
    manifest = out/"campaign.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Campaign settings/checkpoints changed; use a fresh output directory")
    save(manifest, identity)
    oracle = execute(args, "oracle", args.models[0], "gt_replay", None,
                     {"EXCLUDED_TRAJ_IDS": json.dumps(sorted(excluded), separators=(",", ":"))})
    if oracle["success_rate"] < .99:
        raise RuntimeError("Oracle replay <.99; paired model comparisons are blocked")
    target = out/"oracle/plan_targets.pkl"
    with target.open("rb") as handle:
        targets = pickle.load(handle)  # generated locally by this campaign
    samples = targets.get("target_samples")
    if samples is None or any(row["trajectory"] in excluded for row in samples):
        raise RuntimeError("Missing target provenance or overlap with projection calibration")
    save(out/"target_provenance.json", {"sha256": sha256(target), "samples": samples,
                                        "eval_seed": targets["eval_seed"]})
    rows = [oracle]
    for mode in ("zero_action", "random_action"):
        rows.append(execute(args, mode, args.models[0], mode, target))
        save(out/"results.json", rows)
    for model in args.models:
        rows.append(execute(args, f"{model}_full", model, "plan", target))
        save(out/"results.json", rows)
        if model in projections:
            for key in ("goal_residual", "state_pca", "random"):
                rows.append(execute(args, f"{model}_{key}_r{args.rank}", model, "plan", target,
                                    {"OBJECTIVE_BASIS": str(projections[model]),
                                     "OBJECTIVE_BASIS_KEY": key, "OBJECTIVE_RANK": str(args.rank)}))
                save(out/"results.json", rows)
    summaries = []
    for model in args.models:
        full = next(row for row in rows if row["label"] == f"{model}_full")
        for row in rows:
            if row["label"] in ("zero_action", "random_action") or (
                    row["model"] == model and row["label"] not in ("oracle", full["label"])):
                summaries.append({"reference": full["label"], "comparison": row["label"],
                                  **paired_bootstrap(full["successes"], row["successes"], args.seed)})
    save(out/"paired_deltas.json", {"comparisons": summaries,
        "scope": "descriptive goal bootstrap for this trained seed; repeated trajectory IDs "
                 "and training-seed variability require clustered/hierarchical final analysis"})
    lines = ["# Paired planning results", "", "| Variant | Success | State distance |",
             "|---|---:|---:|"]
    lines += [f"| {row['label']} | {row['success_rate']:.3f} | {row['mean_state_dist']} |" for row in rows]
    lines += ["", "Identical saved goals and simulator seeds. Alpha=0 (visual-only).",
              "Zero/random use the same total action budget as MPC.",
              "Projection changes the terminal objective; the full predictor still runs."]
    (out/"results.md").write_text("\n".join(lines)+"\n")
    print(f"Paired evaluation complete: {out/'results.md'}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-base", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--basis-root")
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--n-evals", type=int, default=10)
    p.add_argument("--goal-h", type=int, default=5)
    p.add_argument("--max-iter", type=int, default=3)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--cem-steps", type=int, default=10)
    p.add_argument("--cem-samples", type=int, default=300)
    p.add_argument("--cem-batch-size", type=int, default=32)
    run(p.parse_args())


if __name__ == "__main__":
    main()
