"""Frozen AdaLN checkpoints: persistence, actual locality, goal-cost projections,
and optional simulator action-order tests. No checkpoint training or mutation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from experiments.budget import save
from experiments.probe_core import (compression_rows, dynamics_diagnostics, fit_bases,
                                    latent_rollout, locality_probe, scene_variance_diagnostics)
from utils import seed


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_manifest(dataset, count, frames, seed_value):
    rng = np.random.default_rng(seed_value)
    valid = [i for i in range(len(dataset)) if dataset.get_seq_length(i) >= frames]
    if len(valid) < count:
        raise ValueError(f"Need {count} distinct trajectories; only {len(valid)} are long enough")
    chosen = rng.choice(valid, count, replace=False)
    return [{"trajectory": int(i),
             "offset": int(rng.integers(dataset.get_seq_length(i)-frames+1)),
             "split": "calibration" if n < count//2 else "test"}
            for n, i in enumerate(chosen)]


def candidates(actions, history_length, count, zero, sigma, rng):
    result = actions.unsqueeze(0).repeat(count, 1, 1)
    future = slice(history_length-1, None)
    result[1, future] = zero
    # Same action multiset, different order. No assertion about inverse actions.
    result[2, history_length-1] = actions[history_length]
    result[2, history_length] = actions[history_length-1]
    noise = torch.as_tensor(rng.normal(size=tuple(result[3:, future].shape)),
                            device=actions.device, dtype=actions.dtype)
    result[3:, future] += sigma * noise
    return result


@torch.no_grad()
def simulator_order(wm, dataset, config, latest_state, info, candidate_actions,
                    predicted, history_length, simulator_seed):
    import gym
    import env  # noqa: F401; registration
    from env.serial_vector_env import SerialVectorEnv

    environment = gym.make(config.env.name, *config.env.args, **config.env.kwargs)
    vector = SerialVectorEnv([environment])
    terminals, states = [], []
    try:
        vector.update_env([info])
        for index in (0, 2):
            normalized = candidate_actions[index, history_length-1:].cpu().reshape(-1, dataset.action_dim)
            physical = normalized * dataset.action_std + dataset.action_mean
            observations, state_path = vector.rollout(
                [simulator_seed], latest_state.cpu().numpy()[None], physical.numpy()[None])
            raw = torch.as_tensor(observations["visual"][:, -1:]).permute(0, 1, 4, 2, 3).float() / 255
            visual = dataset.transform(raw)
            terminal = wm.encode_obs_linked({"visual": visual.to(predicted.device)})["visual"]
            terminals.append(terminal[0, -1])
            states.append(state_path[0, -1].tolist())
    finally:
        environment.close()
    truth = (terminals[1]-terminals[0]).flatten()
    estimate = (predicted[2]-predicted[0]).flatten()
    norm = float(truth.norm())
    cosine = float(torch.nn.functional.cosine_similarity(truth, estimate, dim=0)) if norm > 1e-10 else None
    return {"true_delta_norm": norm, "predicted_delta_norm": float(estimate.norm()),
            "delta_cosine": cosine,
            "relative_delta_error": float((truth-estimate).norm())/norm if norm > 1e-10 else None,
            "original_final_state": states[0], "swapped_final_state": states[1],
            "scope": "paired simulator action swap; not a Lie-bracket certificate"}


@torch.no_grad()
def run(args):
    from plan import load_model

    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output exists: {out}; use a fresh attempt directory")
    if args.samples < 4 or args.samples % 2 or args.candidates < 4 or args.horizon < 2:
        raise ValueError("Need an even samples >=4, candidates >=4, horizon >=2")
    if args.locality_samples < 0 or args.simulator_samples < 0:
        raise ValueError("Diagnostic sample counts must be nonnegative")
    out.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir).resolve()
    checkpoint = run_dir / "checkpoints" / f"model_{args.epoch}.pth"
    config = OmegaConf.load(run_dir / "hydra.yaml")
    seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("GPU runtime required; choose a Colab GPU before running this phase")
    wm = load_model(checkpoint, config, config.num_action_repeat, device)
    wm.eval()
    if wm.action_conditioning != "adaln":
        raise ValueError("This adapter supports AdaLN LpWM/LeWM, not DINO-WM concat checkpoints")
    _, trajectories = hydra.utils.call(config.env.dataset, num_hist=config.num_hist,
                                        num_pred=config.num_pred, frameskip=config.frameskip)
    dataset = trajectories["valid"]
    history_length = int(getattr(args, "context_frames", None) or config.num_hist)
    if not 1 <= history_length <= int(config.num_hist):
        raise ValueError("context_frames must be between 1 and the trained num_hist")
    skip = int(config.frameskip)
    frame_count = (history_length-1+args.horizon)*skip+1
    manifest = sample_manifest(dataset, args.samples, frame_count, args.seed)
    save(out / "samples.json", manifest)
    metadata = {"schema": 1, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "config_sha256": sha256(run_dir / "hydra.yaml"), "args": vars(args),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "torch": torch.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "scope": "frozen checkpoint; disjoint calibration/test trajectories; no training"}
    save(out / "manifest.json", metadata)
    observations_for_fit, residuals_for_fit, residuals_test = [], [], []
    rows, locality, order = [], [], []
    scene_observations = []
    rng = np.random.default_rng(args.seed+1000)
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, sample in enumerate(manifest):
        frames = range(sample["offset"], sample["offset"]+frame_count)
        obs, act, states, info = dataset.get_frames(sample["trajectory"], frames)
        obs = {key: val[::skip].unsqueeze(0).to(device) for key, val in obs.items()}
        encoded = wm.encode_obs_linked(obs)["visual"]
        history = encoded[:, :history_length]
        target = encoded[:, -1]
        act = act[:-1].reshape(history_length-1+args.horizon, -1).to(device)
        zero = (-dataset.action_mean/dataset.action_std).repeat(skip).to(device)
        action_set = candidates(act, history_length, args.candidates, zero, args.action_sigma, rng)
        prediction = latent_rollout(wm, history.expand(args.candidates, -1, -1, -1), action_set)[:, -1]
        residual = prediction-target
        persistence = float((history[:, -1]-target).square().mean())
        row = {"sample": index, **sample, "persistence_mse": persistence,
               "model_gt_mse": float(residual[0].square().mean()),
               "zero_action_goal_mse": float(residual[1].square().mean()),
               "rollout_to_persistence_ratio": float(residual[0].square().mean())/max(persistence, 1e-12),
               "target_channel_std": float(encoded.flatten(0, 2).std(dim=0).mean()),
               "action_response_rms": float((prediction[0]-prediction[1]).square().mean().sqrt())}
        row.update(dynamics_diagnostics(encoded[0], prediction[0], prediction[1], history_length))
        rows.append(row)
        scene_observations.append(encoded[0].detach().cpu())
        if sample["split"] == "calibration":
            observations_for_fit.append(encoded.cpu().numpy().reshape(-1, encoded.shape[-1]))
            residuals_for_fit.append(residual.cpu().numpy().reshape(-1, encoded.shape[-1]))
        else:
            residuals_test.append(residual.cpu().numpy())
        if index < args.locality_samples:
            action_embeddings = wm.encode_act(act[None])[:, :history_length]
            locality.append({"sample": index, **locality_probe(wm.predictor, history, action_embeddings,
                                                             seed=args.seed+index)})
        if index >= args.samples//2 and len(order) < args.simulator_samples:
            order.append({"sample": index, **simulator_order(
                wm, dataset, config, states[(history_length-1)*skip], info, action_set,
                prediction, history_length, args.seed*100+index)})
        # Each completed sample survives a disconnected runtime.
        save(out / "progress.json", {"rollout": rows, "locality": locality, "order": order,
                                     "representation_diagnostics": scene_variance_diagnostics(
                                         torch.stack(scene_observations))})
        print(f"{index+1}/{args.samples}: {sample['split']}, rollout/persistence="
              f"{row['rollout_to_persistence_ratio']:.4g}", flush=True)
    bases = fit_bases(np.concatenate(observations_for_fit), np.concatenate(residuals_for_fit), args.seed)
    np.savez(out / "bases.npz", **bases)
    residuals = np.stack(residuals_test)
    np.savez_compressed(out / "heldout_residuals.npz", residuals=residuals)
    compressed = compression_rows(residuals, bases, args.ranks)
    save(out / "compression.json", compressed)
    with (out / "compression.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compressed[0]))
        writer.writeheader()
        writer.writerows(compressed)
    result = {"complete": True, "elapsed_seconds": time.monotonic()-started,
              "peak_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
              "rollout": rows, "locality": locality, "order": order,
              "representation_diagnostics": scene_variance_diagnostics(torch.stack(scene_observations)),
              "diagnostic_scope": "Scene variance keeps each patch/channel fixed. Legacy "
                  "target_channel_std pools positions and cannot exclude a positional shortcut. "
                  "Observed change is measured in this encoder's latent space; none of these "
                  "statistics alone demonstrates physical-state accuracy or actual collapse.",
              "all_finite_candidate_certificates_pass": all(x["certificate_pass"] for x in compressed),
              "scope": "terminal cost projection only: the full predictor is still executed; "
                       "finite-set cost bounds do not certify simulator success"}
    save(out / "results.json", result)
    print(f"Saved frozen probe: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--epoch", default="latest")
    p.add_argument("--seed", type=int, default=991)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--context-frames", type=int, help="Initial observed frames; use 1 to match current MPC")
    p.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64, 96])
    p.add_argument("--action-sigma", type=float, default=.5)
    p.add_argument("--locality-samples", type=int, default=2)
    p.add_argument("--simulator-samples", type=int, default=4)
    p.add_argument("--require-cuda", action="store_true")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    run(p.parse_args())


if __name__ == "__main__":
    main()
