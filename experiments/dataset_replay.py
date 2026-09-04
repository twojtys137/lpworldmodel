"""Compare recorded PushT frames/states with simulator reset and short replay.

No model checkpoint or GPU is required. Unlike oracle planning replay, the
reference here is the recorded dataset. Errors are descriptive: MP4 compression
and block velocities absent from the saved state can prevent exact agreement.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from experiments.budget import save


def state_errors(recorded, simulated):
    """PushT component errors; angle difference wraps correctly for any turns."""
    recorded, simulated = np.asarray(recorded), np.asarray(simulated)
    if recorded.shape != simulated.shape or recorded.ndim != 1 or recorded.size not in (5, 7):
        raise ValueError("Expected matching PushT state vectors with 5 or 7 components")
    if not np.isfinite(recorded).all() or not np.isfinite(simulated).all():
        raise ValueError("Non-finite PushT states")
    difference = simulated-recorded
    angle = float(np.arctan2(np.sin(difference[4]), np.cos(difference[4])))
    result = {"agent_position_error_px": float(np.linalg.norm(difference[:2])),
              "block_position_error_px": float(np.linalg.norm(difference[2:4])),
              "block_angle_error_rad": abs(angle)}
    if recorded.size == 7:
        result["agent_velocity_error"] = float(np.linalg.norm(difference[5:7]))
    return result


def compare_replay(dataset, environment, trajectory, offset, raw_steps, simulator_seed):
    """Dataset must expose raw [0,1] TCHW images (transform=None)."""
    if raw_steps < 1 or offset < 0 or offset+raw_steps >= dataset.get_seq_length(trajectory):
        raise ValueError("Recorded trajectory is too short for the requested replay")
    observations, normalized, recorded_states, info = dataset.get_frames(
        trajectory, range(offset, offset+raw_steps+1))
    recorded = torch.as_tensor(observations["visual"]).detach().cpu().numpy()
    if recorded.ndim != 4 or recorded.shape[1] != 3:
        raise ValueError("Expected raw recorded images in (time,3,height,width)")
    recorded = recorded.transpose(0, 2, 3, 1)
    if not np.isfinite(recorded).all() or recorded.min() < 0 or recorded.max() > 1:
        raise ValueError("Recorded pixels must be unnormalized [0,1]; use transform=None")
    recorded_states = torch.as_tensor(recorded_states).detach().cpu().numpy()
    # PushTDataset has already divided physical displacement by action_scale.
    # Undo its statistical normalization only; the environment applies the scale.
    actions = (torch.as_tensor(normalized[:raw_steps]) * dataset.action_std
               + dataset.action_mean).detach().cpu().numpy()
    environment.update_env(info)
    replay_obs, replay_states = environment.rollout(simulator_seed, recorded_states[0], actions)
    simulated = np.asarray(replay_obs["visual"], dtype=np.float64) / 255.0
    if simulated.shape != recorded.shape:
        raise ValueError(f"Native recorded/rendered image shapes differ: {recorded.shape} vs {simulated.shape}")
    if not np.isfinite(simulated).all():
        raise ValueError("Non-finite simulator pixels")
    rows = []
    for step in range(raw_steps+1):
        pixel_difference = simulated[step] - recorded[step]
        rows.append({"raw_step": step,
                     "pixel_mse_0_1": float(np.mean(pixel_difference**2)),
                     "pixel_mae_0_1": float(np.mean(abs(pixel_difference))),
                     **state_errors(recorded_states[step], replay_states[step])})
    result = {"trajectory": int(trajectory), "offset": int(offset), "simulator_seed": int(simulator_seed),
              "raw_steps": raw_steps, "reset": rows[0], "terminal": rows[-1],
              "per_step": rows,
              "maximum_errors": {key: max(row[key] for row in rows) for key in rows[0] if key != "raw_step"}}
    # Top: reset; bottom: terminal. Recorded frame is always on the left.
    panels = [np.concatenate((recorded[step], simulated[step]), axis=1)
              for step in (0, raw_steps)]
    preview = np.round(np.clip(np.concatenate(panels, axis=0), 0, 1)*255).astype(np.uint8)
    return result, preview


def run(args):
    import gym
    import imageio.v2 as imageio
    import env  # noqa: F401; register PushT

    if args.samples < 1 or args.raw_steps < 1:
        raise ValueError("samples and raw_steps must be positive")
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output exists: {output}; use a fresh diagnostic directory")
    config_path = Path(args.run_dir).resolve()/"hydra.yaml"
    config = OmegaConf.load(config_path)
    if config.env.name != "pusht":
        raise ValueError("This diagnostic currently supports the repository's PushT states only")
    if args.data_path:
        config.env.dataset.data_path = str(Path(args.data_path).resolve())
    _, datasets = hydra.utils.call(config.env.dataset, transform=None,
                                  num_hist=config.num_hist, num_pred=config.num_pred,
                                  frameskip=config.frameskip)
    dataset = datasets[args.split]
    valid = [i for i in range(len(dataset)) if dataset.get_seq_length(i) >= args.raw_steps+1]
    if len(valid) < args.samples:
        raise ValueError(f"Need {args.samples} distinct trajectories; only {len(valid)} are long enough")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(valid, args.samples, replace=False)
    manifest = [{"trajectory": int(i),
                 "offset": int(rng.integers(dataset.get_seq_length(i)-args.raw_steps))}
                for i in chosen]
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "args": vars(args), "samples": manifest, "checkpoint_required": False,
                "pixel_scale": "raw decoded/rendered RGB in [0,1], no training transform",
                "preview_layout": "top=reset, bottom=terminal; left=recorded, right=simulated",
                "scope": "Dataset-versus-simulator diagnostic, not oracle self-replay. "
                    "MP4 compression and unsaved block linear/angular velocities may cause errors. "
                    "No automatic fidelity or success threshold is applied."}
    save(output/"manifest.json", metadata)
    rows = []
    environment = gym.make(config.env.name, *config.env.args, **config.env.kwargs)
    try:
        for index, sample in enumerate(manifest):
            result, preview = compare_replay(dataset, environment, sample["trajectory"],
                                            sample["offset"], args.raw_steps, args.seed+index)
            result["preview"] = f"sample_{index:03d}_recorded_vs_simulated.png"
            imageio.imwrite(output/result["preview"], preview)
            rows.append(result)
            save(output/"progress.json", {"complete": False, "results": rows})
            print(f"{index+1}/{args.samples}: reset pixel MSE={result['reset']['pixel_mse_0_1']:.6g}, "
                  f"terminal block distance={result['terminal']['block_position_error_px']:.6g}px", flush=True)
    finally:
        environment.close()
    save(output/"results.json", {"complete": True, "metadata": metadata, "results": rows})
    print(f"Saved dataset replay diagnostic: {output/'results.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Training directory containing hydra.yaml; checkpoint optional")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-path", help="Optional path to pusht_noise (containing train/ and val/)")
    parser.add_argument("--split", choices=("train", "valid"), default="valid")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--raw-steps", type=int, default=25, help="Physical simulator steps, not frameskip blocks")
    parser.add_argument("--seed", type=int, default=99)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
