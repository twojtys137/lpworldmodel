"""Verify preserved LeWM checkpoints and select the latest completed epoch."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from lewm_checkpoint import checkpoint_digest, validate_resume


def select_checkpoint(output, run_name, minimum=5, max_epochs=10, seed=3072):
    output = Path(output).resolve()
    config_path = output / "lewm/checkpoints" / run_name / "config.yaml"
    # Preserve unrelated Hydra expressions (including the upstream eval resolver).
    # The three recipe fields being checked are literal launcher overrides.
    recipe = (OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
              if config_path.is_file() else None)
    inventory_path = output / "lewm/recovered-checkpoints/inventory.json"
    inventory = json.loads(inventory_path.read_text())
    valid = []
    for row in inventory:
        path = Path(row["checkpoint"]).resolve()
        state = None
        try:
            if not path.is_relative_to(output):
                raise ValueError("Checkpoint must be a preserved file under output")
            state = torch.load(path, map_location="cpu", weights_only=False)
            validate_resume(state, run_name, max_epochs, seed, legacy_recipe=recipe)
            epoch, step = int(state["epoch"]) + 1, int(state["global_step"])
            if epoch > max_epochs:
                raise ValueError("Checkpoint exceeds the requested training horizon")
            print(f"Verified: {path.name}: completed_epochs={epoch}, global_step={step}", flush=True)
            valid.append((epoch, step, str(path)))
        except Exception as error:
            print(f"Skipped: {path}\nReason: {error}", flush=True)
        finally:
            del state
    if not valid:
        raise RuntimeError(f"No matching checkpoint. Original recipe expected at {config_path}")
    epoch, step, filename = max(valid)
    if epoch < minimum:
        raise RuntimeError(f"Latest checkpoint has only {epoch} completed epochs; minimum={minimum}")
    path = Path(filename)
    digest = checkpoint_digest(path)
    # Keep original checkpoint bytes intact. A hash-bound sidecar supports legacy resume.
    if recipe is not None:
        sidecar = path.with_suffix(".recipe.json")
        temporary = sidecar.with_suffix(".tmp")
        temporary.write_text(json.dumps({"checkpoint_sha256": digest,
                                          "config_source": str(config_path), "config": recipe}, indent=2))
        temporary.replace(sidecar)
    selected = {"completed_epochs": epoch, "global_step": step, "checkpoint": filename,
                "run_name": run_name, "checkpoint_sha256": digest}
    destination = output / "lewm/resume_selected.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(selected, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(selected, indent=2), flush=True)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--min-epoch", type=int, default=5)
    args = parser.parse_args()
    select_checkpoint(args.output, args.run_name, args.min_epoch)
