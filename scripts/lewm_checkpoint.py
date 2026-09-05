"""Validate native LeWM checkpoint metadata without importing GPU libraries."""
import hashlib
import json
from pathlib import Path


def validate_resume(checkpoint, run_name, max_epochs, seed, legacy_recipe=None):
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "loops", "epoch", "global_step"):
        if key not in checkpoint or checkpoint[key] is None:
            raise ValueError(f"Full training checkpoint required; missing {key}")
    if not checkpoint["optimizer_states"] or not checkpoint["lr_schedulers"]:
        raise ValueError("Checkpoint lacks optimizer or scheduler state")
    batch = checkpoint["loops"].get("fit_loop", {}).get("epoch_loop.batch_progress", {})
    if batch.get("is_last_batch") is not True:
        raise ValueError("Use a checkpoint saved at the end of a completed epoch; mid-epoch data replay is not supported")
    params = checkpoint.get("hyper_parameters", {})
    expected_recipe = {"output_model_name": run_name, "trainer.max_epochs": max_epochs,
                       "seed": seed}
    missing = [key for key in expected_recipe if key not in params]
    if missing and legacy_recipe is not None:
        # Upstream LeWM passes instantiated objects to Manager, which therefore
        # cannot inject the Hydra recipe into Lightning hyper_parameters.
        # Bind its separately saved config to this checkpoint through the run ID.
        if checkpoint.get("wandb", {}).get("id") != run_name:
            raise ValueError("Legacy checkpoint W&B run ID does not match the requested run")
        legacy = {"output_model_name": legacy_recipe.get("output_model_name"),
                  "trainer.max_epochs": legacy_recipe.get("trainer", {}).get("max_epochs"),
                  "seed": legacy_recipe.get("seed")}
        if legacy != expected_recipe:
            raise ValueError(f"Legacy resume recipe mismatch: {legacy!r}, expected {expected_recipe!r}")
        params = {**legacy, **params}  # Never replace conflicting checkpoint metadata.
    for key, expected in {"output_model_name": run_name, "trainer.max_epochs": max_epochs,
                          "seed": seed}.items():
        if params.get(key) != expected:
            raise ValueError(f"Resume recipe mismatch: {key}={params.get(key)!r}, expected {expected!r}")


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_recipe(path):
    sidecar = Path(path).with_suffix(".recipe.json")
    if not sidecar.is_file():
        return None
    record = json.loads(sidecar.read_text())
    if record["checkpoint_sha256"] != checkpoint_digest(path):
        raise ValueError("Legacy recipe sidecar does not match checkpoint contents")
    return record["config"]

