"""Run unchanged upstream LeWM with bounded sessions and explicit full-state resume."""
import argparse
import json
import os
from pathlib import Path
import random
import runpy
import sys

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


from lewm_checkpoint import checkpoint_digest, load_legacy_recipe, validate_resume


class SessionBoundary(Callback):
    def __init__(self, epochs, status_path):
        self.epochs = epochs
        self.status_path = Path(status_path)
        self.stop_epoch = None
        self.completed = 0
        self.rng = None

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.rng = checkpoint.get("native_session_rng")

    def on_train_start(self, trainer, pl_module):
        self.completed = trainer.current_epoch
        self.stop_epoch = min(trainer.max_epochs, trainer.current_epoch + self.epochs)
        if self.rng is not None:
            random.setstate(self.rng["python"])
            np.random.set_state(self.rng["numpy"])
            torch.set_rng_state(self.rng["torch"])
            if torch.cuda.is_available() and self.rng["cuda"]:
                torch.cuda.set_rng_state_all(self.rng["cuda"])
            generator = getattr(trainer.train_dataloader, "generator", None)
            if generator is not None and self.rng["loader"] is not None:
                generator.set_state(self.rng["loader"])
        print(f"Session: continue at epoch {trainer.current_epoch + 1}; "
              f"stop after epoch {self.stop_epoch}; scheduler horizon stays {trainer.max_epochs}", flush=True)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        generator = getattr(trainer.train_dataloader, "generator", None)
        checkpoint["native_session_rng"] = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state() if generator is not None else None,
        }

    def on_train_epoch_end(self, trainer, pl_module):
        self.completed = trainer.current_epoch + 1
        if self.completed >= self.stop_epoch:
            trainer.should_stop = True

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        checkpoints = [Path(cb.dirpath) / "last.ckpt" for cb in trainer.checkpoint_callbacks
                       if cb.dirpath and (Path(cb.dirpath) / "last.ckpt").is_file()]
        if not checkpoints:
            raise RuntimeError("Training stopped but no full last.ckpt was saved")
        status = {"completed_epochs": self.completed, "target_epochs": trainer.max_epochs,
                  "global_step": trainer.global_step, "checkpoint": str(checkpoints[-1].resolve()),
                  "training_complete": self.completed >= trainer.max_epochs}
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(self.status_path)
        print(json.dumps(status, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--session-epochs", type=int, default=4)
    parser.add_argument("--status", required=True)
    parser.add_argument("--resume-checkpoint")
    args, overrides = parser.parse_known_args()
    if args.session_epochs <= 0:
        parser.error("session-epochs must be positive")
    os.environ["MPLBACKEND"] = "Agg"
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import stable_pretraining as spt
    # Manager's native epoch-end last.ckpt is retained; its cache now lives on Drive.
    spt.set(cache_dir=os.environ["SPT_CACHE_DIR"], requeue_checkpoint=True,
            requeue_checkpoint_every_n_steps=0)
    original_manager = spt.Manager

    def session_manager(*positional, **kwargs):
        if positional:
            raise RuntimeError("Unexpected upstream Manager invocation")
        trainer = kwargs["trainer"]
        if args.resume_checkpoint:
            path = Path(args.resume_checkpoint).resolve(strict=True)
            # Only resume locally produced full Lightning checkpoints selected by the user.
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            from omegaconf import OmegaConf
            # Task values are in the upstream module config injected below, not Hydra's own config.
            cfg = OmegaConf.from_dotlist(overrides)
            validate_resume(checkpoint, cfg.output_model_name, trainer.max_epochs, cfg.seed,
                            legacy_recipe=load_legacy_recipe(path))
            if checkpoint["epoch"] + 1 >= trainer.max_epochs:
                raise ValueError("This checkpoint has completed the target epochs; run evaluation instead")
            if "native_session_rng" not in checkpoint:
                print("Legacy checkpoint: optimizer/scheduler/loop resume; exact RNG continuation unavailable.",
                      flush=True)
            del checkpoint
            kwargs["ckpt_path"] = str(path)
            kwargs["weights_only"] = False
            trainer.num_sanity_val_steps = 0
        elif kwargs.get("ckpt_path"):
            raise RuntimeError("Use explicit --resume-checkpoint to continue an existing run")
        from omegaconf import OmegaConf
        cfg = OmegaConf.from_dotlist(overrides)
        kwargs["module"].save_hyperparameters({"output_model_name": cfg.output_model_name,
                                              "trainer.max_epochs": trainer.max_epochs,
                                              "seed": cfg.seed})
        trainer.callbacks.append(SessionBoundary(args.session_epochs, args.status))
        return original_manager(**kwargs)

    spt.Manager = session_manager
    sys.argv = [str(repo / "train.py"), *overrides]
    runpy.run_path(str(repo / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
