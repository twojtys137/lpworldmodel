"""CPU integration check for Lightning session boundaries and optimizer/scheduler resume."""
import json
from pathlib import Path
import tempfile

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import DataLoader, TensorDataset

from lewm_session import SessionBoundary, validate_resume


class Probe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)
        self.save_hyperparameters({"output_model_name": "probe", "trainer.max_epochs": 3, "seed": 3072})

    def training_step(self, batch, batch_idx):
        x, y = batch
        return (self.layer(x) - y).square().mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def fit_probe(directory, session_epochs, checkpoint=None):
    torch.manual_seed(3072)
    model = Probe()
    x = torch.arange(16, dtype=torch.float32).reshape(8, 2) / 16
    loader = DataLoader(TensorDataset(x, x.sum(1, keepdim=True)), batch_size=4,
                        generator=torch.Generator().manual_seed(3072))
    boundary = SessionBoundary(session_epochs, directory / "progress.json")
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_epochs=3, logger=False,
                         enable_progress_bar=False, enable_model_summary=False,
                         callbacks=[boundary, ModelCheckpoint(dirpath=directory, save_last=True)],
                         num_sanity_val_steps=0)
    trainer.fit(model, train_dataloaders=loader, ckpt_path=checkpoint, weights_only=False)
    state = torch.load(directory / "last.ckpt", map_location="cpu", weights_only=False)
    validate_resume(state, "probe", 3, 3072)
    return state


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right, (left, right)


def main():
    with tempfile.TemporaryDirectory(prefix="lewm-resume-probe-") as temp:
        root = Path(temp)
        full = fit_probe(root / "full", 3)
        first = fit_probe(root / "first", 1)
        assert first["global_step"] == 2
        progress = json.loads((root / "first/progress.json").read_text())
        assert progress["completed_epochs"] == 1 and not progress["training_complete"]
        resumed = fit_probe(root / "resumed", 2, str(root / "first/last.ckpt"))
        for key in ("state_dict", "optimizer_states", "lr_schedulers", "global_step", "epoch"):
            assert_equal(full[key], resumed[key])
        assert resumed["global_step"] == 6
    print("PASS: 3 continuous epochs equal 1+2 resumed epochs for weights, AdamW state, scheduler and counters.")


if __name__ == "__main__":
    main()
