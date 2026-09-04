"""Checkpoint continuation must reproduce the next stochastic optimizer step."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from train import Trainer, capture_rng_state, restore_rng_state, training_epoch_range
from utils import load_trusted_checkpoint, seed


class TinyEncoder(torch.nn.Linear):
    latent_ndim = 1
    num_patches = 1
    emb_dim = 4

    def __init__(self):
        super().__init__(4, 4)


class TinyEmbedder(torch.nn.Linear):
    emb_dim = 4

    def __init__(self):
        super().__init__(4, 4)


class TinyWorldModel(torch.nn.Module):
    def __init__(self, encoder, predictor, action_encoder, proprio_encoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.proprio_encoder = proprio_encoder
        self.dropout = torch.nn.Dropout(.25)

    def forward(self, x):
        return self.predictor(self.dropout(
            self.encoder(x) + self.action_encoder(x) + self.proprio_encoder(x)
        ))


class CPUAccelerator:
    num_processes = 1
    process_index = 0
    is_main_process = True
    scaler = None

    def prepare(self, *values):
        return values[0] if len(values) == 1 else values

    def wait_for_everyone(self):
        pass

    def unwrap_model(self, value):
        return value


def tiny_trainer(path, monkeypatch, *, allow_partial=False):
    def instantiate(cfg, **kwargs):
        kind = cfg["_target_"]
        if kind == "tiny.encoder":
            return TinyEncoder()
        if kind == "tiny.embedder":
            return TinyEmbedder()
        if kind == "tiny.predictor":
            return torch.nn.Linear(4, 4)
        if kind == "tiny.model":
            return TinyWorldModel(**kwargs)
        raise AssertionError(kind)

    monkeypatch.setattr("train.hydra.utils.instantiate", instantiate)
    trainer = object.__new__(Trainer)
    trainer.cfg = OmegaConf.create({
        "saved_folder": str(path), "img_size": 16, "concat_dim": 0,
        "has_predictor": True, "has_decoder": False, "num_hist": 1,
        "action_conditioning": "adaln", "action_emb_dim": 4, "proprio_emb_dim": 4,
        "num_action_repeat": 1, "num_proprio_repeat": 1,
        "encoder": {"_target_": "tiny.encoder"},
        "action_encoder": {"_target_": "tiny.embedder"},
        "proprio_encoder": {"_target_": "tiny.embedder"},
        "predictor": {"_target_": "tiny.predictor"},
        "model": {"_target_": "tiny.model"},
        "training": {"encoder_lr": .01, "predictor_lr": .01,
                     "action_encoder_lr": .01, "allow_partial_resume": allow_partial},
    })
    trainer.device = torch.device("cpu")
    trainer.accelerator = CPUAccelerator()
    trainer.wandb_run = SimpleNamespace(watch=lambda *args: None)
    trainer.datasets = {"train": SimpleNamespace(action_dim=4, proprio_dim=4)}
    trainer.train_encoder = trainer.train_predictor = True
    trainer.train_decoder = False
    trainer.epoch = 0
    trainer._keys_to_save = ["epoch", "encoder", "predictor", "decoder",
                             "action_encoder", "proprio_encoder", "link"]
    for name in trainer._keys_to_save[1:]:
        setattr(trainer, name, None)
    trainer._resume_checkpoint = None
    trainer.resume_metadata = {"training_state_complete": True, "partial_resume_reasons": []}
    trainer.init_models()
    trainer.init_optimizers()
    trainer.restore_training_state()
    return trainer


def stochastic_update(trainer):
    x = torch.randn(8, 4) + np.random.normal() + random.random()
    for name in trainer.optimizer_names():
        getattr(trainer, name).zero_grad()
    loss = trainer.model(x).square().mean()
    loss.backward()
    for name in trainer.optimizer_names():
        getattr(trainer, name).step()
    trainer.epoch += 1
    return loss.detach().clone()


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_resume_preserves_embeddings_optimizer_binding_and_next_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed(13)
    uninterrupted = tiny_trainer(tmp_path, monkeypatch)
    for _ in range(3):
        stochastic_update(uninterrupted)
    action_weights = uninterrupted.action_encoder.weight.detach().clone()
    proprio_weights = uninterrupted.proprio_encoder.weight.detach().clone()
    uninterrupted.save_ckpt()
    checkpoint = load_trusted_checkpoint(tmp_path / "checkpoints/model_latest.pth")
    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["epoch"] == 3
    assert "action_encoder_optimizer" in checkpoint["optimizer_state_dicts"]
    assert "decoder" not in checkpoint  # Planning loader calls .to on saved modules.
    assert "predictor_optimizer" not in checkpoint  # No stale optimizer objects.
    expected_loss = stochastic_update(uninterrupted)

    seed(987)  # Resume must restore all RNGs after initialization consumes them.
    resumed = tiny_trainer(tmp_path, monkeypatch)
    assert resumed.epoch == 3
    torch.testing.assert_close(resumed.action_encoder.weight, action_weights, rtol=0, atol=0)
    torch.testing.assert_close(resumed.proprio_encoder.weight, proprio_weights, rtol=0, atol=0)
    active_ids = {id(p) for p in resumed.model.parameters()}
    for name in resumed.optimizer_names():
        optimizer = getattr(resumed, name)
        assert optimizer.state
        assert all(id(p) in active_ids for group in optimizer.param_groups for p in group["params"])
    actual_loss = stochastic_update(resumed)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    assert_nested_equal(resumed.model.state_dict(), uninterrupted.model.state_dict())
    for name in resumed.optimizer_names():
        assert_nested_equal(getattr(resumed, name).state_dict(), getattr(uninterrupted, name).state_dict())


def test_legacy_checkpoint_requires_explicit_partial_resume(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    seed(13)
    original = tiny_trainer(tmp_path, monkeypatch)
    stochastic_update(original)
    original.save_ckpt()
    checkpoint_path = tmp_path / "checkpoints/model_latest.pth"
    checkpoint = load_trusted_checkpoint(checkpoint_path)
    for name in ("checkpoint_version", "optimizer_state_dicts", "rng_states", "resume_metadata"):
        checkpoint.pop(name)
    checkpoint["encoder_optimizer"] = original.encoder_optimizer
    checkpoint["predictor_optimizer"] = original.predictor_optimizer
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match="allow_partial_resume=true"):
        tiny_trainer(tmp_path, monkeypatch)
    resumed = tiny_trainer(tmp_path, monkeypatch, allow_partial=True)
    assert "PARTIAL TRAINING RESUME" in caplog.text
    assert not resumed.resume_metadata["training_state_complete"]
    assert any("action_encoder_optimizer" in reason for reason in resumed.resume_metadata["partial_resume_reasons"])
    assert resumed.encoder_optimizer.state  # Recover legacy moments when present.
    assert not resumed.action_encoder_optimizer.state  # Explicitly missing in old format.
    resumed.save_ckpt()
    assert not load_trusted_checkpoint(checkpoint_path)["resume_metadata"]["training_state_complete"]


def test_epochs_total_mode_and_legacy_additional_mode():
    assert list(training_epoch_range(2, 10, "total")) == list(range(3, 11))
    assert list(training_epoch_range(2, 10)) == list(range(3, 13))
    assert list(training_epoch_range(10, 10, "total")) == []
    with pytest.raises(ValueError, match="epochs_mode"):
        training_epoch_range(0, 10, "ambiguous")


def test_rng_round_trip():
    seed(59)
    state = capture_rng_state()
    expected = (random.random(), np.random.random(), torch.randn(3))
    restore_rng_state(state)
    actual = (random.random(), np.random.random(), torch.randn(3))
    assert_nested_equal(expected, actual)
