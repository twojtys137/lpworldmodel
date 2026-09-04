import json

import numpy as np
import pytest
import torch

from experiments.dataset_replay import compare_replay, state_errors


class RecordedFixture:
    action_mean = torch.tensor([.1, -.2])
    action_std = torch.tensor([.2, .5])

    def __init__(self):
        self.states = torch.tensor([[100, 100, 200, 200, 0, 0, 0],
                                    [101, 102, 200, 200, 0, 1, 2],
                                    [104, 105, 201, 202, .1, 3, 3]], dtype=torch.float64)
        self.images = torch.arange(36, dtype=torch.float64).reshape(3, 3, 2, 2)/255
        self.physical_actions = torch.tensor([[.25, -.15], [.1, .2], [0, 0]])

    def get_seq_length(self, trajectory):
        return 3

    def __len__(self):
        return 1

    def get_frames(self, trajectory, frames):
        indices = list(frames)
        return ({"visual": self.images[indices]},
                (self.physical_actions[indices]-self.action_mean)/self.action_std,
                self.states[indices], {"shape": "T"})


class ReplayFixture:
    def __init__(self, dataset, change_recorded_goal=False):
        self.dataset = dataset
        self.change_recorded_goal = change_recorded_goal

    def update_env(self, info):
        assert info == {"shape": "T"}

    def close(self):
        self.closed = True

    def rollout(self, seed, initial, actions):
        np.testing.assert_allclose(initial, self.dataset.states[0].numpy())
        # Normalized actions must be undone exactly once; no extra factor100.
        np.testing.assert_allclose(actions, self.dataset.physical_actions[:2], atol=1e-7)
        self.actions = actions
        pixels = (self.dataset.images.numpy().transpose(0, 2, 3, 1)*255).round().astype(np.uint8)
        states = self.dataset.states.numpy().copy()
        if self.change_recorded_goal:
            pixels[-1] = np.clip(pixels[-1].astype(int)+10, 0, 255)
            states[-1, 2] += 3
            states[-1, 4] += 2*np.pi+.2
        return {"visual": pixels}, states


def test_dataset_replay_matches_recorded_reference_and_denormalizes_actions():
    dataset = RecordedFixture()
    result, preview = compare_replay(dataset, ReplayFixture(dataset), 0, 0, 2, 99)
    assert result["reset"]["pixel_mse_0_1"] == 0
    assert result["terminal"]["pixel_mse_0_1"] == 0
    assert result["terminal"]["block_position_error_px"] == 0
    assert preview.shape == (4, 4, 3) and preview.dtype == np.uint8
    json.dumps(result, allow_nan=False)


def test_deterministic_simulator_replay_can_still_disagree_with_dataset():
    dataset = RecordedFixture()
    simulator = ReplayFixture(dataset, change_recorded_goal=True)
    first, _ = compare_replay(dataset, simulator, 0, 0, 2, 99)
    second, _ = compare_replay(dataset, simulator, 0, 0, 2, 99)
    assert first == second  # Self-replay is perfect; dataset agreement is not.
    assert first["reset"]["pixel_mse_0_1"] == 0
    assert first["terminal"]["pixel_mse_0_1"] == pytest.approx((10/255)**2)
    assert first["terminal"]["block_position_error_px"] == 3
    assert first["terminal"]["block_angle_error_rad"] == pytest.approx(.2)


def test_push_t_components_separate_position_angle_and_velocity():
    first = np.array([0, 0, 0, 0, 0, 0, 0.])
    second = np.array([3, 4, 5, 12, 4*np.pi-.3, 8, 15])
    errors = state_errors(first, second)
    assert errors == pytest.approx({"agent_position_error_px": 5,
                                  "block_position_error_px": 13,
                                  "block_angle_error_rad": .3,
                                  "agent_velocity_error": 17})
    assert "agent_velocity_error" not in state_errors(first[:5], second[:5])
    with pytest.raises(ValueError):
        state_errors(first[:5], second)


def test_replay_rejects_training_normalization_and_too_short_segments():
    dataset = RecordedFixture()
    with pytest.raises(ValueError, match="short"):
        compare_replay(dataset, ReplayFixture(dataset), 0, 0, 3, 99)
    dataset.images = dataset.images*2-1
    with pytest.raises(ValueError, match="unnormalized"):
        compare_replay(dataset, ReplayFixture(dataset), 0, 0, 2, 99)


def test_cli_run_needs_only_config_and_persists_raw_pixel_results(tmp_path, monkeypatch):
    from argparse import Namespace
    import gym
    from omegaconf import OmegaConf
    from experiments import dataset_replay

    run_dir = tmp_path/"training"
    run_dir.mkdir()
    OmegaConf.save(OmegaConf.create({"num_hist": 3, "num_pred": 1, "frameskip": 5,
                    "env": {"name": "pusht", "args": [], "kwargs": {}, "dataset": {}}}),
                   run_dir/"hydra.yaml")
    dataset = RecordedFixture()
    simulator = ReplayFixture(dataset)
    calls = []
    def load_dataset(config, **kwargs):
        calls.append(kwargs)
        return {}, {"valid": dataset}
    monkeypatch.setattr(dataset_replay.hydra.utils, "call", load_dataset)
    monkeypatch.setattr(gym, "make", lambda *args, **kwargs: simulator)
    dataset_replay.run(Namespace(run_dir=str(run_dir), output=str(tmp_path/"replay"),
        samples=1, raw_steps=2, seed=99, split="valid", data_path=None))
    result = json.loads((tmp_path/"replay/results.json").read_text())
    assert result["complete"] and result["metadata"]["checkpoint_required"] is False
    assert calls[0]["transform"] is None
    assert result["results"][0]["terminal"]["pixel_mse_0_1"] == 0
    assert (tmp_path/"replay"/result["results"][0]["preview"]).exists()
    assert simulator.closed
