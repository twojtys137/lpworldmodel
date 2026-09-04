"""Prevent positional diversity from concealing missing scene information."""
import json

import numpy as np
import pytest
import torch

from experiments.probe_core import dynamics_diagnostics, scene_variance_diagnostics


@pytest.mark.parametrize("patches", [1, 64])
def test_image_independent_code_has_no_scene_variance(patches):
    rng = np.random.default_rng(14)
    template = rng.normal(size=(1, 1, patches, 12))
    z = np.broadcast_to(template, (8, 4, patches, 12)).copy()
    result = scene_variance_diagnostics(z)
    assert result["across_trajectory_variance_fixed_time_patch_channel"] == pytest.approx(0, abs=1e-25)
    assert result["within_trajectory_temporal_variance_fixed_patch_channel"] == pytest.approx(0, abs=1e-25)
    if patches > 1:
        assert result["pooled_channel_variance"] > .5
        assert result["spatial_template_fraction_of_pooled_variance"] == pytest.approx(1)
        assert result["scene_to_pooled_variance_ratio"] == pytest.approx(0, abs=1e-25)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("patches", [1, 9])
def test_scene_variance_ignores_static_position_offsets(patches):
    rng = np.random.default_rng(29)
    z = rng.normal(size=(8, 5, patches, 6))
    huge_position_offsets = rng.normal(size=(1, 1, patches, 6)) * 100
    original = scene_variance_diagnostics(z)
    shifted = scene_variance_diagnostics(z + huge_position_offsets)
    for key in ("across_trajectory_variance_fixed_time_patch_channel",
                "within_trajectory_temporal_variance_fixed_patch_channel"):
        assert shifted[key] == pytest.approx(original[key])
    assert original["across_trajectory_variance_fixed_time_patch_channel"] > .5


def test_time_pattern_shared_by_every_scene_does_not_count_as_scene_variation():
    z = np.broadcast_to(np.arange(5)[None, :, None, None], (8, 5, 4, 2)).copy()
    result = scene_variance_diagnostics(z)
    assert result["across_trajectory_variance_fixed_time_patch_channel"] == 0
    assert result["within_trajectory_temporal_variance_fixed_patch_channel"] == 2


def test_single_scene_or_frame_is_not_enough_for_its_variance_estimate():
    result = scene_variance_diagnostics(np.zeros((1, 1, 2, 3)))
    assert result["across_trajectory_variance_fixed_time_patch_channel"] is None
    assert result["within_trajectory_temporal_variance_fixed_patch_channel"] is None
    assert result["scene_to_pooled_variance_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_persistence_and_action_response_use_observed_change():
    z = torch.arange(4, dtype=torch.float32)[:, None, None].expand(4, 3, 2)
    result = dynamics_diagnostics(z, z[-1], z[1], history_length=2)
    assert result["observed_change_rms"] == 2
    assert result["model_to_persistence_mse_ratio"] == 0
    assert result["model_beats_persistence"] is True
    assert result["action_response_to_observed_change_ratio"] == 1
    # A persistence predictor is a tie, not an improvement.
    unchanged = dynamics_diagnostics(z, z[1], z[1], history_length=2)
    assert unchanged["model_to_persistence_mse_ratio"] == 1
    assert unchanged["model_beats_persistence"] is False
    assert unchanged["action_response_to_observed_change_ratio"] == 0


def test_zero_change_has_null_ratios_instead_of_apparent_perfect_model():
    z = torch.ones(4, 2, 3)
    result = dynamics_diagnostics(z, z[-1], z[-1], history_length=1)
    assert result["persistence_comparison_defined"] is False
    assert result["model_to_persistence_mse_ratio"] is None
    assert result["model_beats_persistence"] is None
    assert result["action_response_to_observed_change_ratio"] is None
    json.dumps(result, allow_nan=False)


def test_diagnostics_reject_invalid_or_nonfinite_inputs():
    for z in (np.zeros((2, 3)), np.zeros((0, 3, 2, 1)), np.full((2, 3, 2, 1), np.nan)):
        with pytest.raises(ValueError):
            scene_variance_diagnostics(z)
    with pytest.raises(ValueError):
        dynamics_diagnostics(torch.zeros(3, 2, 4), torch.zeros(2, 4), torch.zeros(2, 4), 3)
