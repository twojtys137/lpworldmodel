import ast
import json
import sys
import time

import numpy as np
import pytest
import torch

from experiments.budget import charged, run_command, save
from experiments.probe_core import compression_rows, fit_bases, latent_rollout, locality_probe, paired_bootstrap
from experiments.theory_checks import check_locality, check_observables, check_order
from models.infojepa_modules import SparseGeneratorPredictor, Link
from planning.objectives import create_objective_fn


class FixtureTrajectories:
    action_dim = 2
    action_mean = torch.tensor([.1, -.2])
    action_std = torch.tensor([.7, .8])

    def __len__(self):
        return 6

    def get_seq_length(self, index):
        return 12

    def get_frames(self, index, frames):
        generator = torch.Generator().manual_seed(index+1)
        images = torch.randn(12, 3, 3, 3, generator=generator)
        actions = torch.randn(12, 2, generator=generator)
        states = torch.arange(24).reshape(12, 2).float()
        frames = list(frames)
        return {"visual": images[frames]}, actions[frames], states[frames], {}


def fixture_dataset(**kwargs):
    return {}, {"valid": FixtureTrajectories()}


def test_synthetic_statements_and_serialization():
    result = {"observable": check_observables(), "locality": check_locality(), "order": check_order()}
    json.dumps(result)
    assert result["observable"]["observable_rank"] == 4
    assert result["observable"]["pass"]
    assert all(row["pass"] for key in ("locality", "order") for row in result[key])
    # Holding distance fixed, enlarging the chain leaves the local response unchanged.
    assert result["locality"][0]["goal_error"] == pytest.approx(result["locality"][3]["goal_error"])


def test_goal_basis_recovers_cost_directions_ignored_by_state_pca():
    rng = np.random.default_rng(10)
    states = rng.normal(size=(100, 4)) * [50, 2, .1, .01]
    calibration = np.zeros((100, 4))
    calibration[:, 3] = rng.normal(size=100)
    bases = fit_bases(states, calibration)
    heldout = np.zeros((4, 10, 3, 4))
    heldout[..., 3] = rng.normal(size=(4, 10, 3))
    rows = compression_rows(heldout, bases, [1])
    goal_rows = [r for r in rows if r["basis"] == "goal_residual" and r["rank"] == 1]
    assert max(r["relative_cost_mae"] for r in goal_rows) < 1e-12
    assert all(r["certificate_pass"] for r in rows)
    assert max(r["finite_set_regret"] for r in rows if r["rank"] == 4) < 1e-12


def test_projected_objective_preserves_full_rank_and_scale(tmp_path):
    rng = np.random.default_rng(8)
    v, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    path = tmp_path/"bases.npz"
    np.savez(path, goal_residual=v)
    torch.manual_seed(2)
    pred = {"visual": torch.randn(3, 4, 2, 6), "proprio": torch.randn(3, 4, 2)}
    target = {"visual": torch.randn(3, 1, 2, 6), "proprio": torch.randn(3, 1, 2)}
    full = create_objective_fn(.3, 2)(pred, target)
    projected = create_objective_fn(.3, 2, basis_path=path, rank=6)(pred, target)
    torch.testing.assert_close(full, projected)
    rank2 = create_objective_fn(0, 2, basis_path=path, rank=2)(pred, target)
    difference = pred["visual"][:, -1:]-target["visual"]
    expected = (difference @ torch.tensor(v[:, :2], dtype=difference.dtype)).square().sum(-1).mean((1, 2))/6
    torch.testing.assert_close(rank2, expected)
    assert torch.all(rank2 <= create_objective_fn(0, 2)(pred, target)+1e-6)
    for mode in ("all", "last"):
        torch.testing.assert_close(create_objective_fn(.3, 2, mode)(pred, target),
            create_objective_fn(.3, 2, mode, path, rank=6)(pred, target))


def test_locality_uses_values_with_stable_sparse_support():
    torch.manual_seed(9)
    model = SparseGeneratorPredictor(input_dim=8, num_patches=5, num_frames=2,
                                    num_laws=2, law_rank=3, law_topk=1, edge_topk=1).double()
    x, action = torch.randn(1, 2, 5, 8, dtype=torch.float64), torch.randn(1, 2, 8, dtype=torch.float64)
    result = locality_probe(model, x, action, epsilons=(1e-6,))
    assert result["supported"] and result["graph_density_with_self"] < 1
    assert result["rows"][0]["sources_with_support_switch"] == 0
    assert result["rows"][0]["stable_off_graph_energy_fraction"] < 1e-12
    assert model.record_graph is False


def test_latent_probe_matches_full_world_model_rollout():
    # Actual VWorldModel and predictor, including linked sparse states and history indexing.
    from models.visual_world_model import VWorldModel
    from models.infojepa_modules import Embedder
    from test_sparse_generator import TinyPatchEncoder
    for kind in ("identity", "reprelu"):
        torch.manual_seed(2)
        wm = VWorldModel(image_size=3, num_hist=3, num_pred=1, encoder=TinyPatchEncoder(),
                         proprio_encoder=torch.nn.Identity(), action_encoder=Embedder(in_chans=2, emb_dim=16),
                         decoder=None, predictor=SparseGeneratorPredictor(input_dim=16, num_frames=3,
                         num_patches=9, num_laws=2, law_rank=4, law_topk=1, edge_topk=2),
                         action_conditioning="adaln", link=Link(kind=kind))
        wm.eval()
        observations = {"visual": torch.randn(2, 3, 3, 3, 3)}
        actions = torch.randn(2, 7, 2)
        with torch.no_grad():
            full, _ = wm.rollout(observations, actions)
            history = wm.encode_obs_linked(observations)["visual"]
            probe = latent_rollout(wm, history, actions)
        torch.testing.assert_close(probe, full["visual"])


def test_budget_blocks_overspend_and_keeps_crash_reservation(tmp_path):
    path = tmp_path/"ledger.json"
    save(path, {"total_cu": 10, "runs": [{"label": "crashed", "reserved_cu": 8, "status": "running"}]})
    with pytest.raises(ValueError, match="Insufficient"):
        run_command(path, 10, 1, 3, "next", [sys.executable, "-c", "raise Exception('must not run')"])
    assert charged(json.loads(path.read_text())) == 8


def test_budget_times_out_and_does_not_rerun_completed_command(tmp_path):
    path = tmp_path/"ledger.json"
    began = time.monotonic()
    assert run_command(path, 10, 3600, .3, "timeout", [sys.executable, "-c", "import time; time.sleep(10)"]) == 124
    assert time.monotonic()-began < 4
    marker = tmp_path/"marker"
    command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('once')", str(marker)]
    assert run_command(path, 10, 3600, 2, "ok", command) == 0
    marker.write_text("preserved")
    assert run_command(path, 10, 3600, 2, "ok", command) == 0
    assert marker.read_text() == "preserved"


def test_paired_statistics_use_goal_pairs():
    result = paired_bootstrap([0, 1, 0, 1], [0, 1, 0, 1])
    assert result["delta_b_minus_a"] == 0 and result["ci95"] == [0, 0]


def test_full_frozen_probe_loads_checkpoint_and_writes_results(tmp_path):
    from argparse import Namespace
    from omegaconf import OmegaConf
    from test_sparse_generator import TinyPatchEncoder
    from experiments.frozen_probe import run
    from models.infojepa_modules import Embedder
    run_dir = tmp_path/"fixture"
    (run_dir/"checkpoints").mkdir(parents=True)
    torch.save({"epoch": 1, "encoder": TinyPatchEncoder(),
                "predictor": SparseGeneratorPredictor(input_dim=16, num_frames=2, num_patches=9,
                    num_laws=2, law_rank=4, law_topk=1, edge_topk=2),
                "action_encoder": Embedder(in_chans=2, emb_dim=16),
                "proprio_encoder": torch.nn.Identity(), "link": Link("identity")},
               run_dir/"checkpoints/model_latest.pth")
    OmegaConf.save(OmegaConf.create({"num_hist": 2, "num_pred": 1, "frameskip": 1,
                    "num_action_repeat": 1, "num_proprio_repeat": 1,
                    "has_decoder": False, "proprio_emb_dim": 2, "action_emb_dim": 16,
                    "concat_dim": 0, "action_conditioning": "adaln",
                    "model": {"_target_": "models.visual_world_model.VWorldModel",
                              "image_size": 3, "num_hist": 2, "num_pred": 1},
                    "env": {"dataset": {"_target_": "test_theory_experiments.fixture_dataset"}}}),
                   run_dir/"hydra.yaml")
    args = Namespace(run_dir=str(run_dir), output=str(tmp_path/"probe"), epoch="latest",
                     seed=7, samples=4, candidates=4, horizon=2, ranks=[4, 8],
                     action_sigma=.5, locality_samples=1, simulator_samples=0, require_cuda=False)
    run(args)
    result = json.loads((tmp_path/"probe/results.json").read_text())
    assert result["complete"] and result["all_finite_candidate_certificates_pass"]
    assert len(result["rollout"]) == 4
    assert result["locality"][0]["supported"]
    samples = json.loads((tmp_path/"probe/samples.json").read_text())
    assert {r["trajectory"] for r in samples[:2]}.isdisjoint(r["trajectory"] for r in samples[2:])


def test_goal_sampling_excludes_calibration_and_preserves_action_alignment():
    from plan import PlanWorkspace
    workspace = object.__new__(PlanWorkspace)
    workspace.cfg_dict = {"excluded_traj_ids": [0, 1, 2, 3, 4]}
    workspace.dset = FixtureTrajectories()
    workspace.n_evals, workspace.frameskip, workspace.goal_H = 3, 2, 2
    observations, states, actions, _ = workspace.sample_traj_segment_from_dset(5)
    assert all(r["trajectory"] == 5 for r in workspace.target_samples)
    for sample, act in zip(workspace.target_samples, actions):
        expected = workspace.dset.get_frames(5, range(sample["offset"], sample["offset"]+5))[1][:4]
        torch.testing.assert_close(act, expected)


def test_zero_action_control_uses_physical_zero_and_full_mpc_budget():
    from plan import PlanWorkspace
    workspace = object.__new__(PlanWorkspace)
    workspace.evaluation_mode = "zero_action"
    workspace.goal_H, workspace.n_evals, workspace.action_dim, workspace.frameskip = 5, 2, 4, 2
    workspace.cfg_dict = {"planner": {"max_iter": 3}}
    workspace.dset, workspace.device = FixtureTrajectories(), "cpu"
    workspace._baseline_lengths = lambda actions: np.full(len(actions), np.inf)
    workspace._evaluate_and_log = lambda actions, *_: actions
    result = workspace.perform_planning()
    assert result.shape == (2, 15, 4)
    physical = result.reshape(2, 30, 2)*workspace.dset.action_std + workspace.dset.action_mean
    torch.testing.assert_close(physical, torch.zeros_like(physical))


def test_cached_chunked_cem_preserves_selected_actions_and_encodes_once():
    from types import SimpleNamespace
    from test_sparse_generator import TinyPatchEncoder
    from models.visual_world_model import VWorldModel
    from models.infojepa_modules import Embedder
    from planning.cem import CEMPlanner
    torch.manual_seed(21)
    predictor = SparseGeneratorPredictor(input_dim=16, num_frames=2, num_patches=9,
        num_laws=2, law_rank=4, law_topk=1, edge_topk=2)
    with torch.no_grad():
        predictor.law_up.mul_(500)
    wm = VWorldModel(image_size=3, num_hist=2, num_pred=1, encoder=TinyPatchEncoder(),
        proprio_encoder=torch.nn.Identity(), action_encoder=Embedder(in_chans=2, emb_dim=16),
        decoder=None, predictor=predictor, action_conditioning="adaln", link=Link("reprelu"))
    wm.eval()
    observations = {"visual": torch.randn(2, 1, 3, 3, 3)}
    goals = {"visual": torch.randn(2, 1, 3, 3, 3)}
    from unittest.mock import patch
    results, counts = [], []
    with patch.object(wm.encoder, "forward", wraps=wm.encoder.forward) as encoder_call:
        for cache, batch in ((False, 12), (True, 5)):
            planner = CEMPlanner(horizon=2, topk=4, num_samples=12, var_scale=1,
                opt_steps=2, eval_every=2, wm=wm, action_dim=2,
                objective_fn=create_objective_fn(0, 2),
                preprocessor=SimpleNamespace(transform_obs=lambda value: value),
                evaluator=None, wandb_run=SimpleNamespace(log=lambda *_: None), log_filename=None,
                candidate_batch_size=batch, cache_initial_encoding=cache)
            encoder_call.reset_mock()
            torch.manual_seed(122)
            result, _ = planner.plan(observations, goals)
            results.append(result)
            counts.append(sum(len(call.args[0]) for call in encoder_call.call_args_list))
    torch.testing.assert_close(results[0], results[1], rtol=1e-5, atol=1e-6)
    assert counts == [50, 4]


def test_baseline_counts_earlier_success_on_same_mpc_boundaries():
    from plan import PlanWorkspace
    from types import SimpleNamespace
    workspace = object.__new__(PlanWorkspace)
    workspace.goal_H, workspace.n_evals, workspace.frameskip = 2, 2, 1
    workspace.data_preprocessor = SimpleNamespace(denormalize_actions=lambda value: value)
    workspace.eval_seed, workspace.state_0, workspace.state_g = [1, 2], None, None
    states = np.zeros((2, 7, 1))
    states[0, 2] = 1  # reaches at 2, leaves before endpoint 6
    states[1, 6] = 1
    workspace.env = SimpleNamespace(rollout=lambda *_: (None, states),
        eval_state=lambda _, current: {"success": current[:, 0] == 1})
    lengths = workspace._baseline_lengths(torch.zeros(2, 6, 2))
    np.testing.assert_equal(lengths, [2, 6])


def test_colab_cells_are_valid_python():
    from pathlib import Path
    notebook = Path(__file__).parents[1]/"notebooks/theory_experiments_colab.ipynb"
    if not notebook.exists():
        pytest.skip("Notebook not generated yet")
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
