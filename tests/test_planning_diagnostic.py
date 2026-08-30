import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts" / "diagnose_planning_colab.sh"
COLLECTOR = ROOT / "scripts" / "collect_planning_diagnostic.py"
PLAN_SCRIPT = ROOT / "scripts" / "plan.sh"
PLAN_PY = ROOT / "plan.py"
NOTEBOOK = ROOT / "notebooks" / "sparse_generator_colab.ipynb"


def register(run_dir, kind, model, alpha=None):
    args = [
        sys.executable,
        str(COLLECTOR),
        "register",
        "--run-dir",
        str(run_dir),
        "--kind",
        kind,
        "--model",
        model,
        "--train-seed",
        "0",
        "--plan-seed",
        "99",
        "--n-evals",
        "10",
        "--goal-h",
        "5",
        "--max-iter",
        "3",
    ]
    if alpha is not None:
        args.extend(["--alpha", str(alpha)])
    subprocess.run(args, check=True)


def test_diagnostic_dry_run_has_oracle_and_paired_alpha_matrix(tmp_path):
    env = os.environ.copy()
    env.update({"RUN": "0", "CKPT_BASE": str(tmp_path / "drive")})
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("[dry-run][oracle]") == 1
    assert result.stdout.count("[dry-run][plan]") == 8
    for model in ("dense_dense", "dense_sparse", "sparse_dense", "sparse_sparse"):
        assert f"model={model} alpha=0" in result.stdout
        assert f"model={model} alpha=1" in result.stdout
    assert "paired_targets=oracle" in result.stdout


def test_collector_enforces_oracle_gate_and_writes_summary(tmp_path):
    root = tmp_path / "diagnostic"
    oracle = root / "oracle"
    planner = root / "dense_sparse_alpha1"
    register(oracle, "oracle", "dense_dense")
    register(planner, "plan", "dense_sparse", 1)
    (oracle / "logs.json").write_text(
        json.dumps(
            {
                "oracle_eval/success_rate": 1.0,
                "oracle_eval/mean_state_dist": 0.0,
                "oracle_eval/successes": [1] * 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (planner / "logs.json").write_text(
        json.dumps(
            {
                "mpc/success_rate": 0.0,
                "mpc/mean_state_dist": 100.0,
                "step": 1,
            }
        )
        + "\n"
        + json.dumps(
            {
                "mpc/success_rate": 0.1,
                "mpc/mean_state_dist": 80.0,
                "step": 2,
            }
        )
        + "\n"
        + json.dumps(
            {
                "final_eval/success_rate": 0.1,
                "final_eval/mean_state_dist": 80.0,
                "final_eval/successes": [1] + [0] * 9,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    gate = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            "check-oracle",
            "--logs",
            str(oracle / "logs.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert gate.returncode == 0
    assert "success_rate=1.0" in gate.stdout

    output = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            "collect",
            "--root",
            str(root),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    rows = json.loads((output / "diagnostic_results.json").read_text())
    assert len(rows) == 2
    assert rows[0]["kind"] == "oracle"
    assert rows[1]["success_rate"] == 0.1
    assert rows[1]["mpc_first_state_dist"] == 100.0
    assert rows[1]["mpc_last_state_dist"] == 80.0
    assert "**PASS:**" in (output / "diagnostic_results.md").read_text()


def test_oracle_gate_blocks_inconsistent_protocol(tmp_path):
    logs = tmp_path / "logs.json"
    logs.write_text(
        json.dumps({"oracle_eval/success_rate": 0.9}) + "\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            "check-oracle",
            "--logs",
            str(logs),
            "--min-success",
            "0.99",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "Do not run MPC comparisons" in result.stderr


def test_planning_runtime_exposes_true_gt_replay_and_paired_overrides():
    plan_source = PLAN_PY.read_text(encoding="utf-8")
    shell_source = PLAN_SCRIPT.read_text(encoding="utf-8")
    notebook = NOTEBOOK.read_text(encoding="utf-8")

    assert 'self.evaluation_mode == "gt_replay"' in plan_source
    assert "self.gt_actions.to(self.device)" in plan_source
    assert 'prefix="oracle_eval"' in plan_source
    assert '"env_info": self.env_info' in plan_source
    assert '"eval_seed": self.eval_seed' in plan_source
    assert 'EXTRA+=("objective.alpha=${OBJECTIVE_ALPHA}")' in shell_source
    assert 'EXTRA+=("goal_file_path=${GOAL_FILE_PATH}")' in shell_source
    assert "diagnose_planning_colab.sh" in notebook
