import argparse
import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "collect_lpwm_benchmark.py"
SPEC = importlib.util.spec_from_file_location("collect_lpwm_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_collects_persisted_training_and_planning_metrics(tmp_path):
    ckpt_base = tmp_path / "drive"
    benchmark_dir = ckpt_base / "benchmarks" / "fair_test"
    run_name = "fair_test_screen_dense_sparse_seed0"
    planning_dir = ckpt_base / "planning" / run_name

    MODULE.register(
        argparse.Namespace(
            benchmark_dir=benchmark_dir,
            benchmark_id="fair_test",
            profile="screen",
            model="dense_sparse",
            architecture="patch_generator",
            state_sparse="false",
            law_sparse="true",
            run_name=run_name,
            seed=0,
            planning_dir=planning_dir,
        )
    )

    run_dir = ckpt_base / "outputs" / run_name
    checkpoint = run_dir / "checkpoints" / "model_latest.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (run_dir / "hydra.yaml").write_text(
        """
wandb_run_id: abc123
link:
  kind: identity
regularizer:
  target_p: 2
  num_projections: 256
predictor:
  _target_: models.infojepa_modules.SparseGeneratorPredictor
env:
  dataset:
    n_rollout: 50
training:
  epochs: 2
  batch_size: 16
""".strip(),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "epoch": 2,
                "val_z_loss": 0.02,
                "val_z_visual_err_rollout": 0.04,
                "num_parameters": 1234,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    planning_dir.mkdir(parents=True)
    (planning_dir / "logs.json").write_text(
        json.dumps(
            {
                "final_eval/success_rate": 0.6,
                "final_eval/successes": [1, 0, 1, 0, 1],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    MODULE.collect(
        argparse.Namespace(
            benchmark_dir=benchmark_dir,
            ckpt_base=ckpt_base,
            wandb_entity="entity",
            wandb_project="project",
            use_wandb=False,
        )
    )

    with (benchmark_dir / "benchmark_results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["training_complete"] == "True"
    assert rows[0]["planning_complete"] == "True"
    assert rows[0]["val_z_loss"] == "0.02"
    assert rows[0]["planning_success_rate"] == "0.6"
    assert rows[0]["planning_n_evals"] == "5"
    assert (benchmark_dir / "benchmark_aggregate.csv").exists()
    assert (benchmark_dir / "benchmark_results.md").exists()
