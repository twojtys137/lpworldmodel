import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_lpwm_sparse_generator_colab.sh"
NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "sparse_generator_colab.ipynb"


def run_launcher(tmp_path, trainer_body):
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "pusht_noise").mkdir(parents=True)
    ckpt_base = tmp_path / "drive"
    trainer = tmp_path / "fake_trainer.sh"
    trainer.write_text("#!/bin/bash\nset -euo pipefail\n" + trainer_body, encoding="utf-8")
    trainer.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "RUN": "1",
            "PROFILE": "smoke",
            "STAGE": "train",
            "MODELS": "dense_dense",
            "SEEDS": "0",
            "DATASET_DIR": str(dataset_dir),
            "CKPT_BASE": str(ckpt_base),
            "TRAIN_PATCH": str(trainer),
            "WANDB_MODE": "online",
            "REQUIRE_WANDB_ONLINE": "1",
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    run_dir = ckpt_base / "outputs" / "fair_lpwm_v1_smoke_dense_dense_seed0"
    return result, run_dir


def test_launcher_persists_stdout_stderr_and_exit_code(tmp_path):
    result, run_dir = run_launcher(
        tmp_path,
        'echo "trainer stdout"\necho "trainer stderr" >&2\nexit 7\n',
    )

    assert result.returncode == 7
    launcher_log = (run_dir / "launcher.log").read_text(encoding="utf-8")
    assert "trainer stdout" in launcher_log
    assert "trainer stderr" in launcher_log
    assert "status=7" in launcher_log


def test_launcher_rejects_success_without_metrics_or_checkpoint(tmp_path):
    result, run_dir = run_launcher(tmp_path, 'echo "premature clean exit"\n')

    assert result.returncode == 4
    launcher_log = (run_dir / "launcher.log").read_text(encoding="utf-8")
    assert "premature clean exit" in launcher_log
    assert "metrics.jsonl is missing or empty" in launcher_log


def test_launcher_accepts_only_verified_training_artifacts(tmp_path):
    result, run_dir = run_launcher(
        tmp_path,
        "mkdir -p \"${CKPT_BASE}/outputs/${RUN_NAME}/checkpoints\"\n"
        "printf '{\"epoch\": 1}\\n' > \"${CKPT_BASE}/outputs/${RUN_NAME}/metrics.jsonl\"\n"
        "touch \"${CKPT_BASE}/outputs/${RUN_NAME}/checkpoints/model_latest.pth\"\n",
    )

    assert result.returncode == 0
    launcher_log = (run_dir / "launcher.log").read_text(encoding="utf-8")
    assert "verified metrics and checkpoint" in launcher_log


def test_colab_installs_planning_dependencies():
    notebook = NOTEBOOK.read_text(encoding="utf-8")

    assert "'submitit>=1.5,<2'" in notebook
    assert "'hydra-submitit-launcher>=1.2,<2'" in notebook
    assert "'pymunk==6.11.1'" in notebook
    assert "import hydra_plugins.hydra_submitit_launcher" in notebook
    assert "hasattr(pymunk.Space, 'add_collision_handler')" in notebook


def test_launcher_preflights_pymunk_collision_api():
    launcher = SCRIPT.read_text(encoding="utf-8")

    assert 'hasattr(space, "add_collision_handler")' in launcher
    assert "space.add_collision_handler(0, 0)" in launcher
    assert "install pymunk==6.11.1" in launcher
