"""Resume must reject model-only, incomplete-epoch and wrong-recipe checkpoints."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")
root = Path(__file__).parents[1]
sys.path.insert(0, str(root / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load_script("lewm_session")
sys.modules["lewm_session"] = session
selection = load_script("select_lewm_checkpoint")
native = load_script("native_worldmodels")


def checkpoint():
    return {"state_dict": {}, "optimizer_states": [{}], "lr_schedulers": [{}],
            "loops": {"fit_loop": {"epoch_loop.batch_progress": {"is_last_batch": True}}},
            "epoch": 1, "global_step": 100,
            "hyper_parameters": {"output_model_name": "run", "trainer.max_epochs": 10, "seed": 3072}}


def test_model_only_and_mid_epoch_cannot_resume():
    with pytest.raises(ValueError, match="missing optimizer_states"):
        session.validate_resume({"state_dict": {}}, "run", 10, 3072)
    state = checkpoint()
    state["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"] = False
    with pytest.raises(ValueError, match="completed epoch"):
        session.validate_resume(state, "run", 10, 3072)


@pytest.mark.parametrize("key,value", [("output_model_name", "another"),
                                       ("trainer.max_epochs", 4), ("seed", 0)])
def test_resume_preserves_run_and_scheduler_horizon(key, value):
    state = checkpoint()
    session.validate_resume(state, "run", 10, 3072)
    state["hyper_parameters"][key] = value
    with pytest.raises(ValueError, match="recipe mismatch"):
        session.validate_resume(state, "run", 10, 3072)


def test_preserve_copies_full_checkpoint_without_deleting_source(tmp_path, monkeypatch):
    cache = tmp_path / "cache/runs/one/checkpoints"
    cache.mkdir(parents=True)
    source = cache / "last.ckpt"
    source.write_bytes(b"checkpoint-payload")
    (cache / "weights_epoch_1.pt").write_bytes(b"model-only")
    monkeypatch.setenv("SPT_CACHE_DIR", str(tmp_path / "cache"))
    output = tmp_path / "persistent"
    native.preserve_checkpoints(SimpleNamespace(output=str(output)))
    copies = list(output.rglob("*.ckpt"))
    assert len(copies) == 1
    assert source.read_bytes() == copies[0].read_bytes() == b"checkpoint-payload"
    assert not list(output.rglob("*.pt"))


def legacy_checkpoint():
    state = checkpoint()
    state["hyper_parameters"] = {"system.working_dir": "/content/lewm"}
    state["wandb"] = {"id": "run", "project": "lpwm-native"}
    return state


def test_legacy_requires_matching_original_config_and_embedded_run_id():
    state = legacy_checkpoint()
    recipe = {"output_model_name": "run", "trainer": {"max_epochs": 10}, "seed": 3072}
    session.validate_resume(state, "run", 10, 3072, legacy_recipe=recipe)
    with pytest.raises(ValueError, match="recipe mismatch"):
        session.validate_resume(state, "run", 10, 3072)
    with pytest.raises(ValueError, match="recipe mismatch"):
        session.validate_resume(state, "run", 10, 3072, legacy_recipe={**recipe, "seed": 0})
    state["hyper_parameters"]["seed"] = 0
    with pytest.raises(ValueError, match="recipe mismatch"):
        session.validate_resume(state, "run", 10, 3072, legacy_recipe=recipe)
    del state["hyper_parameters"]["seed"]
    state["wandb"]["id"] = "another"
    with pytest.raises(ValueError, match="W&B run ID"):
        session.validate_resume(state, "run", 10, 3072, legacy_recipe=recipe)


def test_select_legacy_checkpoint_preserves_bytes_and_binds_recipe(tmp_path):
    import torch
    config = tmp_path / "lewm/checkpoints/run/config.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("output_model_name: run\nseed: 3072\ntrainer:\n  max_epochs: 10\n")
    recovery = tmp_path / "lewm/recovered-checkpoints"
    recovery.mkdir()
    state = legacy_checkpoint()
    inventory = []
    for epoch in (4, 5):
        state.update(epoch=epoch, global_step=(epoch + 1) * 13933)
        path = recovery / f"epoch={epoch}.ckpt"
        torch.save(state, path)
        inventory.append({"checkpoint": str(path)})
    (recovery / "inventory.json").write_text(json.dumps(inventory))
    before = path.read_bytes()
    selected = selection.select_checkpoint(tmp_path, "run")
    assert selected["completed_epochs"] == 6
    assert selected["global_step"] == 83598
    assert path.read_bytes() == before
    recipe = session.load_legacy_recipe(path)
    session.validate_resume(state, "run", 10, 3072, legacy_recipe=recipe)
    path.write_bytes(before + b"changed")
    with pytest.raises(ValueError, match="does not match checkpoint"):
        session.load_legacy_recipe(path)
