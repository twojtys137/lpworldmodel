"""Resume must reject model-only, incomplete-epoch and wrong-recipe checkpoints."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("lightning")
root = Path(__file__).parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


session = load_script("lewm_session")
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
