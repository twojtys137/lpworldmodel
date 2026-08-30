import json

import pytest
import torch

from metric_logging import append_metrics_jsonl, jsonable_metric, metric_scalar


class FakeTensor:
    def __init__(self, value):
        self.value = value
        self.detached = False
        self.on_cpu = False

    def detach(self):
        self.detached = True
        return self

    def cpu(self):
        self.on_cpu = True
        return self

    def numel(self):
        return 1 if not isinstance(self.value, list) else len(self.value)

    def item(self):
        return self.value

    def tolist(self):
        return self.value


def test_converts_nested_tensor_like_metrics_to_json_types(tmp_path):
    scalar = FakeTensor(0.125)
    vector = FakeTensor([0.25, 0.5])
    payload = {"loss": scalar, "diagnostics": {"usage": vector}}

    converted = append_metrics_jsonl(tmp_path / "metrics.jsonl", payload)

    assert converted == {"loss": 0.125, "diagnostics": {"usage": [0.25, 0.5]}}
    assert scalar.detached and scalar.on_cpu
    assert vector.detached and vector.on_cpu
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == converted


def test_metric_scalar_accepts_tensor_scalar_and_rejects_vector():
    assert metric_scalar(FakeTensor(3.5)) == 3.5
    with pytest.raises(TypeError, match="Expected a scalar metric"):
        metric_scalar(FakeTensor([1.0, 2.0]))


def test_real_torch_tensor_is_detached_and_persisted(tmp_path):
    value = torch.tensor(0.375, requires_grad=True)

    converted = append_metrics_jsonl(tmp_path / "metrics.jsonl", {"rollout": value})

    assert converted == {"rollout": pytest.approx(0.375)}
    assert json.loads((tmp_path / "metrics.jsonl").read_text()) == {"rollout": 0.375}


def test_unknown_metric_type_fails_with_actionable_message():
    with pytest.raises(TypeError, match="object is not JSON serializable"):
        jsonable_metric(object())
