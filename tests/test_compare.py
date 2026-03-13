"""Tests for the compare tool."""

import json
from io import StringIO

import torch

from model_clinic._tools.compare import compare_models, print_compare


def _make_healthy_sd():
    """A clean state dict."""
    torch.manual_seed(42)
    return {
        "embed_tokens.weight": torch.randn(100, 64),
        "layers.0.attention.q_proj.weight": torch.randn(64, 64),
        "layers.0.mlp.gate_proj.weight": torch.randn(256, 64),
        "layers.0.mlp.down_proj.weight": torch.randn(64, 256),
        "layers.0.attn_norm.weight": torch.ones(64),
        "final_norm.weight": torch.ones(64),
        "lm_head.weight": torch.randn(100, 64),
    }


def _make_sick_sd():
    """A state dict with issues."""
    torch.manual_seed(99)
    sd = {
        "embed_tokens.weight": torch.randn(100, 64),
        "layers.0.attention.q_proj.weight": torch.randn(64, 64),
        "layers.0.mlp.gate_proj.weight": torch.randn(256, 64),
        "layers.0.mlp.down_proj.weight": torch.randn(64, 256),
        "layers.0.attn_norm.weight": torch.ones(64),
        "final_norm.weight": torch.full((64,), 2.5),  # norm drift
        "lm_head.weight": torch.randn(100, 64),
    }
    # Dead neurons
    sd["layers.0.mlp.down_proj.weight"][0] = 0
    sd["layers.0.mlp.down_proj.weight"][1] = 0
    return sd


class TestCompareModels:

    def test_identical_models_no_delta(self):
        sd = _make_healthy_sd()
        result = compare_models(sd, {}, sd, {})
        assert result["health_delta"] == 0
        assert result["param_changes"]["modified_count"] == 0
        assert result["findings_delta"]["resolved_total"] == 0
        assert result["findings_delta"]["new_total"] == 0

    def test_sick_to_healthy_shows_resolved(self):
        sick = _make_sick_sd()
        healthy = _make_healthy_sd()
        result = compare_models(sick, {}, healthy, {})
        # Sick model has findings, healthy doesn't
        assert result["findings_delta"]["resolved_total"] > 0
        assert result["findings_delta"]["new_total"] == 0

    def test_healthy_to_sick_shows_new(self):
        sick = _make_sick_sd()
        healthy = _make_healthy_sd()
        result = compare_models(healthy, {}, sick, {})
        assert result["findings_delta"]["new_total"] > 0

    def test_health_delta_sign(self):
        sick = _make_sick_sd()
        healthy = _make_healthy_sd()
        result = compare_models(sick, {}, healthy, {})
        assert result["health_delta"] >= 0  # healthy >= sick

    def test_param_changes_detected(self):
        a = _make_healthy_sd()
        b = _make_healthy_sd()
        # Modify one tensor
        b["final_norm.weight"] = torch.full((64,), 2.0)
        result = compare_models(a, {}, b, {})
        assert result["param_changes"]["modified_count"] >= 1
        assert result["param_changes"]["max_delta_norm"] > 0

    def test_added_removed_params(self):
        a = _make_healthy_sd()
        b = _make_healthy_sd()
        b["new_param"] = torch.randn(32)
        del b["lm_head.weight"]
        result = compare_models(a, {}, b, {})
        assert "lm_head.weight" in result["param_changes"]["only_before"]
        assert "new_param" in result["param_changes"]["only_after"]

    def test_result_structure(self):
        sd = _make_healthy_sd()
        result = compare_models(sd, {}, sd, {})
        assert "health_before" in result
        assert "health_after" in result
        assert "health_delta" in result
        assert "findings_delta" in result
        assert "param_changes" in result
        assert "overall" in result["health_before"]
        assert "grade" in result["health_before"]
        assert "categories" in result["health_before"]

    def test_json_serializable(self):
        sick = _make_sick_sd()
        healthy = _make_healthy_sd()
        result = compare_models(sick, {}, healthy, {})
        # Should not raise
        output = json.dumps(result, default=str)
        parsed = json.loads(output)
        assert "health_delta" in parsed


class TestPrintCompare:

    def test_prints_without_error(self):
        sd = _make_healthy_sd()
        result = compare_models(sd, {}, sd, {})
        buf = StringIO()
        print_compare(result, "a.pt", "b.pt", file=buf)
        output = buf.getvalue()
        assert "a.pt" in output
        assert "b.pt" in output

    def test_prints_all_sections(self):
        sick = _make_sick_sd()
        healthy = _make_healthy_sd()
        result = compare_models(sick, {}, healthy, {})
        buf = StringIO()
        print_compare(result, "sick.pt", "healthy.pt", file=buf)
        output = buf.getvalue()
        assert "Health Score" in output
        assert "Category Breakdown" in output
        assert "Findings Delta" in output
        assert "Parameter Changes" in output

    def test_shows_categories(self):
        sd = _make_healthy_sd()
        result = compare_models(sd, {}, sd, {})
        buf = StringIO()
        print_compare(result, "a.pt", "b.pt", file=buf)
        output = buf.getvalue()
        for cat in ["weights", "stability", "output", "activations"]:
            assert cat in output
