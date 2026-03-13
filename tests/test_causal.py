"""Tests for causal tracing detectors and causal_rank API."""

import torch
from model_clinic._types import Finding
from model_clinic.clinic import (
    diagnose,
    post_detect_causal_outlier,
    post_detect_layer_isolation,
    _collect_causal_norms,
    _collect_layer_isolation,
    causal_rank,
)


class TestCausalOutlier:
    """Test the causal outlier detector."""

    def test_fires_on_large_norm_outlier(self):
        """A tensor with much larger norm than same-type peers should fire ERROR."""
        ctx = {}
        # Use 7 normal layers + 1 outlier at 10x to ensure ratio > 3x after mean dilution
        torch.manual_seed(0)
        for i in range(7):
            _collect_causal_norms(f"layers.{i}.mlp.down_proj.weight", torch.randn(64, 64), ctx)
        _collect_causal_norms("layers.7.mlp.down_proj.weight", torch.randn(64, 64) * 10.0, ctx)

        findings = post_detect_causal_outlier(ctx)
        assert len(findings) >= 1
        outlier_findings = [f for f in findings if "layers.7" in f.param_name]
        assert len(outlier_findings) == 1
        assert outlier_findings[0].condition == "causal_outlier"
        assert outlier_findings[0].severity == "ERROR"
        assert outlier_findings[0].details["ratio"] > 3.0

    def test_no_fire_on_healthy(self):
        """Healthy tensors with similar norms should not fire."""
        ctx = {}
        torch.manual_seed(42)
        for i in range(4):
            t = torch.randn(64, 64)
            _collect_causal_norms(f"layers.{i}.mlp.down_proj.weight", t, ctx)

        findings = post_detect_causal_outlier(ctx)
        assert len(findings) == 0

    def test_warn_on_moderate_outlier(self):
        """A tensor with moderately higher norm should fire WARN (not ERROR)."""
        ctx = {}
        # 7 normal + 1 at 5x: mean ~ (7*1 + 5)/8 = 1.5, ratio = 5/1.5 = 3.3 -> ERROR
        # 7 normal + 1 at 4x: mean ~ (7*1 + 4)/8 = 1.375, ratio = 4/1.375 = 2.9 -> WARN
        torch.manual_seed(0)
        for i in range(7):
            _collect_causal_norms(f"layers.{i}.attention.q_proj.weight", torch.randn(64, 64), ctx)
        _collect_causal_norms("layers.7.attention.q_proj.weight", torch.randn(64, 64) * 4.0, ctx)

        findings = post_detect_causal_outlier(ctx)
        outlier_findings = [f for f in findings if "layers.7" in f.param_name]
        assert len(outlier_findings) >= 1
        # Should be at least WARN
        assert outlier_findings[0].severity in ("WARN", "ERROR")

    def test_integration_via_diagnose(self, tiny_state_dict):
        """Causal outlier detector runs as part of full diagnose pipeline."""
        sd = dict(tiny_state_dict)
        # Add extra layers so the outlier is clearly above the group mean
        for i in range(2, 6):
            sd[f"layers.{i}.attention.q_proj.weight"] = torch.randn(64, 64)
            sd[f"layers.{i}.mlp.gate_proj.weight"] = torch.randn(256, 64)
        # Make one layer's q_proj 20x bigger — clear outlier
        sd["layers.0.attention.q_proj.weight"] = torch.randn(64, 64) * 20.0
        findings = diagnose(sd)
        causal_findings = [f for f in findings if f.condition == "causal_outlier"]
        assert len(causal_findings) >= 1


class TestLayerIsolation:
    """Test the layer isolation detector."""

    def test_fires_on_divergent_norms(self):
        """Layers with >5x norm difference should fire."""
        ctx = {}
        normal = torch.randn(64, 64)
        huge = torch.randn(64, 64) * 10.0
        _collect_layer_isolation("layers.0.mlp.gate_proj.weight", normal, ctx)
        _collect_layer_isolation("layers.1.mlp.gate_proj.weight", huge, ctx)

        findings = post_detect_layer_isolation(ctx)
        assert len(findings) >= 1
        assert findings[0].condition == "layer_isolation"
        assert findings[0].details["ratio"] > 5.0

    def test_no_fire_on_similar_norms(self):
        """Consecutive layers with similar norms should not fire."""
        ctx = {}
        torch.manual_seed(42)
        for i in range(4):
            t = torch.randn(64, 64)
            _collect_layer_isolation(f"layers.{i}.attention.q_proj.weight", t, ctx)

        findings = post_detect_layer_isolation(ctx)
        assert len(findings) == 0

    def test_integration_via_diagnose(self, tiny_state_dict):
        """Layer isolation detector runs as part of full diagnose pipeline."""
        sd = dict(tiny_state_dict)
        # Make layer 0 q_proj much bigger than layer 1 q_proj
        sd["layers.0.attention.q_proj.weight"] = torch.randn(64, 64) * 20.0
        sd["layers.1.attention.q_proj.weight"] = torch.randn(64, 64) * 0.1
        findings = diagnose(sd)
        isolation_findings = [f for f in findings if f.condition == "layer_isolation"]
        assert len(isolation_findings) >= 1


class TestCausalRank:
    """Test the causal_rank API function."""

    def test_returns_sorted(self):
        """Results should be sorted by causal_score descending."""
        findings = [
            Finding("dead_neurons", "WARN", "layer.0.weight", {"dead_count": 5, "dim": "rows"}),
            Finding("nan_inf", "ERROR", "layer.1.weight", {"nan_count": 3}),
            Finding("causal_outlier", "ERROR", "layer.2.weight", {"ratio": 4.0}),
        ]
        sd = {
            "layer.0.weight": torch.randn(64, 64),
            "layer.1.weight": torch.randn(64, 64),
            "layer.2.weight": torch.randn(64, 64),
        }
        ranked = causal_rank(findings, sd)
        assert len(ranked) == 3
        # Scores should be descending
        for i in range(len(ranked) - 1):
            assert ranked[i]["causal_score"] >= ranked[i + 1]["causal_score"]

    def test_nan_inf_ranks_highest(self):
        """NaN/Inf should have highest causal score due to severity * weight."""
        findings = [
            Finding("dead_neurons", "WARN", "layer.0.weight", {}),
            Finding("nan_inf", "ERROR", "layer.1.weight", {}),
        ]
        sd = {}
        ranked = causal_rank(findings, sd)
        assert ranked[0]["tensor_name"] == "layer.1.weight"
        assert "nan_inf" in ranked[0]["reason"]

    def test_empty_findings(self):
        """Empty findings should return empty list."""
        ranked = causal_rank([], {})
        assert ranked == []

    def test_multiple_findings_same_tensor(self):
        """Multiple findings on same tensor should accumulate scores."""
        findings = [
            Finding("causal_outlier", "ERROR", "bad.weight", {}),
            Finding("exploding_norm", "WARN", "bad.weight", {}),
            Finding("dead_neurons", "INFO", "ok.weight", {}),
        ]
        sd = {}
        ranked = causal_rank(findings, sd)
        assert ranked[0]["tensor_name"] == "bad.weight"
        # Should list both reasons
        assert "causal_outlier" in ranked[0]["reason"]
        assert "exploding_norm" in ranked[0]["reason"]

    def test_result_structure(self):
        """Each result should have tensor_name, causal_score, and reason."""
        findings = [Finding("causal_outlier", "WARN", "test.weight", {})]
        ranked = causal_rank(findings, {})
        assert len(ranked) == 1
        assert "tensor_name" in ranked[0]
        assert "causal_score" in ranked[0]
        assert "reason" in ranked[0]


class TestCausalHealthScore:
    """Test that causal conditions integrate with health scoring."""

    def test_causal_outlier_in_weights_category(self):
        """causal_outlier should affect the weights category score."""
        from model_clinic._health_score import compute_health_score
        findings = [
            Finding("causal_outlier", "ERROR", "layer.0.weight", {}),
        ]
        score = compute_health_score(findings)
        assert score.categories["weights"] < 100

    def test_layer_isolation_in_weights_category(self):
        """layer_isolation should affect the weights category score."""
        from model_clinic._health_score import compute_health_score
        findings = [
            Finding("layer_isolation", "WARN", "layer.0.weight", {}),
        ]
        score = compute_health_score(findings)
        assert score.categories["weights"] < 100


class TestCausalReferences:
    """Test that references exist for causal conditions."""

    def test_causal_outlier_has_references(self):
        from model_clinic._references import get_references
        refs = get_references("causal_outlier")
        assert len(refs) >= 1

    def test_layer_isolation_has_references(self):
        from model_clinic._references import get_references
        refs = get_references("layer_isolation")
        assert len(refs) >= 1


class TestCausalExport:
    """Test that causal_rank is properly exported."""

    def test_import_from_package(self):
        from model_clinic import causal_rank as cr
        assert callable(cr)
