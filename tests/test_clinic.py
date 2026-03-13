"""Tests for the clinic module — diagnosis, prescription, treatment."""

import torch
from model_clinic._types import Finding, Prescription
from model_clinic.clinic import (
    diagnose, prescribe, apply_treatment, rollback_treatment,
    detect_dead_neurons, detect_stuck_gates, detect_nan_inf,
    detect_norm_drift, detect_exploding_norm, detect_heavy_tails,
    detect_identical_rows, _is_metadata_tensor,
)


class TestDetectors:
    """Test individual condition detectors."""

    def test_dead_neurons_rows(self):
        t = torch.randn(64, 32)
        t[0] = 0  # kill row 0
        t[5] = 0  # kill row 5
        findings = detect_dead_neurons("test.weight", t, {})
        assert any(f.condition == "dead_neurons" and f.details["dim"] == "rows" for f in findings)
        row_f = [f for f in findings if f.details.get("dim") == "rows"][0]
        assert row_f.details["dead_count"] == 2

    def test_dead_neurons_none_healthy(self):
        t = torch.randn(64, 32)
        findings = detect_dead_neurons("test.weight", t, {})
        assert not any(f.details.get("dim") == "rows" for f in findings)

    def test_stuck_gate_closed(self):
        t = torch.tensor(-10.0)
        findings = detect_stuck_gates("wrapper/gate", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "stuck_gate_closed"

    def test_stuck_gate_open(self):
        t = torch.tensor(10.0)
        findings = detect_stuck_gates("wrapper/gate", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "stuck_gate_open"

    def test_gate_healthy(self):
        t = torch.tensor(0.0)  # sigmoid = 0.5
        findings = detect_stuck_gates("wrapper/gate", t, {})
        assert len(findings) == 0

    def test_nan_detection(self):
        t = torch.randn(32, 32)
        t[5, 5] = float("nan")
        findings = detect_nan_inf("bad.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "nan_inf"
        assert findings[0].details["nan_count"] == 1

    def test_inf_detection(self):
        t = torch.randn(32, 32)
        t[0, 0] = float("inf")
        findings = detect_nan_inf("bad.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["inf_count"] == 1

    def test_norm_drift(self):
        # threshold is 1.5: |mean - 1.0| > 1.5, so mean must exceed 2.5
        t = torch.full((64,), 3.0)
        findings = detect_norm_drift("final_norm.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "norm_drift"

    def test_norm_healthy(self):
        t = torch.ones(64)
        findings = detect_norm_drift("final_norm.weight", t, {})
        assert len(findings) == 0

    def test_exploding_norm(self):
        t = torch.randn(64, 64) * 100
        findings = detect_exploding_norm("big.weight", t, {})
        assert len(findings) == 1

    def test_heavy_tails(self):
        # Create distribution with extreme outliers
        t = torch.randn(10000)
        t[0] = 100.0
        t[1] = -100.0
        findings = detect_heavy_tails("outlier.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["kurtosis"] > 50

    def test_identical_rows(self):
        t = torch.randn(64, 32)
        t[1] = t[0].clone()  # make row 1 identical to row 0
        t[3] = t[2].clone()  # make row 3 identical to row 2
        findings = detect_identical_rows("dup.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["duplicate_pairs"] >= 2


class TestDiagnose:
    """Test the full diagnosis pipeline."""

    def test_healthy_model(self, tiny_state_dict):
        findings = diagnose(tiny_state_dict)
        errors = [f for f in findings if f.severity == "ERROR"]
        assert len(errors) == 0

    def test_sick_model(self, sick_state_dict):
        findings = diagnose(sick_state_dict)
        conditions = {f.condition for f in findings}
        assert "dead_neurons" in conditions
        assert "norm_drift" in conditions
        assert "stuck_gate_closed" in conditions
        assert "stuck_gate_open" in conditions
        assert "nan_inf" in conditions

    def test_metadata_tensors_skipped(self):
        """Growth/tracking metadata tensors should not be diagnosed."""
        sd = {
            "model.layers.0.mlp.gate_proj.neuron_age": torch.tensor(1790),
            "model.layers.0.mlp.gate_proj.total_steps": torch.tensor(1790),
            "model.layers.0.mlp.gate_proj.gradient_sq_ema": torch.zeros(128),
            "model.layers.0.mlp.gate_proj.loss_idx": torch.tensor(5),
            "model.layers.0.mlp.gate_proj.original_size": torch.tensor(128),
            # A real weight — should still be diagnosed
            "model.layers.0.mlp.gate_proj.weight": torch.randn(64, 64) * 100,
        }
        findings = diagnose(sd)
        # Only the real weight should produce findings, none from metadata
        for f in findings:
            assert not _is_metadata_tensor(f.param_name, sd.get(f.param_name, torch.tensor(0)))

    def test_metadata_filter_function(self):
        """Direct test of _is_metadata_tensor."""
        assert _is_metadata_tensor("layer.0.neuron_age", torch.tensor(1))
        assert _is_metadata_tensor("layer.0.total_steps", torch.tensor(1))
        assert _is_metadata_tensor("layer.0.gradient_sq_ema", torch.zeros(10))
        assert not _is_metadata_tensor("layer.0.weight", torch.randn(10, 10))
        assert not _is_metadata_tensor("layer.0.bias", torch.randn(10))
        assert not _is_metadata_tensor("final_norm.weight", torch.ones(10))


class TestPrescribe:
    """Test prescription generation."""

    def test_prescriptions_generated(self, sick_state_dict):
        findings = diagnose(sick_state_dict)
        prescriptions = prescribe(findings)
        assert len(prescriptions) > 0
        names = {rx.name for rx in prescriptions}
        assert "fix_nan_inf" in names
        assert "reset_norm" in names

    def test_conservative_mode(self, sick_state_dict):
        findings = diagnose(sick_state_dict)
        all_rx = prescribe(findings, conservative=False)
        safe_rx = prescribe(findings, conservative=True)
        assert len(safe_rx) <= len(all_rx)
        for rx in safe_rx:
            assert rx.risk == "low"


class TestTreatment:
    """Test treatment application and rollback."""

    def test_fix_nan(self):
        sd = {"bad": torch.randn(32, 32)}
        sd["bad"][5, 5] = float("nan")
        finding = Finding("nan_inf", "ERROR", "bad",
                          {"nan_count": 1, "inf_count": 0, "total": 1024})
        rx = Prescription("fix_nan_inf", "Fix NaN", "high", finding,
                          "fix_nan_inf")
        result = apply_treatment(sd, rx)
        assert result.success
        assert not torch.isnan(sd["bad"]).any()

    def test_reset_norm(self):
        sd = {"final_norm.weight": torch.full((64,), 2.5)}
        finding = Finding("norm_drift", "WARN", "final_norm.weight",
                          {"mean": 2.5, "expected": 1.0})
        rx = Prescription("reset_norm", "Reset norm", "low", finding,
                          "reset_norm_weights")
        result = apply_treatment(sd, rx)
        assert result.success
        assert abs(sd["final_norm.weight"].mean().item() - 1.0) < 0.01

    def test_rollback(self):
        sd = {"final_norm.weight": torch.full((64,), 2.5)}
        original = sd["final_norm.weight"].clone()
        finding = Finding("norm_drift", "WARN", "final_norm.weight",
                          {"mean": 2.5, "expected": 1.0})
        rx = Prescription("reset_norm", "Reset norm", "low", finding,
                          "reset_norm_weights")
        result = apply_treatment(sd, rx)
        assert sd["final_norm.weight"].mean().item() == 1.0

        # Rollback
        rollback_treatment(sd, result)
        assert torch.allclose(sd["final_norm.weight"], original)

    def test_dry_run(self):
        sd = {"final_norm.weight": torch.full((64,), 2.5)}
        finding = Finding("norm_drift", "WARN", "final_norm.weight",
                          {"mean": 2.5, "expected": 1.0})
        rx = Prescription("reset_norm", "Reset norm", "low", finding,
                          "reset_norm_weights")
        result = apply_treatment(sd, rx, dry_run=True)
        assert result.success
        # Should NOT have changed the tensor
        assert sd["final_norm.weight"].mean().item() == 2.5

    def test_reinit_dead_neurons(self):
        sd = {"test.weight": torch.randn(64, 32)}
        sd["test.weight"][0] = 0
        sd["test.weight"][5] = 0
        finding = Finding("dead_neurons", "WARN", "test.weight",
                          {"dead_indices": [0, 5], "dead_count": 2,
                           "total": 64, "pct": 2/64, "dim": "rows"})
        rx = Prescription("reinit_dead_neurons", "Reinit", "low", finding,
                          "reinit_dead", {"indices": [0, 5], "dim": "rows"})
        result = apply_treatment(sd, rx)
        assert result.success
        assert sd["test.weight"][0].norm().item() > 0
        assert sd["test.weight"][5].norm().item() > 0
