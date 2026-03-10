"""Tests for error handling — bad paths, corrupt data, edge cases."""

import torch
import pytest
from model_clinic._loader import load_state_dict, build_meta, save_state_dict
from model_clinic.clinic import diagnose, prescribe, apply_treatment
from model_clinic._types import Finding, Prescription


class TestLoaderErrors:
    """Test loader handles bad input gracefully."""

    def test_missing_file(self):
        with pytest.raises((FileNotFoundError, ValueError, RuntimeError, OSError)):
            load_state_dict("/nonexistent/path/model.pt")

    def test_empty_state_dict(self):
        meta = build_meta({})
        assert meta.num_params == 0
        assert meta.num_tensors == 0
        assert meta.hidden_size == 0

    def test_non_tensor_values(self):
        """State dict with non-tensor values should be handled."""
        sd = {"config": "not a tensor", "weight": torch.randn(32, 32)}
        findings = diagnose(sd)
        # Should not crash, should only analyze the tensor
        param_names = {f.param_name for f in findings}
        assert "config" not in param_names

    def test_save_and_reload_roundtrip(self, tmp_path):
        """Save then reload should preserve all tensors."""
        sd = {"w": torch.randn(16, 16), "b": torch.randn(16)}
        orig = tmp_path / "orig.pt"
        out = tmp_path / "out.pt"
        torch.save({"model_state_dict": sd}, str(orig))

        loaded, _ = load_state_dict(str(orig))
        loaded["w"] = torch.ones(16, 16)
        save_state_dict(loaded, str(orig), str(out))

        reloaded, _ = load_state_dict(str(out))
        assert torch.allclose(reloaded["w"], torch.ones(16, 16))


class TestTreatmentErrors:
    """Test treatment handles edge cases."""

    def test_missing_param(self):
        """Treatment on non-existent param should fail gracefully."""
        sd = {"other": torch.randn(32, 32)}
        f = Finding("nan_inf", "ERROR", "missing_param",
                    {"nan_count": 1, "inf_count": 0, "total": 1024})
        rx = Prescription("fix_nan", "Fix", "high", f, "fix_nan_inf")
        result = apply_treatment(sd, rx)
        assert not result.success
        assert "not found" in result.description

    def test_non_float_tensor(self):
        """Treatment on int tensor should skip gracefully."""
        sd = {"counter": torch.tensor(42, dtype=torch.long)}
        f = Finding("exploding_norm", "WARN", "counter",
                    {"per_elem_norm": 42.0, "shape": []})
        rx = Prescription("scale", "Scale", "medium", f, "scale_norm",
                          {"target_per_elem": 1.0})
        result = apply_treatment(sd, rx)
        assert not result.success
        assert "non-float" in result.description.lower() or "Skipped" in result.description

    def test_unknown_action(self):
        """Unknown treatment action should fail gracefully."""
        sd = {"w": torch.randn(32, 32)}
        f = Finding("custom", "WARN", "w", {})
        rx = Prescription("custom_fix", "Fix", "medium", f, "nonexistent_action")
        result = apply_treatment(sd, rx)
        assert not result.success

    def test_zero_norm_scale(self):
        """Scaling a zero-norm tensor should not crash."""
        sd = {"w": torch.zeros(32, 32)}
        f = Finding("vanishing_norm", "WARN", "w",
                    {"per_elem_norm": 0.0, "shape": [32, 32]})
        rx = Prescription("reinit", "Reinit", "low", f, "reinit_full")
        result = apply_treatment(sd, rx)
        assert result.success


class TestDiagnoseEdgeCases:
    """Test diagnosis on unusual inputs."""

    def test_single_param(self):
        """Single-parameter state dict should work."""
        sd = {"w": torch.randn(64, 64)}
        findings = diagnose(sd)
        # Should not crash
        assert isinstance(findings, list)

    def test_scalar_tensor(self):
        """Scalar tensors (gates) should be handled."""
        sd = {"wrapper/gate": torch.tensor(-5.0)}
        findings = diagnose(sd)
        gates = [f for f in findings if "gate" in f.condition]
        # detect_stuck_gates runs for both stuck_gate_closed and stuck_gate_open registrations
        assert len(gates) >= 1
        assert all(f.condition == "stuck_gate_closed" for f in gates)

    def test_1d_tensor(self):
        """1D tensors (biases, norms) should be handled."""
        sd = {"layer.0.bias": torch.randn(64)}
        findings = diagnose(sd)
        assert isinstance(findings, list)

    def test_empty_state_dict(self):
        """Empty state dict should return no findings."""
        findings = diagnose({})
        assert len(findings) == 0

    def test_very_small_tensor(self):
        """Tiny tensors should not trigger false positives."""
        sd = {"small": torch.randn(2, 2)}
        findings = diagnose(sd)
        # Should not have dead neurons etc on 2x2
        dead = [f for f in findings if f.condition == "dead_neurons"]
        assert len(dead) == 0
