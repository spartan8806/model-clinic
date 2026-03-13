"""Tests for the --explain flag and prescription explanations."""

import torch
from model_clinic._types import Finding, Prescription
from model_clinic.clinic import (
    diagnose, prescribe, print_exam,
    _rx_dead_neurons, _rx_nudge_gate, _rx_pull_gate,
    _rx_scale_norm, _rx_reinit_vanishing, _rx_clamp_tails,
    _rx_reset_norm, _rx_desaturate, _rx_fix_nan, _rx_perturb_identical,
)


# ── Helper fixtures ──────────────────────────────────────────────────────

def _make_finding(condition, **details):
    """Create a minimal Finding for prescription testing."""
    return Finding(condition=condition, severity="WARN", param_name="test.weight",
                   details=details)


# ── Tests ────────────────────────────────────────────────────────────────

class TestPrescriptionExplanations:
    """Every prescription function must produce a non-empty explanation."""

    def test_rx_dead_neurons_has_explanation(self):
        f = _make_finding("dead_neurons", dead_indices=[0, 1], dead_count=2,
                          total=64, pct=2/64, dim="rows")
        rx = _rx_dead_neurons(f)
        assert rx.explanation
        assert "dead neurons" in rx.explanation.lower()

    def test_rx_nudge_gate_has_explanation(self):
        f = _make_finding("stuck_gate_closed", raw=-10.0, sigmoid=0.00005)
        rx = _rx_nudge_gate(f)
        assert rx.explanation
        assert "stuck closed" in rx.explanation.lower() or "signal" in rx.explanation.lower()

    def test_rx_pull_gate_has_explanation(self):
        f = _make_finding("stuck_gate_open", raw=10.0, sigmoid=0.99995)
        rx = _rx_pull_gate(f)
        assert rx.explanation
        assert "gate" in rx.explanation.lower()

    def test_rx_scale_norm_has_explanation(self):
        f = _make_finding("exploding_norm", per_elem_norm=50.0, shape=[64, 64])
        rx = _rx_scale_norm(f)
        assert rx.explanation
        assert "norm" in rx.explanation.lower() or "amplif" in rx.explanation.lower()

    def test_rx_reinit_vanishing_has_explanation(self):
        f = _make_finding("vanishing_norm", per_elem_norm=1e-9, shape=[64, 64])
        rx = _rx_reinit_vanishing(f)
        assert rx.explanation
        assert "zero" in rx.explanation.lower() or "reinitializ" in rx.explanation.lower()

    def test_rx_clamp_tails_has_explanation(self):
        f = _make_finding("heavy_tails", kurtosis=200, std=0.5)
        rx = _rx_clamp_tails(f)
        assert rx.explanation
        assert "outlier" in rx.explanation.lower() or "kurtosis" in rx.explanation.lower()

    def test_rx_reset_norm_has_explanation(self):
        f = _make_finding("norm_drift", mean=2.5, expected=1.0)
        rx = _rx_reset_norm(f)
        assert rx.explanation
        assert "norm" in rx.explanation.lower() or "1.0" in rx.explanation

    def test_rx_desaturate_has_explanation(self):
        f = _make_finding("saturated_weights", near_max_pct=0.5, abs_max=1.0)
        rx = _rx_desaturate(f)
        assert rx.explanation
        assert "saturated" in rx.explanation.lower() or "dynamic range" in rx.explanation.lower()

    def test_rx_fix_nan_has_explanation(self):
        f = _make_finding("nan_inf", nan_count=5, inf_count=0, total=1024)
        rx = _rx_fix_nan(f)
        assert rx.explanation
        assert "nan" in rx.explanation.lower() or "cascading" in rx.explanation.lower()

    def test_rx_perturb_identical_has_explanation(self):
        f = _make_finding("identical_rows", duplicate_pairs=3, max_similarity=0.9999)
        rx = _rx_perturb_identical(f)
        assert rx.explanation
        assert "identical" in rx.explanation.lower() or "symmetry" in rx.explanation.lower()


class TestExplainIntegration:
    """Test that --explain works end-to-end."""

    def test_all_prescriptions_have_explanations(self):
        """Diagnose a sick model and ensure every prescription has an explanation."""
        torch.manual_seed(42)
        sd = {
            "layers.0.mlp.down_proj.weight": torch.randn(64, 256),
            "final_norm.weight": torch.full((64,), 2.5),
            "wrapper/gate": torch.tensor(-10.0),
            "bad_param": torch.randn(32, 32),
        }
        sd["layers.0.mlp.down_proj.weight"][0] = 0
        sd["layers.0.mlp.down_proj.weight"][1] = 0
        sd["bad_param"][5, 5] = float("nan")

        findings = diagnose(sd)
        prescriptions = prescribe(findings)
        assert len(prescriptions) > 0
        for rx in prescriptions:
            assert rx.explanation, f"Prescription {rx.name} missing explanation"

    def test_explanation_in_prescription_dataclass(self):
        """Verify the explanation field exists and defaults correctly."""
        f = Finding("test", "WARN", "p", {})
        rx = Prescription("test", "desc", "low", f, "none")
        assert rx.explanation == ""

        rx2 = Prescription("test", "desc", "low", f, "none",
                           explanation="This is why")
        assert rx2.explanation == "This is why"

    def test_print_exam_with_explain(self, capsys):
        """Verify --explain flag prints WHY lines."""
        f = _make_finding("norm_drift", mean=2.5, expected=1.0)
        rx = _rx_reset_norm(f)
        print_exam([f], [rx], explain=True)
        out = capsys.readouterr().out
        assert "WHY:" in out

    def test_print_exam_without_explain(self, capsys):
        """Without --explain, WHY lines should not appear."""
        f = _make_finding("norm_drift", mean=2.5, expected=1.0)
        rx = _rx_reset_norm(f)
        print_exam([f], [rx], explain=False)
        out = capsys.readouterr().out
        assert "WHY:" not in out
