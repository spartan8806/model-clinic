"""Tests for synthetic model generators."""

import torch
import pytest

from model_clinic._synthetic import (
    SYNTHETIC_MODELS,
    make_healthy_mlp,
    make_dead_neuron_model,
    make_nan_model,
    make_exploding_model,
    make_norm_drift_model,
    make_collapsed_model,
    make_heavy_tails_model,
    make_duplicate_rows_model,
    make_stuck_gates_model,
    make_corrupted_model,
    make_everything_broken,
)
from model_clinic.clinic import diagnose


class TestSyntheticModelsDict:
    """Test the SYNTHETIC_MODELS registry."""

    def test_all_presets_present(self):
        expected = {
            "healthy", "dead-neurons", "nan", "exploding", "norm-drift",
            "collapsed", "heavy-tails", "duplicate-rows", "stuck-gates",
            "corrupted", "everything-broken",
        }
        assert set(SYNTHETIC_MODELS.keys()) == expected

    def test_all_presets_callable(self):
        for name, factory in SYNTHETIC_MODELS.items():
            assert callable(factory), f"{name} is not callable"

    def test_all_presets_return_state_dicts(self):
        for name, factory in SYNTHETIC_MODELS.items():
            sd = factory()
            assert isinstance(sd, dict), f"{name} did not return a dict"
            assert len(sd) > 0, f"{name} returned empty dict"
            for key, val in sd.items():
                assert isinstance(key, str), f"{name}: key {key!r} is not a string"
                assert isinstance(val, torch.Tensor), f"{name}: {key} is not a Tensor"


class TestHealthyModel:
    """Test that make_healthy_mlp produces a clean model."""

    def test_returns_valid_state_dict(self):
        sd = make_healthy_mlp()
        assert len(sd) > 0
        assert "lm_head.weight" in sd

    def test_no_errors(self):
        sd = make_healthy_mlp()
        findings = diagnose(sd)
        errors = [f for f in findings if f.severity == "ERROR"]
        assert len(errors) == 0, f"Healthy model has errors: {errors}"

    def test_custom_size(self):
        sd = make_healthy_mlp(hidden=128, layers=2)
        assert sd["layers.0.linear.weight"].shape == (128, 128)
        assert sd["layers.1.linear.weight"].shape == (128, 128)
        assert "layers.2.linear.weight" not in sd


class TestDeadNeuronModel:
    """Test make_dead_neuron_model."""

    def test_has_dead_neurons(self):
        sd = make_dead_neuron_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "dead_neurons" in conditions

    def test_dead_pct_respected(self):
        sd = make_dead_neuron_model(hidden=100, dead_pct=0.5)
        w = sd["layers.0.linear.weight"]
        zero_rows = (w.norm(dim=1) == 0).sum().item()
        assert zero_rows == 50


class TestNanModel:
    """Test make_nan_model."""

    def test_has_nans(self):
        sd = make_nan_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "nan_inf" in conditions

    def test_nan_count(self):
        sd = make_nan_model(nan_count=5)
        w = sd["layers.2.linear.weight"]
        assert torch.isnan(w).sum().item() == 5


class TestExplodingModel:
    """Test make_exploding_model."""

    def test_has_exploding_norms(self):
        sd = make_exploding_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "exploding_norm" in conditions


class TestNormDriftModel:
    """Test make_norm_drift_model."""

    def test_has_norm_drift(self):
        sd = make_norm_drift_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "norm_drift" in conditions


class TestCollapsedModel:
    """Test make_collapsed_model."""

    def test_has_identical_rows(self):
        sd = make_collapsed_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "identical_rows" in conditions


class TestHeavyTailsModel:
    """Test make_heavy_tails_model."""

    def test_has_heavy_tails(self):
        sd = make_heavy_tails_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "heavy_tails" in conditions


class TestDuplicateRowsModel:
    """Test make_duplicate_rows_model."""

    def test_has_duplicate_rows(self):
        sd = make_duplicate_rows_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "identical_rows" in conditions


class TestStuckGatesModel:
    """Test make_stuck_gates_model."""

    def test_has_stuck_gates(self):
        sd = make_stuck_gates_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "stuck_gate_closed" in conditions
        assert "stuck_gate_open" in conditions


class TestCorruptedModel:
    """Test make_corrupted_model."""

    def test_has_dead_neurons(self):
        sd = make_corrupted_model()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "dead_neurons" in conditions


class TestEverythingBroken:
    """Test the kitchen-sink broken model."""

    def test_multiple_finding_types(self):
        sd = make_everything_broken()
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        # Should have at least these core issues
        assert len(conditions) >= 4, f"Only found {conditions}"
        assert "dead_neurons" in conditions
        assert "nan_inf" in conditions
        assert "exploding_norm" in conditions
        assert "norm_drift" in conditions

    def test_has_errors_and_warnings(self):
        sd = make_everything_broken()
        findings = diagnose(sd)
        severities = {f.severity for f in findings}
        assert "ERROR" in severities
        assert "WARN" in severities

    def test_returns_valid_state_dict(self):
        sd = make_everything_broken()
        assert len(sd) > 0
        for key, val in sd.items():
            assert isinstance(val, torch.Tensor)
