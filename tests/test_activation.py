"""Tests for Level 5: Activation-Guided Repair."""

import math

import torch
import torch.nn as nn
import pytest

from model_clinic._repair.activation import (
    effective_rank,
    token_entropy,
    activation_audit,
    find_destructive_layers,
    activation_repair,
    LayerStats,
    ActivationReport,
)


# ---------------------------------------------------------------------------
# Tiny test model
# ---------------------------------------------------------------------------

class TinyModel(nn.Module):
    """3-layer model with hidden_dim=32.  Layer 1 is made destructive."""

    def __init__(self, hidden=32, destructive: bool = True):
        super().__init__()
        self.layer0 = nn.Linear(hidden, hidden)
        self.layer1 = nn.Linear(hidden, hidden)  # will be made destructive
        self.layer2 = nn.Linear(hidden, hidden)

        # Initialise normally
        nn.init.eye_(self.layer0.weight)
        nn.init.eye_(self.layer2.weight)
        nn.init.zeros_(self.layer0.bias)
        nn.init.zeros_(self.layer2.bias)

        if destructive:
            # Make layer1 explode norms (weight * 100)
            with torch.no_grad():
                self.layer1.weight.copy_(torch.eye(hidden) * 100.0)
                self.layer1.bias.zero_()
        else:
            nn.init.eye_(self.layer1.weight)
            nn.init.zeros_(self.layer1.bias)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class SingleLayerModel(nn.Module):
    """Model with a single linear layer."""

    def __init__(self, hidden=32):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


def _make_calibration(n=5, hidden=32):
    """Create small calibration tensors (batch=1, seq=4, hidden)."""
    torch.manual_seed(42)
    return [torch.randn(1, 4, hidden) for _ in range(n)]


# ---------------------------------------------------------------------------
# effective_rank tests
# ---------------------------------------------------------------------------

class TestEffectiveRank:

    def test_identity_full_rank(self):
        """Identity matrix should have effective rank close to its dimension."""
        eye = torch.eye(16)
        r = effective_rank(eye)
        # All singular values equal → max entropy → rank = 16
        assert abs(r - 16.0) < 0.01

    def test_rank_1_matrix(self):
        """Rank-1 matrix should have effective rank close to 1."""
        u = torch.randn(32, 1)
        v = torch.randn(1, 32)
        t = u @ v
        r = effective_rank(t)
        assert r < 1.5

    def test_zeros(self):
        """All-zero matrix should have rank 0."""
        r = effective_rank(torch.zeros(8, 8))
        assert r == 0.0

    def test_1d_nonzero(self):
        r = effective_rank(torch.tensor([1.0, 2.0, 3.0]))
        assert r == 1.0

    def test_1d_zero(self):
        r = effective_rank(torch.zeros(5))
        assert r == 0.0

    def test_3d_tensor(self):
        """3-D tensor should be flattened to 2-D for SVD."""
        t = torch.randn(2, 8, 8)
        r = effective_rank(t)
        assert r >= 1.0

    def test_empty(self):
        r = effective_rank(torch.tensor([]))
        assert r == 0.0

    def test_rank_between_extremes(self):
        """A random matrix should have rank between 1 and min(M, N)."""
        torch.manual_seed(0)
        t = torch.randn(16, 16)
        r = effective_rank(t)
        assert 1.0 <= r <= 16.0


# ---------------------------------------------------------------------------
# token_entropy tests
# ---------------------------------------------------------------------------

class TestTokenEntropy:

    def test_uniform_high_entropy(self):
        """Uniform norms should yield high entropy."""
        t = torch.ones(1, 8, 4)  # 8 tokens, all same norm
        e = token_entropy(t)
        # Max entropy for 8 tokens = ln(8) ≈ 2.08
        assert e > 1.5

    def test_peaked_low_entropy(self):
        """One dominant token should yield lower entropy."""
        t = torch.zeros(1, 8, 4)
        t[0, 0, :] = 100.0  # all energy in first token
        e = token_entropy(t)
        # Should be close to 0 (one token dominates)
        assert e < 0.5

    def test_all_zero(self):
        e = token_entropy(torch.zeros(4, 4))
        assert e == 0.0

    def test_1d(self):
        e = token_entropy(torch.tensor([1.0, 2.0]))
        assert e == 0.0  # 1-D → not enough dims


# ---------------------------------------------------------------------------
# activation_audit tests
# ---------------------------------------------------------------------------

class TestActivationAudit:

    def test_captures_all_layers(self):
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        # Should have entries for layer0, layer1, layer2
        assert "layer0" in stats
        assert "layer1" in stats
        assert "layer2" in stats

    def test_stats_are_layer_stats(self):
        model = TinyModel(destructive=False)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        for name, s in stats.items():
            assert isinstance(s, LayerStats)
            assert s.name == name

    def test_healthy_model_no_extreme_norms(self):
        """A healthy model should not have extreme norm ratios."""
        model = TinyModel(destructive=False)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        for s in stats.values():
            assert 0.05 < s.norm_ratio < 20.0, f"{s.name} norm_ratio={s.norm_ratio}"

    def test_destructive_model_has_extreme_norms(self):
        """Layer1 with weight*100 should show extreme norm ratio."""
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        assert stats["layer1"].norm_ratio > 10.0

    def test_single_layer_model(self):
        model = SingleLayerModel()
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        assert "linear" in stats

    def test_empty_calibration(self):
        model = TinyModel(destructive=False)
        stats = activation_audit(model, [])
        assert len(stats) == 0


# ---------------------------------------------------------------------------
# find_destructive_layers tests
# ---------------------------------------------------------------------------

class TestFindDestructiveLayers:

    def test_finds_bad_layer(self):
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        destructive = find_destructive_layers(stats)
        assert "layer1" in destructive

    def test_healthy_model_no_destructive(self):
        model = TinyModel(destructive=False)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        destructive = find_destructive_layers(stats)
        assert len(destructive) == 0

    def test_custom_thresholds(self):
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        stats = activation_audit(model, cal)
        # Very lenient thresholds — nothing should be flagged
        destructive = find_destructive_layers(
            stats, rank_threshold=-10.0, norm_low=0.001, norm_high=10000.0,
            entropy_threshold=-100.0,
        )
        assert len(destructive) == 0

    def test_empty_stats(self):
        destructive = find_destructive_layers({})
        assert destructive == []


# ---------------------------------------------------------------------------
# activation_repair tests
# ---------------------------------------------------------------------------

class TestActivationRepairShrink:

    def test_shrink_reduces_weights(self):
        model = TinyModel(destructive=True)
        orig_norm = model.layer1.weight.norm().item()
        sd, repaired = activation_repair(model, ["layer1"], strategy="shrink", shrink_factor=0.1)
        assert "layer1" in repaired
        new_norm = sd["layer1.weight"].norm().item()
        assert new_norm < orig_norm * 0.2  # should be ~10% of original

    def test_shrink_no_destructive_is_noop(self):
        model = TinyModel(destructive=False)
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}
        sd, repaired = activation_repair(model, [], strategy="shrink")
        assert repaired == []
        for k in sd_before:
            assert torch.allclose(sd[k], sd_before[k])

    def test_after_shrink_no_longer_destructive(self):
        """After shrinking the bad layer, it should no longer be flagged."""
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        # Repair
        sd, repaired = activation_repair(model, ["layer1"], strategy="shrink", shrink_factor=0.01)
        assert "layer1" in repaired
        # Re-load and re-audit
        model.load_state_dict(sd)
        stats = activation_audit(model, cal)
        destructive = find_destructive_layers(stats)
        assert "layer1" not in destructive


class TestActivationRepairPassthrough:

    def test_passthrough_makes_identity(self):
        model = TinyModel(destructive=True)
        sd, repaired = activation_repair(model, ["layer1"], strategy="passthrough")
        assert "layer1" in repaired
        w = sd["layer1.weight"]
        assert torch.allclose(w, torch.eye(32), atol=1e-6)
        assert torch.allclose(sd["layer1.bias"], torch.zeros(32), atol=1e-6)

    def test_after_passthrough_no_longer_destructive(self):
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        sd, repaired = activation_repair(model, ["layer1"], strategy="passthrough")
        model.load_state_dict(sd)
        stats = activation_audit(model, cal)
        destructive = find_destructive_layers(stats)
        assert "layer1" not in destructive


class TestActivationRepairInterpolate:

    def test_interpolate_uses_neighbours(self):
        model = TinyModel(destructive=True)
        # layer0 and layer2 are identity — their average is identity
        sd, repaired = activation_repair(model, ["layer1"], strategy="interpolate")
        assert "layer1" in repaired
        w = sd["layer1.weight"]
        # Average of two identity matrices is identity
        assert torch.allclose(w, torch.eye(32), atol=1e-6)

    def test_interpolate_edge_layer_falls_back(self):
        """First layer has only one neighbour; should still work."""
        model = TinyModel(destructive=False)
        # Make layer0 destructive
        with torch.no_grad():
            model.layer0.weight.mul_(100.0)
        sd, repaired = activation_repair(model, ["layer0"], strategy="interpolate")
        assert "layer0" in repaired
        # layer0 has only one neighbour (layer1) — its weight should be copied
        assert torch.allclose(sd["layer0.weight"], torch.eye(32), atol=1e-6)

    def test_after_interpolate_no_longer_destructive(self):
        model = TinyModel(destructive=True)
        cal = _make_calibration()
        sd, repaired = activation_repair(model, ["layer1"], strategy="interpolate")
        model.load_state_dict(sd)
        stats = activation_audit(model, cal)
        destructive = find_destructive_layers(stats)
        assert "layer1" not in destructive


class TestActivationRepairEdgeCases:

    def test_unknown_strategy_raises(self):
        model = TinyModel(destructive=True)
        with pytest.raises(ValueError, match="Unknown repair strategy"):
            activation_repair(model, ["layer1"], strategy="magic")

    def test_nonexistent_layer_skipped(self):
        model = TinyModel(destructive=True)
        sd, repaired = activation_repair(model, ["nonexistent"], strategy="shrink")
        assert repaired == []

    def test_single_layer_model_interpolate_falls_back(self):
        """A model with only one linear layer can't interpolate; should fall back to shrink."""
        model = SingleLayerModel()
        with torch.no_grad():
            model.linear.weight.mul_(100.0)
        orig_norm = model.linear.weight.norm().item()
        sd, repaired = activation_repair(model, ["linear"], strategy="interpolate")
        assert "linear" in repaired
        # Should have shrunk (fallback) — norm should be much smaller
        new_norm = sd["linear.weight"].norm().item()
        assert new_norm < orig_norm * 0.5


# ---------------------------------------------------------------------------
# ActivationReport dataclass
# ---------------------------------------------------------------------------

class TestActivationReport:

    def test_creation(self):
        report = ActivationReport(
            all_stats={},
            destructive_layers=[],
            repairs_applied=["layer1"],
            strategy_used="shrink",
        )
        assert report.strategy_used == "shrink"
        assert report.repairs_applied == ["layer1"]

    def test_defaults(self):
        report = ActivationReport(all_stats={}, destructive_layers=[])
        assert report.repairs_applied == []
        assert report.strategy_used is None


# ---------------------------------------------------------------------------
# End-to-end: audit → find → repair → re-audit
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self):
        model = TinyModel(destructive=True)
        cal = _make_calibration()

        # 1. Audit
        stats = activation_audit(model, cal)
        assert len(stats) > 0

        # 2. Find destructive
        destructive = find_destructive_layers(stats)
        assert "layer1" in destructive

        # 3. Repair
        sd, repaired = activation_repair(model, destructive, strategy="shrink", shrink_factor=0.01)
        assert len(repaired) > 0

        # 4. Re-audit
        model.load_state_dict(sd)
        stats2 = activation_audit(model, cal)
        destructive2 = find_destructive_layers(stats2)

        # 5. Previously destructive layer should be fixed
        assert "layer1" not in destructive2

    def test_healthy_model_pipeline(self):
        """Full pipeline on a healthy model should be a no-op."""
        model = TinyModel(destructive=False)
        cal = _make_calibration()

        stats = activation_audit(model, cal)
        destructive = find_destructive_layers(stats)
        assert len(destructive) == 0

        sd, repaired = activation_repair(model, destructive, strategy="shrink")
        assert repaired == []
