"""Tests for cross-checkpoint grafting (Level 4 deep repair)."""

import json
import os
import subprocess
import sys

import torch
import pytest

from model_clinic._repair.graft import (
    graft,
    graft_modules,
    score_parameter,
    GraftManifest,
    _group_keys_by_module,
)

# Common subprocess kwargs for Windows compatibility
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
_RUN_KWARGS = {"stdin": subprocess.DEVNULL, "env": _SUBPROCESS_ENV}


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_healthy_sd():
    """Create a healthy state dict with well-conditioned parameters."""
    torch.manual_seed(42)
    return {
        "layers.0.attention.weight": torch.randn(64, 64) * 0.02,
        "layers.0.attention.bias": torch.zeros(64),
        "layers.0.norm.weight": torch.ones(64),
        "layers.1.feed_forward.weight": torch.randn(128, 64) * 0.02,
        "layers.1.feed_forward.bias": torch.zeros(128),
        "layers.1.norm.weight": torch.ones(128),
        "embed.weight": torch.randn(256, 64) * 0.02,
    }


def _make_damaged_sd_a():
    """SD with healthy layers.0 but damaged layers.1 and embed."""
    sd = _make_healthy_sd()
    # Damage layers.1: inject NaN
    sd["layers.1.feed_forward.weight"][0, 0] = float("nan")
    sd["layers.1.feed_forward.weight"][1, 1] = float("inf")
    # Damage embed: all zeros (dead)
    sd["embed.weight"] = torch.zeros(256, 64)
    return sd


def _make_damaged_sd_b():
    """SD with healthy layers.1 and embed but damaged layers.0."""
    sd = _make_healthy_sd()
    # Damage layers.0: inject NaN
    sd["layers.0.attention.weight"][0, 0] = float("nan")
    sd["layers.0.attention.weight"][10, 10] = float("inf")
    # Damage layers.0 norm: extreme drift
    sd["layers.0.norm.weight"] = torch.full((64,), 50.0)
    return sd


# ── score_parameter tests ────────────────────────────────────────────────


class TestScoreParameter:
    """Tests for the score_parameter function."""

    def test_healthy_tensor_high_score(self):
        """A clean random tensor should score high."""
        t = torch.randn(64, 64) * 0.02
        score = score_parameter("test.weight", t)
        assert score >= 80, f"Healthy tensor scored {score}, expected >= 80"

    def test_nan_tensor_low_score(self):
        """A tensor with NaN should score lower."""
        t = torch.randn(64, 64) * 0.02
        t[0, 0] = float("nan")
        score = score_parameter("test.weight", t)
        score_healthy = score_parameter("test.weight", torch.randn(64, 64) * 0.02)
        assert score < score_healthy, "NaN tensor should score lower than healthy"

    def test_scalar_tensor_perfect(self):
        """Scalar tensors should return 100."""
        t = torch.tensor(3.14)
        score = score_parameter("scale", t)
        assert score == 100.0

    def test_tiny_tensor_perfect(self):
        """Very small tensors (< 4 elements) should return 100."""
        t = torch.tensor([1.0, 2.0, 3.0])
        score = score_parameter("small", t)
        assert score == 100.0

    def test_ones_norm_weight(self):
        """A norm weight of all ones should be healthy."""
        t = torch.ones(128)
        score = score_parameter("norm.weight", t)
        assert score >= 80


# ── graft tests ──────────────────────────────────────────────────────────


class TestGraft:
    """Tests for the graft function."""

    def test_complementary_damage(self, tmp_path):
        """Graft should pick healthy layers from each checkpoint."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()

        path_a = tmp_path / "ckpt_a.pt"
        path_b = tmp_path / "ckpt_b.pt"
        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        merged, manifest = graft([str(path_a), str(path_b)])

        # layers.0 should come from A (healthy there, damaged in B)
        assert manifest.sources["layers.0.attention.weight"]["source_checkpoint"] == str(path_a)

        # layers.1 should come from B (healthy there, damaged in A due to NaN)
        assert manifest.sources["layers.1.feed_forward.weight"]["source_checkpoint"] == str(path_b)

        # embed should come from B (zeros in A = dead, random in B = healthy)
        assert manifest.sources["embed.weight"]["source_checkpoint"] == str(path_b)

        # Merged should have no NaN
        for key, tensor in merged.items():
            assert not torch.isnan(tensor).any(), f"NaN in merged {key}"
            assert not torch.isinf(tensor).any(), f"Inf in merged {key}"

    def test_single_checkpoint(self, tmp_path):
        """Single checkpoint should return a copy of itself."""
        sd = _make_healthy_sd()
        path = tmp_path / "only.pt"
        torch.save(sd, str(path))

        merged, manifest = graft([str(path)])
        assert len(merged) == len(sd)
        assert manifest.summary["total_params"] == len(sd)
        for key in sd:
            assert torch.equal(merged[key], sd[key])

    def test_three_checkpoints(self, tmp_path):
        """Graft should work with 3+ checkpoints and produce valid output."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()
        # Third checkpoint: also damaged but differently
        sd_c = _make_healthy_sd()
        sd_c["layers.0.attention.weight"][5, 5] = float("nan")
        sd_c["layers.1.feed_forward.weight"][3, 3] = float("nan")
        sd_c["embed.weight"][0, 0] = float("nan")

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        path_c = tmp_path / "c.pt"
        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))
        torch.save(sd_c, str(path_c))

        merged, manifest = graft([str(path_a), str(path_b), str(path_c)])

        # All three sources should be represented
        total = manifest.summary["total_params"]
        assert total == len(sd_a)
        # Merged should exist and have all keys
        assert len(merged) == total
        # At least two different sources should be used (each has different damage)
        num_sources = len(manifest.summary["params_from_each_source"])
        assert num_sources >= 2, f"Expected params from >= 2 sources, got {num_sources}"

    def test_mismatched_keys(self, tmp_path):
        """Params that exist in only one checkpoint should be included."""
        sd_a = _make_healthy_sd()
        sd_b = _make_healthy_sd()
        # Add unique key to A
        sd_a["unique_to_a.weight"] = torch.randn(32, 32) * 0.02
        # Add unique key to B
        sd_b["unique_to_b.weight"] = torch.randn(16, 16) * 0.02

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        merged, manifest = graft([str(path_a), str(path_b)])

        assert "unique_to_a.weight" in merged
        assert "unique_to_b.weight" in merged
        assert manifest.sources["unique_to_a.weight"]["source_checkpoint"] == str(path_a)
        assert manifest.sources["unique_to_b.weight"]["source_checkpoint"] == str(path_b)
        assert manifest.sources["unique_to_a.weight"]["runner_up_score"] is None
        assert manifest.sources["unique_to_b.weight"]["runner_up_score"] is None

    def test_identical_checkpoints(self, tmp_path):
        """Two identical checkpoints should produce the same result."""
        sd = _make_healthy_sd()
        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        torch.save(sd, str(path_a))
        torch.save(sd, str(path_b))

        merged, manifest = graft([str(path_a), str(path_b)])
        assert len(merged) == len(sd)
        for key in sd:
            assert torch.equal(merged[key], sd[key])

    def test_empty_checkpoint_raises(self):
        """No checkpoints should raise ValueError."""
        with pytest.raises(ValueError):
            graft([])

    def test_merged_score_not_worse(self, tmp_path):
        """Merged score should be >= the best individual checkpoint score."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        merged, manifest = graft([str(path_a), str(path_b)])

        before_scores = manifest.summary["overall_score_before"]
        best_before = max(before_scores.values())
        after = manifest.summary["overall_score_after"]
        assert after >= best_before, f"Merged ({after}) should be >= best individual ({best_before})"


# ── graft_modules tests ─────────────────────────────────────────────────


class TestGraftModules:
    """Tests for module-level grafting."""

    def test_module_grouping(self):
        """_group_keys_by_module should group by prefix."""
        keys = [
            "layers.0.attention.weight",
            "layers.0.attention.bias",
            "layers.0.norm.weight",
            "layers.1.ff.weight",
            "embed.weight",
        ]
        groups = _group_keys_by_module(keys, depth=2)
        assert "layers.0" in groups
        assert len(groups["layers.0"]) == 3
        assert "layers.1" in groups
        assert len(groups["layers.1"]) == 1
        # embed.weight has only 2 parts, so prefix is "embed"
        assert "embed" in groups

    def test_module_level_complementary(self, tmp_path):
        """Module-level graft should pick healthy modules."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        merged, manifest = graft_modules([str(path_a), str(path_b)], depth=2)

        # All params in a module group should come from the same checkpoint
        # Check layers.0 group — should come from A
        layers0_sources = set()
        for key in ["layers.0.attention.weight", "layers.0.attention.bias", "layers.0.norm.weight"]:
            if key in manifest.sources:
                layers0_sources.add(manifest.sources[key]["source_checkpoint"])
        assert len(layers0_sources) == 1, f"layers.0 params came from multiple sources: {layers0_sources}"

        # Merged should have no NaN
        for key, tensor in merged.items():
            assert not torch.isnan(tensor).any(), f"NaN in merged {key}"

    def test_module_level_single_checkpoint(self, tmp_path):
        """Module-level graft with one checkpoint should work."""
        sd = _make_healthy_sd()
        path = tmp_path / "only.pt"
        torch.save(sd, str(path))

        merged, manifest = graft_modules([str(path)])
        assert len(merged) == len(sd)

    def test_module_level_false_falls_back(self, tmp_path):
        """module_level=False should fall back to per-param graft."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        merged, manifest = graft_modules([str(path_a), str(path_b)], module_level=False)
        assert len(merged) > 0


# ── GraftManifest tests ─────────────────────────────────────────────────


class TestGraftManifest:
    """Tests for the GraftManifest dataclass."""

    def test_to_json_serializable(self):
        """to_json should return a JSON-serializable dict."""
        m = GraftManifest(
            sources={
                "layer.weight": {
                    "source_checkpoint": "/tmp/a.pt",
                    "score": 95.0,
                    "runner_up_score": 80.0,
                }
            },
            summary={
                "total_params": 1,
                "params_from_each_source": {"/tmp/a.pt": 1},
                "overall_score_before": {"/tmp/a.pt": 90.0},
                "overall_score_after": 95.0,
            },
        )
        j = m.to_json()
        # Should be serializable
        serialized = json.dumps(j)
        parsed = json.loads(serialized)
        assert parsed["sources"]["layer.weight"]["score"] == 95.0
        assert parsed["summary"]["total_params"] == 1

    def test_print_report(self, capsys):
        """print_report should produce readable output."""
        m = GraftManifest(
            sources={
                "layer.weight": {
                    "source_checkpoint": "a.pt",
                    "score": 95.0,
                    "runner_up_score": 70.0,
                }
            },
            summary={
                "total_params": 1,
                "params_from_each_source": {"a.pt": 1},
                "overall_score_before": {"a.pt": 90.0},
                "overall_score_after": 95.0,
            },
        )
        m.print_report()
        captured = capsys.readouterr()
        assert "Graft Report" in captured.out
        assert "a.pt" in captured.out
        assert "95" in captured.out

    def test_empty_manifest(self):
        """Empty manifest should serialize without error."""
        m = GraftManifest()
        j = m.to_json()
        assert j == {"sources": {}, "summary": {}}


# ── CLI tests ────────────────────────────────────────────────────────────


class TestGraftCLI:
    """Tests for the graft CLI command."""

    def test_graft_cli_basic(self, tmp_path):
        """model-clinic graft should merge two checkpoints."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        out_path = tmp_path / "merged.pt"
        manifest_path = tmp_path / "manifest.json"

        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        result = subprocess.run(
            [
                sys.executable, "-m", "model_clinic.cli", "graft",
                str(path_a), str(path_b),
                "-o", str(out_path),
                "--manifest", str(manifest_path),
            ],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert out_path.exists(), "Merged checkpoint not created"
        assert manifest_path.exists(), "Manifest not created"

        # Verify merged checkpoint loads
        merged = torch.load(str(out_path), weights_only=True)
        assert len(merged) == len(sd_a)

        # Verify manifest is valid JSON
        with open(manifest_path) as f:
            m = json.load(f)
        assert "sources" in m
        assert "summary" in m

    def test_graft_cli_json_output(self, tmp_path):
        """model-clinic graft --json should output JSON manifest."""
        sd_a = _make_healthy_sd()
        sd_b = _make_healthy_sd()

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        out_path = tmp_path / "merged.pt"

        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        result = subprocess.run(
            [
                sys.executable, "-m", "model_clinic.cli", "graft",
                str(path_a), str(path_b),
                "-o", str(out_path),
                "--json",
            ],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        data = json.loads(result.stdout)
        assert "sources" in data
        assert "summary" in data

    def test_graft_cli_module_level(self, tmp_path):
        """model-clinic graft --module-level should work."""
        sd_a = _make_damaged_sd_a()
        sd_b = _make_damaged_sd_b()

        path_a = tmp_path / "a.pt"
        path_b = tmp_path / "b.pt"
        out_path = tmp_path / "merged.pt"

        torch.save(sd_a, str(path_a))
        torch.save(sd_b, str(path_b))

        result = subprocess.run(
            [
                sys.executable, "-m", "model_clinic.cli", "graft",
                str(path_a), str(path_b),
                "-o", str(out_path),
                "--module-level",
            ],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert out_path.exists()
