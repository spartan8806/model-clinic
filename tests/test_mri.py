"""Tests for Model MRI -- deep per-layer weight analysis."""

import subprocess
import sys
import json
import math

import torch
import pytest

from model_clinic._mri import model_mri, mri_summary, analyze_layer, LayerMRI


class TestAnalyzeLayer:
    """Test single-layer analysis."""

    def test_basic_random_matrix(self):
        torch.manual_seed(42)
        t = torch.randn(64, 64)
        result = analyze_layer("test.weight", t)
        assert isinstance(result, LayerMRI)
        assert result.name == "test.weight"
        assert result.shape == (64, 64)
        assert "float32" in result.dtype

    def test_effective_rank_in_range(self):
        torch.manual_seed(42)
        t = torch.randn(32, 16)
        result = analyze_layer("layer.weight", t)
        k = min(32, 16)
        assert 1 <= result.effective_rank <= k
        assert result.effective_rank > 0

    def test_numerical_rank_in_range(self):
        torch.manual_seed(42)
        t = torch.randn(50, 30)
        result = analyze_layer("layer.weight", t)
        k = min(50, 30)
        assert 0 <= result.numerical_rank <= k

    def test_rank_utilization_in_range(self):
        torch.manual_seed(42)
        t = torch.randn(64, 64)
        result = analyze_layer("layer.weight", t)
        assert 0.0 <= result.rank_utilization <= 1.0

    def test_entropy_in_range(self):
        torch.manual_seed(42)
        t = torch.randn(32, 32)
        result = analyze_layer("layer.weight", t)
        assert 0.0 <= result.entropy <= 1.0

    def test_near_rank_1_matrix_is_low_rank(self):
        # Create a near-rank-1 matrix: outer product
        u = torch.randn(64, 1)
        v = torch.randn(1, 64)
        t = u @ v  # rank 1
        result = analyze_layer("low_rank.weight", t)
        assert result.is_low_rank is True
        assert result.numerical_rank <= 2

    def test_zero_matrix_is_degenerate(self):
        t = torch.zeros(32, 32)
        result = analyze_layer("zero.weight", t)
        assert result.is_degenerate is True
        assert result.numerical_rank == 0
        assert result.effective_rank == 0.0

    def test_identity_matrix(self):
        t = torch.eye(64)
        result = analyze_layer("identity.weight", t)
        # Identity has full rank
        assert result.numerical_rank == 64
        assert result.rank_utilization == 1.0
        assert not result.is_low_rank
        assert not result.is_degenerate

    def test_sparse_matrix_detected(self):
        t = torch.zeros(64, 64)
        # Make only a few entries non-zero
        t[0, 0] = 1.0
        t[10, 5] = 2.0
        result = analyze_layer("sparse.weight", t)
        assert result.is_sparse is True
        assert result.sparsity > 0.5

    def test_heavy_tailed_detection(self):
        torch.manual_seed(42)
        # Create heavy-tailed distribution (most near 0, some extreme)
        t = torch.zeros(100, 100)
        t[0, 0] = 1000.0
        t[1, 1] = -1000.0
        result = analyze_layer("heavy.weight", t)
        assert result.kurtosis > 10.0
        assert result.is_heavy_tailed is True

    def test_3d_tensor(self):
        torch.manual_seed(42)
        t = torch.randn(8, 32, 16)
        result = analyze_layer("conv.weight", t)
        assert result.shape == (8, 32, 16)
        # Should still compute metrics (reshaped to 2D)
        assert result.effective_rank > 0

    def test_role_inference_attention(self):
        t = torch.randn(64, 64)
        result = analyze_layer("model.layers.0.self_attn.q_proj.weight", t)
        assert result.inferred_role == "attention"

    def test_role_inference_mlp_gate(self):
        t = torch.randn(256, 64)
        result = analyze_layer("model.layers.0.mlp.gate_proj.weight", t)
        assert result.inferred_role == "mlp_gate"

    def test_role_inference_embedding(self):
        t = torch.randn(100, 64)
        result = analyze_layer("embed_tokens.weight", t)
        assert result.inferred_role == "embedding"

    def test_role_inference_unknown(self):
        t = torch.randn(64, 64)
        result = analyze_layer("something.arbitrary.weight", t)
        assert result.inferred_role == "unknown"


class TestModelMRI:
    """Test model_mri on full state dicts."""

    def test_runs_on_tiny_state_dict(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        assert isinstance(results, list)
        assert len(results) > 0
        # Should only analyze 2D+ tensors
        n_2d = sum(1 for v in tiny_state_dict.values()
                   if isinstance(v, torch.Tensor) and v.dim() >= 2)
        assert len(results) == n_2d

    def test_sorted_by_name(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        names = [r.name for r in results]
        assert names == sorted(names)

    def test_max_layers_limits(self, tiny_state_dict):
        results = model_mri(tiny_state_dict, max_layers=3)
        assert len(results) == 3

    def test_all_results_are_layer_mri(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        for r in results:
            assert isinstance(r, LayerMRI)

    def test_effective_rank_range(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        for r in results:
            k = min(r.shape[0], r.shape[-1]) if len(r.shape) >= 2 else r.shape[0]
            assert 1 <= r.effective_rank <= k, (
                f"{r.name}: effective_rank={r.effective_rank} not in [1, {k}]"
            )

    def test_skips_1d_tensors(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        # The tiny_state_dict has 1D norm weights; they should be excluded
        result_names = {r.name for r in results}
        for name, tensor in tiny_state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.dim() < 2:
                assert name not in result_names

    def test_empty_state_dict(self):
        results = model_mri({})
        assert results == []


class TestMRISummary:
    """Test mri_summary aggregation."""

    def test_returns_expected_keys(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        summary = mri_summary(results)
        expected_keys = {
            "total_layers", "analyzed_layers",
            "mean_rank_utilization", "median_rank_utilization",
            "n_low_rank", "n_degenerate", "n_heavy_tailed", "n_sparse",
            "role_distribution", "information_score",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_information_score_range(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        summary = mri_summary(results)
        assert 0 <= summary["information_score"] <= 100

    def test_role_distribution_sums(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        summary = mri_summary(results)
        total_roles = sum(summary["role_distribution"].values())
        assert total_roles == summary["analyzed_layers"]

    def test_empty_results(self):
        summary = mri_summary([])
        assert summary["total_layers"] == 0
        assert summary["information_score"] == 0

    def test_counts_match(self, tiny_state_dict):
        results = model_mri(tiny_state_dict)
        summary = mri_summary(results)
        assert summary["n_low_rank"] == sum(1 for r in results if r.is_low_rank)
        assert summary["n_degenerate"] == sum(1 for r in results if r.is_degenerate)
        assert summary["n_heavy_tailed"] == sum(1 for r in results if r.is_heavy_tailed)
        assert summary["n_sparse"] == sum(1 for r in results if r.is_sparse)


class TestCLI:
    """Test the MRI CLI command."""

    def test_cli_runs(self, tmp_path, tiny_state_dict):
        # Save state dict to temp file
        pt_path = str(tmp_path / "test_model.pt")
        torch.save(tiny_state_dict, pt_path)

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic._tools.mri", pt_path],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
        assert result.returncode == 0
        assert "Model MRI" in result.stdout
        assert "rank=" in result.stdout

    def test_cli_json(self, tmp_path, tiny_state_dict):
        pt_path = str(tmp_path / "test_model.pt")
        torch.save(tiny_state_dict, pt_path)

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic._tools.mri", pt_path, "--json"],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "summary" in data
        assert "layers" in data
        assert len(data["layers"]) > 0

    def test_cli_top(self, tmp_path, tiny_state_dict):
        pt_path = str(tmp_path / "test_model.pt")
        torch.save(tiny_state_dict, pt_path)

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic._tools.mri", pt_path, "--top", "3"],
            capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL,
        )
        assert result.returncode == 0
        assert "Summary" in result.stdout


class TestExports:
    """Test that MRI symbols are exported from model_clinic."""

    def test_model_mri_exported(self):
        from model_clinic import model_mri as mri_fn
        assert callable(mri_fn)

    def test_layer_mri_exported(self):
        from model_clinic import LayerMRI as LM
        assert LM is not None

    def test_mri_summary_exported(self):
        from model_clinic import mri_summary as ms_fn
        assert callable(ms_fn)
