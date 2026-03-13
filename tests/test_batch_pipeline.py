"""Tests for batch processing and treatment pipelines."""

import os
import tempfile

import pytest
import torch

from model_clinic.clinic import examine_batch, create_pipeline, TreatmentPipeline
from model_clinic._types import ExamResult, PipelineResult


class TestExamineBatch:
    """Test batch examination of multiple models."""

    def test_batch_two_models(self, tmp_path):
        """Examine two valid .pt files."""
        torch.manual_seed(42)

        p1 = str(tmp_path / "model_a.pt")
        p2 = str(tmp_path / "model_b.pt")
        torch.save({"layer.weight": torch.randn(32, 32)}, p1)
        torch.save({"layer.weight": torch.randn(64, 64)}, p2)

        results = examine_batch([p1, p2])
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ExamResult)
            assert r.error is None
            assert r.meta is not None
            assert r.health_score is not None

    def test_batch_parallel(self, tmp_path):
        """Parallel mode should produce the same results."""
        torch.manual_seed(42)

        p1 = str(tmp_path / "model_a.pt")
        p2 = str(tmp_path / "model_b.pt")
        torch.save({"layer.weight": torch.randn(32, 32)}, p1)
        torch.save({"layer.weight": torch.randn(64, 64)}, p2)

        results = examine_batch([p1, p2], parallel=True)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, ExamResult)
            assert r.error is None

    def test_batch_bad_path(self, tmp_path):
        """A bad path should capture the error, not crash."""
        good = str(tmp_path / "good.pt")
        torch.save({"w": torch.randn(16, 16)}, good)

        bad = str(tmp_path / "nonexistent.pt")
        results = examine_batch([good, bad])
        assert len(results) == 2

        # Good model should succeed
        assert results[0].error is None
        assert results[0].meta is not None

        # Bad model should have an error string
        assert results[1].error is not None
        assert len(results[1].error) > 0
        assert results[1].path == bad


class TestTreatmentPipeline:
    """Test treatment pipeline creation and execution."""

    def test_create_pipeline(self):
        """create_pipeline returns a TreatmentPipeline."""
        pipe = create_pipeline([("dead_neurons", {}), ("norm_drift", {})])
        assert isinstance(pipe, TreatmentPipeline)
        assert len(pipe.steps) == 2

    def test_pipeline_on_sick_model(self, sick_state_dict):
        """Pipeline should find and treat dead_neurons + norm_drift."""
        pipe = create_pipeline([
            ("dead_neurons", {}),
            ("norm_drift", {}),
        ])
        result = pipe.run(sick_state_dict)

        assert isinstance(result, PipelineResult)
        assert result.health_before is not None
        assert result.health_after is not None

        # Should have found relevant conditions
        conditions = {f.condition for f in result.findings}
        assert "dead_neurons" in conditions or "norm_drift" in conditions

        # Should have generated prescriptions and treatments
        assert len(result.prescriptions) > 0
        assert len(result.treatments) > 0

    def test_pipeline_dry_run(self, sick_state_dict):
        """Dry run should not modify the state dict."""
        original_norm = sick_state_dict["final_norm.weight"].clone()

        pipe = create_pipeline([("norm_drift", {})])
        result = pipe.run(sick_state_dict, dry_run=True)

        assert isinstance(result, PipelineResult)
        # Tensor should be unchanged
        assert torch.allclose(sick_state_dict["final_norm.weight"], original_norm)

        # Treatments should all be dry-run
        for t in result.treatments:
            assert "DRY RUN" in t.description

    def test_pipeline_describe(self, capsys):
        """describe() should print the pipeline steps."""
        pipe = create_pipeline([
            ("dead_neurons", {"scale": 0.01}),
            ("norm_drift", {}),
        ])
        pipe.describe()
        captured = capsys.readouterr()
        assert "dead_neurons" in captured.out
        assert "norm_drift" in captured.out
        assert "scale" in captured.out

    def test_pipeline_conservative(self, sick_state_dict):
        """Conservative mode should only apply low-risk treatments."""
        pipe = create_pipeline([
            ("dead_neurons", {}),
            ("norm_drift", {}),
            ("nan_inf", {}),
        ])
        result = pipe.run(sick_state_dict, conservative=True)
        for rx in result.prescriptions:
            assert rx.risk == "low"
