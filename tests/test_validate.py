"""Tests for the validate tool."""

import json
import subprocess
import sys

import pytest
import torch

from model_clinic._tools.validate import validate


@pytest.fixture
def healthy_checkpoint(tmp_path, tiny_state_dict):
    """Save tiny_state_dict as a .pt file."""
    path = tmp_path / "healthy.pt"
    torch.save(tiny_state_dict, path)
    return str(path)


@pytest.fixture
def sick_checkpoint(tmp_path, sick_state_dict):
    """Save sick_state_dict as a .pt file (contains NaN)."""
    path = tmp_path / "sick.pt"
    torch.save(sick_state_dict, path)
    return str(path)


@pytest.fixture
def empty_tensor_checkpoint(tmp_path):
    """Checkpoint with an empty tensor."""
    sd = {
        "good": torch.randn(4, 4),
        "empty": torch.zeros(0, 4),
    }
    path = tmp_path / "empty.pt"
    torch.save(sd, path)
    return str(path)


class TestValidateHealthy:
    """Tests against a healthy checkpoint."""

    def test_all_pass(self, healthy_checkpoint):
        results = validate(healthy_checkpoint)
        statuses = {r["check"]: r["status"] for r in results}
        assert statuses["load"] == "PASS"
        assert statuses["integrity"] == "PASS"
        assert statuses["shapes"] == "PASS"
        assert statuses["dtypes"] == "INFO"

    def test_load_reports_tensors(self, healthy_checkpoint):
        results = validate(healthy_checkpoint)
        load_result = results[0]
        assert load_result["details"]["tensors"] > 0
        assert load_result["details"]["parameters"] > 0
        assert load_result["details"]["size_bytes"] > 0

    def test_no_failures(self, healthy_checkpoint):
        results = validate(healthy_checkpoint)
        failures = [r for r in results if r["status"] == "FAIL"]
        assert len(failures) == 0

    def test_dtype_distribution(self, healthy_checkpoint):
        results = validate(healthy_checkpoint)
        dtype_result = next(r for r in results if r["check"] == "dtypes")
        assert "distribution" in dtype_result["details"]
        assert len(dtype_result["details"]["distribution"]) > 0


class TestValidateSick:
    """Tests against a checkpoint with NaN values."""

    def test_integrity_fails(self, sick_checkpoint):
        results = validate(sick_checkpoint)
        integrity = next(r for r in results if r["check"] == "integrity")
        assert integrity["status"] == "FAIL"
        assert "NaN" in integrity["message"] or "Inf" in integrity["message"]

    def test_bad_tensor_listed(self, sick_checkpoint):
        results = validate(sick_checkpoint)
        integrity = next(r for r in results if r["check"] == "integrity")
        bad_names = [bt["name"] for bt in integrity["details"]["bad_tensors"]]
        assert "bad_param" in bad_names

    def test_load_still_passes(self, sick_checkpoint):
        results = validate(sick_checkpoint)
        load_result = results[0]
        assert load_result["status"] == "PASS"


class TestValidateEdgeCases:
    """Tests for edge cases."""

    def test_missing_file(self, tmp_path):
        results = validate(str(tmp_path / "nonexistent.pt"))
        assert results[0]["status"] == "FAIL"
        # Should stop after load failure
        assert len(results) == 1

    def test_empty_tensor(self, empty_tensor_checkpoint):
        results = validate(empty_tensor_checkpoint)
        shapes = next(r for r in results if r["check"] == "shapes")
        assert shapes["status"] == "FAIL"
        assert "empty" in shapes["details"]["bad_shapes"][0]["name"]

    def test_generate_requires_hf(self):
        """generate=True without hf should skip."""
        # The validate function itself doesn't enforce this (the CLI does),
        # but the generate check should indicate it needs --hf.
        results = validate("dummy.pt", hf=False, generate=True)
        # It will fail at load since dummy.pt doesn't exist
        assert results[0]["status"] == "FAIL"

    def test_single_scalar_tensor(self, tmp_path):
        """Scalar tensors should be valid."""
        sd = {"gate": torch.tensor(0.5)}
        path = tmp_path / "scalar.pt"
        torch.save(sd, path)
        results = validate(str(path))
        failures = [r for r in results if r["status"] == "FAIL"]
        assert len(failures) == 0


class TestValidateJSON:
    """Test JSON output via the validate function."""

    def test_healthy_json_structure(self, healthy_checkpoint):
        results = validate(healthy_checkpoint)
        # Simulate what --json would produce
        failures = sum(1 for r in results if r["status"] == "FAIL")
        output = {
            "model": healthy_checkpoint,
            "valid": failures == 0,
            "checks": results,
        }
        # Should be JSON-serializable
        serialized = json.dumps(output, default=str)
        parsed = json.loads(serialized)
        assert parsed["valid"] is True
        assert len(parsed["checks"]) >= 4

    def test_sick_json_structure(self, sick_checkpoint):
        results = validate(sick_checkpoint)
        failures = sum(1 for r in results if r["status"] == "FAIL")
        output = {
            "model": sick_checkpoint,
            "valid": failures == 0,
            "checks": results,
        }
        serialized = json.dumps(output, default=str)
        parsed = json.loads(serialized)
        assert parsed["valid"] is False
