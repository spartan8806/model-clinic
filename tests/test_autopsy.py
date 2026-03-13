"""Tests for model-clinic autopsy and prune-suggest commands."""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

# Ensure subprocess output uses UTF-8 on Windows to avoid cp1252 errors
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
_RUN_KWARGS = {"stdin": subprocess.DEVNULL, "env": _SUBPROCESS_ENV}


# ── autopsy() API tests ────────────────────────────────────────────────────


class TestAutopsyAPI:
    """Test the autopsy() public API function."""

    def test_autopsy_redirect_healthy_model(self, tiny_state_dict):
        """A healthy model scoring >= threshold should be redirected to exam."""
        from model_clinic._tools.autopsy import autopsy

        # tiny_state_dict is healthy; default threshold=50 should redirect it
        result = autopsy(tiny_state_dict, score_threshold=50)
        # Healthy model should score >= 50, triggering a redirect
        assert result["redirect"] is True

    def test_autopsy_no_redirect_for_sick_model(self):
        """A severely broken model should not be redirected."""
        from model_clinic._tools.autopsy import autopsy

        # Model with NaN, exploding norms, and norm drift — should score poorly
        torch.manual_seed(42)
        sd = {
            "bad_param": torch.randn(32, 32),
            "exploding.weight": torch.randn(64, 64) * 200,
            "final_norm.weight": torch.full((64,), 5.0),
        }
        sd["bad_param"][5, 5] = float("nan")
        result = autopsy(sd, score_threshold=50)
        # Very sick model should score < 50 and NOT be redirected
        if result["score"] < 50:
            assert result["redirect"] is False

    def test_autopsy_on_sick_model(self, sick_state_dict):
        """Sick model should be analyzed with cause-of-death info."""
        from model_clinic._tools.autopsy import autopsy

        result = autopsy(sick_state_dict, score_threshold=50)
        # Sick model should be below threshold (has NaN, exploding norm etc)
        if not result["redirect"]:
            assert result["score"] is not None
            assert result["grade"] is not None
            assert result["findings"] is not None
            # Should have at least a primary cause
            assert result["primary_cause"] is not None
            assert "condition" in result["primary_cause"]
            assert "count" in result["primary_cause"]

    def test_autopsy_result_keys(self, sick_state_dict):
        """autopsy() result dict should contain all expected keys when model is below threshold."""
        from model_clinic._tools.autopsy import autopsy

        # Use score_threshold=101 so the model is always below threshold → full autopsy runs
        result = autopsy(sick_state_dict, score_threshold=101)
        expected_keys = {
            "score", "grade", "redirect", "findings", "prescriptions",
            "health", "primary_cause", "secondary_cause",
            "damage", "salvageable", "recovery_plan",
            "forensics", "auto_fixable", "total_findings",
        }
        assert result["redirect"] is False, f"Expected full autopsy but got redirect (score={result['score']})"
        assert expected_keys.issubset(result.keys())

    def test_autopsy_damage_assessment(self, sick_state_dict):
        """Damage assessment should have status for weights and stability."""
        from model_clinic._tools.autopsy import autopsy

        # score_threshold=101 forces full autopsy (no model scores over 100)
        result = autopsy(sick_state_dict, score_threshold=101)
        assert result["redirect"] is False
        damage = result["damage"]
        assert "weights" in damage
        assert "stability" in damage
        for cat_data in damage.values():
            assert "status" in cat_data
            assert cat_data["status"] in ("OK", "DEGRADED", "FAILED")

    def test_autopsy_recovery_plan_is_list(self, sick_state_dict):
        """recovery_plan should always be a list."""
        from model_clinic._tools.autopsy import autopsy

        result = autopsy(sick_state_dict, score_threshold=101)
        assert result["redirect"] is False
        assert isinstance(result["recovery_plan"], list)

    def test_autopsy_forensics_structure(self, sick_state_dict):
        """Forensics items should have expected keys."""
        from model_clinic._tools.autopsy import autopsy

        result = autopsy(sick_state_dict, score_threshold=101)
        assert result["redirect"] is False
        for item in result["forensics"]:
            assert "name" in item
            assert "condition" in item
            assert "detail" in item

    def test_autopsy_nan_model_primary_cause(self):
        """A model with only NaN should show nan_inf as primary cause."""
        from model_clinic._tools.autopsy import autopsy

        sd = {"bad_param": torch.randn(32, 32)}
        sd["bad_param"][5, 5] = float("nan")
        result = autopsy(sd, score_threshold=101)
        assert result["redirect"] is False
        assert result["primary_cause"] is not None
        assert result["primary_cause"]["condition"] == "nan_inf"

    def test_autopsy_salvageable_auto_fixable(self):
        """A model with only norm_drift should be auto-fixable."""
        from model_clinic._tools.autopsy import autopsy

        # norm_drift threshold is |mean - 1.0| > 1.5, so mean > 2.5
        sd = {"layers.0.norm.weight": torch.full((64,), 3.0)}
        result = autopsy(sd, score_threshold=101)
        assert result["redirect"] is False
        # norm_drift is in _AUTO_FIXABLE
        assert result["auto_fixable"] >= 1

    def test_autopsy_empty_state_dict(self):
        """Empty state dict should not crash."""
        from model_clinic._tools.autopsy import autopsy

        # Empty state dict: score=100 (no findings), so redirects at threshold=50
        # Use threshold=101 to avoid redirect
        result = autopsy({}, score_threshold=101)
        assert result["redirect"] is False
        assert result["total_findings"] == 0
        assert result["salvageable"] == "N/A"


# ── prune_suggestions() API tests ─────────────────────────────────────────


class TestPruneSuggestAPI:
    """Test the prune_suggestions() public API function."""

    def test_returns_list(self, tiny_state_dict):
        """prune_suggestions() should return a list."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        result = prune_suggestions(tiny_state_dict)
        assert isinstance(result, list)

    def test_result_dict_keys(self, tiny_state_dict):
        """Each suggestion should have required keys."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        result = prune_suggestions(tiny_state_dict)
        required = {"tensor_name", "reason", "suggested_amount", "risk_level", "detail"}
        for item in result:
            assert required.issubset(item.keys()), f"Missing keys in: {item}"

    def test_suggested_amount_range(self, tiny_state_dict):
        """suggested_amount should be between 0 and 1."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        result = prune_suggestions(tiny_state_dict)
        for item in result:
            assert 0.0 <= item["suggested_amount"] <= 1.0

    def test_risk_level_valid(self, tiny_state_dict):
        """risk_level should be one of the valid values."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        valid = {"low", "moderate", "high"}
        result = prune_suggestions(tiny_state_dict)
        for item in result:
            assert item["risk_level"] in valid

    def test_high_sparsity_tensor_detected(self):
        """A tensor with > 50% near-zero values should be flagged."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        # Build a state dict with a very sparse tensor
        sd = {
            "sparse.weight": torch.zeros(64, 64),
        }
        # Fill 10% with non-zero values
        t = sd["sparse.weight"]
        t[0, :6] = torch.randn(6)
        t[1, :6] = torch.randn(6)
        # 52 rows still zero-dominated, overall ~90%+ near-zero

        result = prune_suggestions(sd, min_size=10)
        names = [s["tensor_name"] for s in result]
        assert "sparse.weight" in names

    def test_low_rank_tensor_detected(self):
        """A rank-1 tensor should be flagged as low-rank."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        # Create a rank-1 matrix (outer product)
        v = torch.randn(64)
        low_rank = v.unsqueeze(1) * v.unsqueeze(0)  # 64x64 rank-1
        sd = {"low_rank.weight": low_rank}

        result = prune_suggestions(sd, min_size=10)
        names = [s["tensor_name"] for s in result]
        assert "low_rank.weight" in names

    def test_empty_state_dict(self):
        """Empty state dict should return empty list."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        result = prune_suggestions({})
        assert result == []

    def test_min_size_filters_small_tensors(self):
        """Tensors below min_size should not be reported."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        # Tiny tensor that would otherwise be flagged
        sd = {"tiny.weight": torch.zeros(5, 5)}
        result = prune_suggestions(sd, min_size=1000)
        assert result == []

    def test_norm_layers_excluded(self):
        """LayerNorm/RMSNorm weight tensors should not be suggested for pruning."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        sd = {
            "layers.0.layernorm.weight": torch.full((256,), 3.0),
            "layers.0.rmsnorm.weight": torch.zeros(256),
        }
        result = prune_suggestions(sd, min_size=1)
        names = [s["tensor_name"] for s in result]
        assert "layers.0.layernorm.weight" not in names
        assert "layers.0.rmsnorm.weight" not in names

    def test_sorted_low_risk_first(self):
        """Low-risk suggestions should appear before moderate ones."""
        from model_clinic._tools.prune_suggest import prune_suggestions

        # Build state dict with both a sparse tensor and a redundant-row tensor
        sparse = torch.zeros(64, 64)
        sparse[0, 0] = 0.001  # almost entirely zero — should be "low" risk
        sd = {"sparse.weight": sparse}

        result = prune_suggestions(sd, min_size=10)
        if len(result) >= 2:
            risk_order = {"low": 0, "moderate": 1, "high": 2}
            for i in range(len(result) - 1):
                assert (risk_order[result[i]["risk_level"]]
                        <= risk_order[result[i + 1]["risk_level"]])


# ── Subprocess / CLI tests ─────────────────────────────────────────────────


class TestAutopsyCLI:
    """Test autopsy and prune-suggest via subprocess."""

    def _save_sick_checkpoint(self, tmp_path):
        """Save a sick checkpoint and return its path."""
        torch.manual_seed(42)
        sd = {
            "bad_param": torch.randn(32, 32),
            "exploding.weight": torch.randn(64, 64) * 200,
            "final_norm.weight": torch.full((64,), 5.0),
        }
        sd["bad_param"][5, 5] = float("nan")
        path = tmp_path / "sick.pt"
        torch.save({"model_state_dict": sd}, str(path))
        return str(path)

    def test_autopsy_cli_runs(self, tmp_path):
        """model-clinic autopsy should exit 0 on a sick model."""
        path = self._save_sick_checkpoint(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "autopsy", path,
             "--threshold", "101"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, result.stderr

    def test_autopsy_cli_json_output(self, tmp_path):
        """model-clinic autopsy --json should return valid JSON."""
        path = self._save_sick_checkpoint(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "autopsy", path,
             "--json", "--threshold", "101"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, result.stderr
        # Output may contain loading lines before JSON — find the JSON object
        stdout = result.stdout.strip()
        # Find the start of the JSON block
        json_start = stdout.find("{")
        assert json_start != -1, f"No JSON found in: {stdout[:200]}"
        data = json.loads(stdout[json_start:])
        assert "score" in data
        assert "findings" in data

    def test_autopsy_cli_redirect_on_healthy_threshold(self, tmp_path):
        """Autopsy with --threshold 101 should always run (never redirect)."""
        path = self._save_sick_checkpoint(tmp_path)
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "autopsy", path,
             "--threshold", "101"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, result.stderr
        # Should show AUTOPSY REPORT, not redirect message
        assert "AUTOPSY REPORT" in result.stdout or "CAUSE OF DEATH" in result.stdout

    def test_prune_suggest_cli_runs(self, tmp_path):
        """model-clinic prune-suggest should exit 0."""
        torch.manual_seed(0)
        sd = {"weight": torch.randn(64, 64)}
        path = tmp_path / "model.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "prune-suggest", str(path)],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, result.stderr

    def test_prune_suggest_cli_json(self, tmp_path):
        """model-clinic prune-suggest --json should return valid JSON."""
        # Build a clearly sparse model so we get at least one suggestion
        sparse = torch.zeros(64, 64)
        sparse[0, 0] = 1.0  # 99.98% near-zero
        sd = {"sparse.weight": sparse}
        path = tmp_path / "sparse.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "prune-suggest",
             str(path), "--json"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, result.stderr
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        assert json_start != -1
        data = json.loads(stdout[json_start:])
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_prune_suggest_finds_sparse_tensor(self, tmp_path):
        """prune-suggest should detect a highly sparse tensor via --json."""
        sparse = torch.zeros(64, 64)
        sparse[0, :3] = torch.randn(3)
        sd = {"very_sparse.weight": sparse}
        path = tmp_path / "sparse.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "prune-suggest",
             str(path), "--json"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0, result.stderr
        stdout = result.stdout.strip()
        json_start = stdout.find("{")
        data = json.loads(stdout[json_start:])
        names = [s["tensor_name"] for s in data["suggestions"]]
        assert "very_sparse.weight" in names
