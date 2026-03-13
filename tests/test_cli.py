"""Tests for the CLI entry points."""

import os
import subprocess
import sys
import torch
import json

# Ensure subprocess output uses UTF-8 on Windows to avoid cp1252 encoding errors
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}

# Common kwargs for subprocess.run — stdin=DEVNULL avoids WinError 6 (invalid handle)
# that occurs when pytest captures stdin in ways subprocess can't inherit on Windows.
_RUN_KWARGS = {"stdin": subprocess.DEVNULL, "env": _SUBPROCESS_ENV}


class TestCLI:
    """Test CLI commands are callable and produce expected output."""

    def test_clinic_help(self):
        """model-clinic --help should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "--help"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert "model-clinic" in result.stdout or "Diagnose" in result.stdout

    def test_exam_on_checkpoint(self, tmp_path):
        """model-clinic exam should diagnose a checkpoint."""
        sd = {"weight": torch.randn(64, 64), "norm.weight": torch.ones(64)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "exam", str(path), "--json"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "findings" in data
        assert "meta" in data

    def test_exam_on_sick_checkpoint(self, tmp_path):
        """model-clinic exam should find issues in sick model."""
        sd = {"weight": torch.randn(64, 64), "norm.weight": torch.full((64,), 3.0)}
        sd["weight"][0, 0] = float("nan")
        path = tmp_path / "sick.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "exam", str(path), "--json"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        conditions = {f["condition"] for f in data["findings"]}
        assert "nan_inf" in conditions

    def test_treat_dry_run(self, tmp_path):
        """model-clinic treat --dry-run should not modify checkpoint."""
        sd = {"weight": torch.randn(64, 64)}
        sd["weight"][0, 0] = float("nan")
        path = tmp_path / "sick.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "treat", str(path),
             "--dry-run", "--quiet"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0

    def test_treat_and_save(self, tmp_path):
        """model-clinic treat --save should create treated checkpoint."""
        sd = {"norm.weight": torch.full((64,), 3.0)}
        path = tmp_path / "sick.pt"
        out_path = tmp_path / "treated.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "treat", str(path),
             "--save", str(out_path), "--quiet"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert out_path.exists()

    def test_no_command_shows_help(self):
        """Running with no subcommand should show help."""
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert "exam" in result.stdout or "treat" in result.stdout

    def test_verbose_flag_accepted_exam(self, tmp_path):
        """--verbose / -v flag should be accepted by exam."""
        sd = {"weight": torch.randn(64, 64)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "exam", str(path), "--verbose"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert "Checking" in result.stdout

    def test_verbose_short_flag_accepted_exam(self, tmp_path):
        """-v flag should be accepted by exam."""
        sd = {"weight": torch.randn(64, 64)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "exam", str(path), "-v"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert "Checking" in result.stdout

    def test_verbose_flag_accepted_treat(self, tmp_path):
        """--verbose flag should be accepted by treat."""
        sd = {"norm.weight": torch.full((64,), 3.0)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "treat", str(path),
             "--verbose", "--dry-run"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert "Checking" in result.stdout

    def test_verbose_treat_shows_applying_count(self, tmp_path):
        """--verbose during treat should show [N/M] Applying ... messages."""
        sd = {"norm.weight": torch.full((64,), 3.0)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "treat", str(path),
             "--verbose", "--dry-run"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
        assert "Applying" in result.stdout

    def test_example_prompts_flag_accepted_exam(self, tmp_path):
        """--example-prompts flag should be accepted by exam."""
        sd = {"weight": torch.randn(64, 64)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "exam", str(path),
             "--example-prompts"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0

    def test_example_prompts_flag_accepted_treat(self, tmp_path):
        """--example-prompts flag should be accepted by treat."""
        sd = {"weight": torch.randn(64, 64)}
        path = tmp_path / "test.pt"
        torch.save({"model_state_dict": sd}, str(path))

        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "treat", str(path),
             "--example-prompts", "--dry-run"],
            capture_output=True, text=True, timeout=180, **_RUN_KWARGS,
        )
        assert result.returncode == 0
