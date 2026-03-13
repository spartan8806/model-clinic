"""Tests for wandb, mlflow, and TensorBoard integration callbacks."""

import sys
import types
import warnings
from unittest.mock import MagicMock, patch

import pytest

from model_clinic._types import Finding, HealthScore
from model_clinic._integrations import (
    WandbCallback,
    MLflowCallback,
    TensorBoardCallback,
    log_health_to_wandb,
    log_health_to_mlflow,
    _build_metrics,
    _build_finding_counts,
    _grade_to_numeric,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_health(overall=82, grade="B") -> HealthScore:
    return HealthScore(
        overall=overall,
        grade=grade,
        categories={"weights": 90, "stability": 75, "activations": 95, "output": 85},
        summary="Test health score.",
    )


def _make_findings():
    return [
        Finding(condition="dead_neurons", severity="ERROR", param_name="layers.0.weight"),
        Finding(condition="norm_drift", severity="WARN", param_name="final_norm.weight"),
        Finding(condition="norm_drift", severity="WARN", param_name="layers.1.norm.weight"),
    ]


# ── _build_metrics ────────────────────────────────────────────────────────────

class TestBuildMetrics:

    def test_keys_present(self):
        health = _make_health()
        metrics = _build_metrics(health, [], "model_health")
        assert "model_health/overall_score" in metrics
        assert "model_health/grade_numeric" in metrics
        assert "model_health/weights_score" in metrics
        assert "model_health/stability_score" in metrics
        assert "model_health/activations_score" in metrics
        assert "model_health/output_score" in metrics
        assert "model_health/n_errors" in metrics
        assert "model_health/n_warnings" in metrics

    def test_overall_score_value(self):
        health = _make_health(overall=77)
        metrics = _build_metrics(health, [], "model_health")
        assert metrics["model_health/overall_score"] == 77

    def test_error_warning_counts(self):
        findings = _make_findings()
        metrics = _build_metrics(_make_health(), findings, "model_health")
        assert metrics["model_health/n_errors"] == 1
        assert metrics["model_health/n_warnings"] == 2

    def test_custom_prefix(self):
        health = _make_health()
        metrics = _build_metrics(health, [], "train/health")
        assert "train/health/overall_score" in metrics
        assert "model_health/overall_score" not in metrics

    def test_no_findings(self):
        health = _make_health()
        metrics = _build_metrics(health, [], "model_health")
        assert metrics["model_health/n_errors"] == 0
        assert metrics["model_health/n_warnings"] == 0


class TestBuildFindingCounts:

    def test_counts_per_condition(self):
        findings = _make_findings()
        counts = _build_finding_counts(findings, "model_health")
        assert counts["model_health/findings/dead_neurons"] == 1
        assert counts["model_health/findings/norm_drift"] == 2

    def test_empty_findings(self):
        counts = _build_finding_counts([], "model_health")
        assert counts == {}


class TestGradeToNumeric:

    def test_all_grades(self):
        assert _grade_to_numeric("A") == 5
        assert _grade_to_numeric("B") == 4
        assert _grade_to_numeric("C") == 3
        assert _grade_to_numeric("D") == 2
        assert _grade_to_numeric("F") == 1
        assert _grade_to_numeric("?") == 0


# ── WandbCallback ─────────────────────────────────────────────────────────────

class TestWandbCallback:

    def test_init_without_wandb(self):
        """Callback initializes cleanly even when wandb is not installed."""
        with patch.dict(sys.modules, {"wandb": None}):
            cb = WandbCallback(log_every=100)
        # Should not raise.
        assert cb.log_every == 100
        assert cb.log_findings is True

    def test_on_step_no_wandb_warns_once(self):
        """When wandb is missing, on_step warns exactly once."""
        cb = WandbCallback(log_every=1)
        cb._wandb = None  # Simulate missing package.
        cb._import_warned = False

        health = _make_health()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cb.on_step(0, health, [])
            cb.on_step(1, health, [])  # Second call should not warn again.

        wandb_warns = [w for w in caught if "wandb" in str(w.message).lower()]
        assert len(wandb_warns) == 1

    def test_on_step_respects_log_every(self):
        """on_step only logs when step is a multiple of log_every."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.run = mock_run

        cb = WandbCallback(log_every=10)
        cb._wandb = mock_wandb
        cb.run = mock_run

        health = _make_health()
        cb.on_step(1, health, [])   # Should NOT log (1 % 10 != 0).
        cb.on_step(5, health, [])   # Should NOT log.
        cb.on_step(10, health, [])  # Should log.

        assert mock_wandb.log.call_count == 1

    def test_on_step_logs_correct_keys(self):
        """on_step logs the expected metric keys."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.run = mock_run

        cb = WandbCallback(log_every=1, log_findings=False)
        cb._wandb = mock_wandb
        cb.run = mock_run

        health = _make_health(overall=88, grade="B")
        findings = _make_findings()
        cb.on_step(0, health, findings)

        assert mock_wandb.log.called
        logged_metrics = mock_wandb.log.call_args[0][0]
        assert "model_health/overall_score" in logged_metrics
        assert logged_metrics["model_health/overall_score"] == 88
        assert "model_health/n_errors" in logged_metrics

    def test_on_step_includes_findings_when_enabled(self):
        """log_findings=True adds per-condition counts."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.run = mock_run

        cb = WandbCallback(log_every=1, log_findings=True)
        cb._wandb = mock_wandb
        cb.run = mock_run

        findings = _make_findings()
        cb.on_step(0, _make_health(), findings)

        logged_metrics = mock_wandb.log.call_args[0][0]
        assert "model_health/findings/dead_neurons" in logged_metrics
        assert "model_health/findings/norm_drift" in logged_metrics

    def test_on_step_excludes_findings_when_disabled(self):
        """log_findings=False omits per-condition counts."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.run = mock_run

        cb = WandbCallback(log_every=1, log_findings=False)
        cb._wandb = mock_wandb
        cb.run = mock_run

        findings = _make_findings()
        cb.on_step(0, _make_health(), findings)

        logged_metrics = mock_wandb.log.call_args[0][0]
        finding_keys = [k for k in logged_metrics if "findings/" in k]
        assert finding_keys == []

    def test_on_step_no_active_run_skips_log(self):
        """If wandb is installed but no run is active, nothing is logged."""
        mock_wandb = MagicMock()
        mock_wandb.run = None  # No active run.

        cb = WandbCallback(log_every=1)
        cb._wandb = mock_wandb
        cb.run = None

        cb.on_step(0, _make_health(), [])
        assert not mock_wandb.log.called

    def test_on_step_explicit_run(self):
        """Explicit run= parameter takes priority over wandb.run."""
        mock_wandb = MagicMock()
        mock_wandb.run = None  # No active global run.
        explicit_run = MagicMock()

        cb = WandbCallback(run=explicit_run, log_every=1)
        cb._wandb = mock_wandb

        cb.on_step(0, _make_health(), [])
        assert mock_wandb.log.called

    def test_on_step_exception_does_not_raise(self):
        """If wandb.log raises, on_step catches and warns."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.run = mock_run
        mock_wandb.log.side_effect = RuntimeError("connection refused")

        cb = WandbCallback(log_every=1)
        cb._wandb = mock_wandb
        cb.run = mock_run

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cb.on_step(0, _make_health(), [])  # Should not propagate exception.

        assert any("failed to log" in str(w.message).lower() for w in caught)

    def test_accepts_health_and_findings_types(self):
        """Callback accepts HealthScore and list[Finding] without error."""
        mock_wandb = MagicMock()
        mock_run = MagicMock()
        mock_wandb.run = mock_run

        cb = WandbCallback(log_every=1)
        cb._wandb = mock_wandb
        cb.run = mock_run

        health = _make_health()
        findings = _make_findings()
        # Should not raise.
        cb.on_step(0, health, findings)


# ── MLflowCallback ────────────────────────────────────────────────────────────

class TestMLflowCallback:

    def test_init_without_mlflow(self):
        """Callback initializes cleanly when mlflow is not installed."""
        cb = MLflowCallback(log_every=50)
        cb._mlflow = None  # Simulate missing package.
        assert cb.log_every == 50

    def test_on_step_no_mlflow_warns_once(self):
        """When mlflow is missing, on_step warns exactly once."""
        cb = MLflowCallback(log_every=1)
        cb._mlflow = None
        cb._import_warned = False

        health = _make_health()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cb.on_step(0, health, [])
            cb.on_step(1, health, [])

        mlflow_warns = [w for w in caught if "mlflow" in str(w.message).lower()]
        assert len(mlflow_warns) == 1

    def test_on_step_respects_log_every(self):
        """on_step only logs when step is a multiple of log_every."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        cb = MLflowCallback(log_every=10)
        cb._mlflow = mock_mlflow

        health = _make_health()
        cb.on_step(3, health, [])   # Should NOT log.
        cb.on_step(10, health, [])  # Should log.

        assert mock_mlflow.log_metrics.call_count == 1

    def test_on_step_logs_correct_keys(self):
        """on_step logs dot-separated metric keys."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        cb = MLflowCallback(log_every=1)
        cb._mlflow = mock_mlflow

        health = _make_health(overall=70, grade="C")
        cb.on_step(0, health, [])

        assert mock_mlflow.log_metrics.called
        logged = mock_mlflow.log_metrics.call_args[0][0]
        # MLflow uses dots, not slashes.
        assert "model_health.overall_score" in logged
        assert logged["model_health.overall_score"] == 70

    def test_on_step_no_active_run_skips_log(self):
        """If no mlflow run is active, nothing is logged."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        cb = MLflowCallback(log_every=1)
        cb._mlflow = mock_mlflow

        cb.on_step(0, _make_health(), [])
        assert not mock_mlflow.log_metrics.called

    def test_on_step_exception_does_not_raise(self):
        """If mlflow.log_metrics raises, on_step catches and warns."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()
        mock_mlflow.log_metrics.side_effect = RuntimeError("tracking server down")

        cb = MLflowCallback(log_every=1)
        cb._mlflow = mock_mlflow

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cb.on_step(0, _make_health(), [])  # Should not propagate.

        assert any("failed to log" in str(w.message).lower() for w in caught)

    def test_accepts_health_and_findings_types(self):
        """Callback accepts HealthScore and list[Finding] without error."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        cb = MLflowCallback(log_every=1)
        cb._mlflow = mock_mlflow

        cb.on_step(0, _make_health(), _make_findings())


# ── TensorBoardCallback ───────────────────────────────────────────────────────

class TestTensorBoardCallback:

    def test_init_with_explicit_writer(self):
        """Callback stores a provided writer without creating its own."""
        mock_writer = MagicMock()
        cb = TensorBoardCallback(writer=mock_writer, log_every=50)
        assert cb._writer is mock_writer
        assert not cb._owns_writer

    def test_on_step_calls_add_scalar(self):
        """on_step calls writer.add_scalar for each metric key."""
        mock_writer = MagicMock()
        cb = TensorBoardCallback(writer=mock_writer, log_every=1)

        health = _make_health(overall=91, grade="A")
        cb.on_step(0, health, [])

        assert mock_writer.add_scalar.called
        calls = {call[0][0] for call in mock_writer.add_scalar.call_args_list}
        assert "model_health/overall_score" in calls
        assert "model_health/grade_numeric" in calls

    def test_on_step_respects_log_every(self):
        """on_step only logs at log_every boundaries."""
        mock_writer = MagicMock()
        cb = TensorBoardCallback(writer=mock_writer, log_every=5)

        health = _make_health()
        cb.on_step(1, health, [])
        cb.on_step(5, health, [])
        cb.on_step(6, health, [])

        # Only step 5 is a multiple of 5.
        call_steps = [call[1]["global_step"] for call in mock_writer.add_scalar.call_args_list]
        assert all(s == 5 for s in call_steps)

    def test_on_step_no_writer_warns_once(self):
        """When no writer available, on_step warns exactly once."""
        cb = TensorBoardCallback(writer=None, log_every=1)
        cb._writer = None  # Force no writer.
        cb._import_warned = False

        health = _make_health()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cb.on_step(0, health, [])
            cb.on_step(1, health, [])

        tb_warns = [w for w in caught if "tensorboard" in str(w.message).lower()]
        assert len(tb_warns) == 1

    def test_on_step_exception_does_not_raise(self):
        """If add_scalar raises, on_step catches and warns."""
        mock_writer = MagicMock()
        mock_writer.add_scalar.side_effect = RuntimeError("write error")
        cb = TensorBoardCallback(writer=mock_writer, log_every=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cb.on_step(0, _make_health(), [])

        assert any("failed to log" in str(w.message).lower() for w in caught)

    def test_accepts_health_and_findings_types(self):
        """Callback accepts HealthScore and list[Finding] without error."""
        mock_writer = MagicMock()
        cb = TensorBoardCallback(writer=mock_writer, log_every=1)
        cb.on_step(0, _make_health(), _make_findings())


# ── Standalone helpers ────────────────────────────────────────────────────────

class TestLogHealthToWandb:

    def test_warns_when_wandb_not_installed(self):
        """Warns gracefully when wandb is absent."""
        with patch.dict(sys.modules, {"wandb": None}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                log_health_to_wandb(_make_health(), [], step=0)

        assert any("wandb" in str(w.message).lower() for w in caught)

    def test_noop_when_no_active_run(self):
        """No-ops if wandb.run is None."""
        mock_wandb = MagicMock()
        mock_wandb.run = None

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_health_to_wandb(_make_health(), [], step=5)

        assert not mock_wandb.log.called

    def test_logs_when_run_active(self):
        """Logs to wandb when run is active."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_health_to_wandb(_make_health(overall=60), _make_findings(), step=10)

        assert mock_wandb.log.called
        logged = mock_wandb.log.call_args[0][0]
        assert logged["model_health/overall_score"] == 60
        assert mock_wandb.log.call_args[1]["step"] == 10

    def test_custom_prefix(self):
        """Respects prefix parameter."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            log_health_to_wandb(_make_health(), [], step=0, prefix="eval/health")

        logged = mock_wandb.log.call_args[0][0]
        assert "eval/health/overall_score" in logged

    def test_exception_does_not_propagate(self):
        """If wandb.log raises, the function catches and warns."""
        mock_wandb = MagicMock()
        mock_wandb.run = MagicMock()
        mock_wandb.log.side_effect = RuntimeError("network error")

        with patch.dict(sys.modules, {"wandb": mock_wandb}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                log_health_to_wandb(_make_health(), [], step=0)

        assert any("failed to log" in str(w.message).lower() for w in caught)


class TestLogHealthToMlflow:

    def test_warns_when_mlflow_not_installed(self):
        """Warns gracefully when mlflow is absent."""
        with patch.dict(sys.modules, {"mlflow": None}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                log_health_to_mlflow(_make_health(), [], step=0)

        assert any("mlflow" in str(w.message).lower() for w in caught)

    def test_noop_when_no_active_run(self):
        """No-ops if no mlflow run is active."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = None

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_health_to_mlflow(_make_health(), [], step=0)

        assert not mock_mlflow.log_metrics.called

    def test_logs_when_run_active(self):
        """Logs to mlflow when run is active."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_health_to_mlflow(_make_health(overall=55), [], step=20)

        assert mock_mlflow.log_metrics.called
        logged = mock_mlflow.log_metrics.call_args[0][0]
        assert logged["model_health.overall_score"] == 55
        assert mock_mlflow.log_metrics.call_args[1]["step"] == 20

    def test_metric_keys_use_dots(self):
        """MLflow metrics use dots, not slashes."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            log_health_to_mlflow(_make_health(), [], step=0)

        logged = mock_mlflow.log_metrics.call_args[0][0]
        slash_keys = [k for k in logged if "/" in k]
        assert slash_keys == [], f"Slash keys found in mlflow metrics: {slash_keys}"

    def test_exception_does_not_propagate(self):
        """If mlflow.log_metrics raises, the function catches and warns."""
        mock_mlflow = MagicMock()
        mock_mlflow.active_run.return_value = MagicMock()
        mock_mlflow.log_metrics.side_effect = RuntimeError("tracking error")

        with patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                log_health_to_mlflow(_make_health(), [], step=0)

        assert any("failed to log" in str(w.message).lower() for w in caught)


# ── Public API exports ────────────────────────────────────────────────────────

class TestPublicExports:

    def test_wandb_callback_importable_from_package(self):
        from model_clinic import WandbCallback as WCB
        assert WCB is WandbCallback

    def test_mlflow_callback_importable_from_package(self):
        from model_clinic import MLflowCallback as MFCB
        assert MFCB is MLflowCallback

    def test_tensorboard_callback_importable_from_package(self):
        from model_clinic import TensorBoardCallback as TBCB
        assert TBCB is TensorBoardCallback

    def test_log_health_to_wandb_importable_from_package(self):
        from model_clinic import log_health_to_wandb as fn
        assert fn is log_health_to_wandb

    def test_log_health_to_mlflow_importable_from_package(self):
        from model_clinic import log_health_to_mlflow as fn
        assert fn is log_health_to_mlflow

    def test_all_exports_in_dunder_all(self):
        import model_clinic
        assert "WandbCallback" in model_clinic.__all__
        assert "MLflowCallback" in model_clinic.__all__
        assert "TensorBoardCallback" in model_clinic.__all__
        assert "log_health_to_wandb" in model_clinic.__all__
        assert "log_health_to_mlflow" in model_clinic.__all__
