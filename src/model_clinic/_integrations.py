"""Integration callbacks for logging model health metrics to experiment trackers.

Supported backends:
  - Weights & Biases (wandb)
  - MLflow
  - TensorBoard

All callbacks degrade gracefully when the backend is not installed.
"""

import warnings
from model_clinic._types import HealthScore


# ── Grade to numeric mapping ─────────────────────────────────────────────────

_GRADE_NUMERIC = {"A": 5, "B": 4, "C": 3, "D": 2, "F": 1}


def _grade_to_numeric(grade: str) -> int:
    return _GRADE_NUMERIC.get(grade, 0)


def _build_metrics(health: HealthScore, findings: list, prefix: str) -> dict:
    """Build a flat metrics dict from a HealthScore and findings list."""
    n_errors = sum(1 for f in findings if getattr(f, "severity", None) == "ERROR")
    n_warnings = sum(1 for f in findings if getattr(f, "severity", None) == "WARN")

    metrics = {
        f"{prefix}/overall_score": health.overall,
        f"{prefix}/grade_numeric": _grade_to_numeric(health.grade),
        f"{prefix}/weights_score": health.categories.get("weights", 100),
        f"{prefix}/stability_score": health.categories.get("stability", 100),
        f"{prefix}/activations_score": health.categories.get("activations", 100),
        f"{prefix}/output_score": health.categories.get("output", 100),
        f"{prefix}/n_errors": n_errors,
        f"{prefix}/n_warnings": n_warnings,
    }
    return metrics


def _build_finding_counts(findings: list, prefix: str) -> dict:
    """Count findings per condition and return as a flat metrics dict."""
    counts: dict = {}
    for f in findings:
        cond = getattr(f, "condition", "unknown")
        key = f"{prefix}/findings/{cond}"
        counts[key] = counts.get(key, 0) + 1
    return counts


# ── WandbCallback ─────────────────────────────────────────────────────────────

class WandbCallback:
    """Log model health metrics to Weights & Biases during training.

    Usage::

        import wandb
        from model_clinic import WandbCallback

        cb = WandbCallback(log_every=100, log_findings=True)
        monitor = ClinicMonitor(model, callbacks=[cb])

    If ``run`` is None the callback auto-detects the active ``wandb.run``.
    If wandb is not installed, all calls are silently no-ops.
    """

    def __init__(
        self,
        run=None,
        log_every: int = 100,
        log_findings: bool = True,
        prefix: str = "model_health",
    ):
        self.run = run
        self.log_every = log_every
        self.log_findings = log_findings
        self.prefix = prefix
        self._wandb = None
        self._import_warned = False

        try:
            import wandb as _wandb_mod
            self._wandb = _wandb_mod
        except ImportError:
            pass  # Will warn on first on_step call if needed.

    def _get_run(self):
        if self._wandb is None:
            return None
        if self.run is not None:
            return self.run
        return getattr(self._wandb, "run", None)

    def on_step(self, step: int, health: HealthScore, findings: list):
        """Log health metrics to wandb at the configured interval.

        Args:
            step: Current training step.
            health: HealthScore from compute_health_score().
            findings: List of Finding objects from diagnose().
        """
        if step % self.log_every != 0:
            return

        if self._wandb is None:
            if not self._import_warned:
                warnings.warn(
                    "WandbCallback: wandb is not installed. "
                    "Install it with: pip install wandb",
                    stacklevel=2,
                )
                self._import_warned = True
            return

        active_run = self._get_run()
        if active_run is None:
            return

        metrics = _build_metrics(health, findings, self.prefix)
        if self.log_findings:
            metrics.update(_build_finding_counts(findings, self.prefix))

        try:
            self._wandb.log(metrics, step=step)
        except Exception as exc:
            warnings.warn(f"WandbCallback: failed to log metrics: {exc}", stacklevel=2)


# ── MLflowCallback ────────────────────────────────────────────────────────────

class MLflowCallback:
    """Log model health metrics to MLflow during training.

    Usage::

        from model_clinic import MLflowCallback

        cb = MLflowCallback(log_every=100)
        monitor = ClinicMonitor(model, callbacks=[cb])

    If ``run_id`` is None the callback logs to the currently active MLflow run.
    If mlflow is not installed, all calls are silently no-ops.
    """

    def __init__(
        self,
        run_id=None,
        log_every: int = 100,
        prefix: str = "model_health",
    ):
        self.run_id = run_id
        self.log_every = log_every
        self.prefix = prefix
        self._mlflow = None
        self._import_warned = False

        try:
            import mlflow as _mlflow_mod
            self._mlflow = _mlflow_mod
        except ImportError:
            pass

    def on_step(self, step: int, health: HealthScore, findings: list):
        """Log health metrics to mlflow at the configured interval.

        Args:
            step: Current training step.
            health: HealthScore from compute_health_score().
            findings: List of Finding objects from diagnose().
        """
        if step % self.log_every != 0:
            return

        if self._mlflow is None:
            if not self._import_warned:
                warnings.warn(
                    "MLflowCallback: mlflow is not installed. "
                    "Install it with: pip install mlflow",
                    stacklevel=2,
                )
                self._import_warned = True
            return

        # MLflow metric keys use dots, not slashes.
        raw = _build_metrics(health, [], self.prefix)
        # Convert prefix/key → prefix.key for MLflow naming convention.
        mlflow_metrics = {k.replace("/", "."): v for k, v in raw.items()}

        try:
            if self.run_id is not None:
                with self._mlflow.start_run(run_id=self.run_id, nested=True):
                    self._mlflow.log_metrics(mlflow_metrics, step=step)
            else:
                # Only log if there is an active run; don't start one automatically.
                if self._mlflow.active_run() is None:
                    return
                self._mlflow.log_metrics(mlflow_metrics, step=step)
        except Exception as exc:
            warnings.warn(
                f"MLflowCallback: failed to log metrics: {exc}", stacklevel=2
            )


# ── TensorBoardCallback ───────────────────────────────────────────────────────

class TensorBoardCallback:
    """Log model health metrics to TensorBoard during training.

    Usage::

        from torch.utils.tensorboard import SummaryWriter
        from model_clinic import TensorBoardCallback

        writer = SummaryWriter("runs/my_run")
        cb = TensorBoardCallback(writer=writer, log_every=100)
        monitor = ClinicMonitor(model, callbacks=[cb])

    If ``writer`` is None the callback attempts to create a default
    ``SummaryWriter``. If tensorboard is not installed, all calls are no-ops.
    """

    def __init__(self, writer=None, log_every: int = 100, prefix: str = "model_health"):
        self.log_every = log_every
        self.prefix = prefix
        self._writer = writer
        self._tb = None
        self._import_warned = False
        self._owns_writer = False

        try:
            from torch.utils.tensorboard import SummaryWriter as _SW
            self._tb = _SW
        except ImportError:
            try:
                from tensorboard.summary.writer.record_writer import RecordWriter  # noqa: F401
                # tensorboard installed but not via torch — no SummaryWriter available
                self._tb = None
            except ImportError:
                pass

        if self._writer is None and self._tb is not None:
            try:
                self._writer = self._tb()
                self._owns_writer = True
            except Exception:
                self._writer = None

    def on_step(self, step: int, health: HealthScore, findings: list):
        """Log health metrics to TensorBoard at the configured interval.

        Args:
            step: Current training step.
            health: HealthScore from compute_health_score().
            findings: List of Finding objects from diagnose().
        """
        if step % self.log_every != 0:
            return

        if self._writer is None:
            if not self._import_warned:
                warnings.warn(
                    "TensorBoardCallback: tensorboard is not available or no writer "
                    "provided. Install with: pip install tensorboard",
                    stacklevel=2,
                )
                self._import_warned = True
            return

        metrics = _build_metrics(health, findings, self.prefix)
        try:
            for tag, value in metrics.items():
                self._writer.add_scalar(tag, value, global_step=step)
        except Exception as exc:
            warnings.warn(
                f"TensorBoardCallback: failed to log metrics: {exc}", stacklevel=2
            )

    def close(self):
        """Flush and close the writer if owned by this callback."""
        if self._owns_writer and self._writer is not None:
            try:
                self._writer.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


# ── Standalone one-shot helpers ───────────────────────────────────────────────

def log_health_to_wandb(
    health: HealthScore,
    findings: list,
    step: int = 0,
    prefix: str = "model_health",
):
    """One-shot log of model health to the active wandb run.

    No callback setup needed. Silently no-ops if wandb is not installed or
    there is no active run.

    Args:
        health: HealthScore from compute_health_score().
        findings: List of Finding objects from diagnose().
        step: Training step to log at (default 0).
        prefix: Metric key prefix (default "model_health").
    """
    try:
        import wandb
    except ImportError:
        warnings.warn(
            "log_health_to_wandb: wandb is not installed. "
            "Install it with: pip install wandb",
            stacklevel=2,
        )
        return

    active_run = getattr(wandb, "run", None)
    if active_run is None:
        return

    metrics = _build_metrics(health, findings, prefix)
    metrics.update(_build_finding_counts(findings, prefix))

    try:
        wandb.log(metrics, step=step)
    except Exception as exc:
        warnings.warn(f"log_health_to_wandb: failed to log: {exc}", stacklevel=2)


def log_health_to_mlflow(
    health: HealthScore,
    findings: list,
    step: int = 0,
    prefix: str = "model_health",
):
    """One-shot log of model health to the active MLflow run.

    No callback setup needed. Silently no-ops if mlflow is not installed or
    there is no active run.

    Args:
        health: HealthScore from compute_health_score().
        findings: List of Finding objects from diagnose().
        step: Training step to log at (default 0).
        prefix: Metric key prefix (default "model_health").
    """
    try:
        import mlflow
    except ImportError:
        warnings.warn(
            "log_health_to_mlflow: mlflow is not installed. "
            "Install it with: pip install mlflow",
            stacklevel=2,
        )
        return

    # Check for an active run.
    try:
        active_run = mlflow.active_run()
    except Exception:
        active_run = None

    if active_run is None:
        return

    raw = _build_metrics(health, [], prefix)
    mlflow_metrics = {k.replace("/", "."): v for k, v in raw.items()}

    try:
        mlflow.log_metrics(mlflow_metrics, step=step)
    except Exception as exc:
        warnings.warn(f"log_health_to_mlflow: failed to log: {exc}", stacklevel=2)
