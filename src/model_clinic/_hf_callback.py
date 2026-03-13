"""HuggingFace TrainerCallback wrapper around ClinicMonitor."""

try:
    from transformers import TrainerCallback
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

    class TrainerCallback:
        """Stub when transformers is not installed."""
        pass


class ClinicTrainerCallback(TrainerCallback):
    """Drop-in callback for HuggingFace ``Trainer``.

    Usage::

        from model_clinic import ClinicTrainerCallback

        trainer = Trainer(
            model=model,
            callbacks=[ClinicTrainerCallback(log_every=100)],
            ...
        )

    If wandb or mlflow are active (i.e. ``wandb.run`` or ``mlflow.active_run()``
    is not None), health metrics are automatically forwarded to them at the same
    ``log_every`` cadence.
    """

    def __init__(self, log_every=100, alert_on=None, log_fn=None):
        if not _HAS_TRANSFORMERS:
            raise ImportError(
                "ClinicTrainerCallback requires the `transformers` package. "
                "Install it with: pip install transformers"
            )
        self.log_every = log_every
        self.alert_on = alert_on
        self.log_fn = log_fn
        self.monitor = None
        self._last_health = None
        self._last_findings = []

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from model_clinic._monitor import ClinicMonitor
        self.monitor = ClinicMonitor(
            model,
            log_every=self.log_every,
            alert_on=self.alert_on,
            log_fn=self.log_fn,
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.monitor is None:
            return
        loss = None
        if state.log_history:
            last = state.log_history[-1]
            loss = last.get("loss")
        self.monitor.step(state.global_step, loss=loss)

        # Forward to active experiment trackers if a log_every boundary was hit.
        if state.global_step % self.log_every == 0 and self._last_health is not None:
            self._forward_to_trackers(state.global_step)

    def _forward_to_trackers(self, step: int):
        """Forward the most recent health snapshot to wandb / mlflow if active."""
        health = self._last_health
        findings = self._last_findings
        if health is None:
            return

        # wandb
        try:
            import wandb as _wandb
            if getattr(_wandb, "run", None) is not None:
                from model_clinic._integrations import _build_metrics, _build_finding_counts
                metrics = _build_metrics(health, findings, "model_health")
                metrics.update(_build_finding_counts(findings, "model_health"))
                _wandb.log(metrics, step=step)
        except Exception:
            pass

        # mlflow
        try:
            import mlflow as _mlflow
            if _mlflow.active_run() is not None:
                from model_clinic._integrations import _build_metrics
                raw = _build_metrics(health, findings, "model_health")
                mlflow_metrics = {k.replace("/", "."): v for k, v in raw.items()}
                _mlflow.log_metrics(mlflow_metrics, step=step)
        except Exception:
            pass

    def on_train_end(self, args, state, control, **kwargs):
        if self.monitor is None:
            return
        summary = self.monitor.summary()
        print(f"\n-- ClinicMonitor Summary --")
        print(f"  Steps: {summary.total_steps}")
        print(f"  Alerts: {summary.total_alerts}")
        if summary.alerts_by_condition:
            for cond, count in sorted(summary.alerts_by_condition.items()):
                print(f"    {cond}: {count}")
        print()
