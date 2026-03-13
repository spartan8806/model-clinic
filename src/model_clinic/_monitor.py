"""Training-time monitoring — detect problems mid-training."""

from collections import deque

import torch

from model_clinic._types import MonitorAlert, MonitorSummary


# ── Detection thresholds ────────────────────────────────────────────────

_GRAD_EXPLOSION_ABS = 100.0
_GRAD_EXPLOSION_REL = 10.0
_GRAD_VANISHING = 1e-7
_NEURON_DEATH_THRESH = 1e-7
_NEURON_DEATH_FRAC = 0.05
_LOSS_SPIKE_FACTOR = 2.0
_WEIGHT_DIVERGENCE = 10.0
_LAYER_COLLAPSE_STD = 1e-6
_ROLLING_WINDOW = 10


class ClinicMonitor:
    """Callback-style monitor that attaches to a training loop.

    Usage::

        monitor = ClinicMonitor(model, log_every=100)
        for step, batch in enumerate(dataloader):
            loss = train_step(batch)
            monitor.step(step, loss=loss.item())
        print(monitor.summary())
    """

    def __init__(self, model, log_every=100, alert_on=None, log_fn=None):
        self.model = model
        self.log_every = log_every
        self.alert_on = set(alert_on) if alert_on else None  # None = all
        self.log_fn = log_fn

        self._alerts: list = []
        self._grad_norm_history: list = []
        self._loss_history: list = []
        self._dead_neuron_history: list = []
        self._rolling_grad = deque(maxlen=_ROLLING_WINDOW)
        self._rolling_loss = deque(maxlen=_ROLLING_WINDOW)
        self._last_step = 0

        # Snapshot initial weight norms for divergence detection.
        self._initial_norms = {}
        for name, p in model.named_parameters():
            self._initial_norms[name] = p.data.norm().item()

        # Hooks for layer-collapse detection.
        self._layer_stds: dict = {}
        self._hooks = []
        self._register_hooks()

    # ── Hook management ─────────────────────────────────────────────────

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if module is self.model:
                continue
            # Only hook leaf modules that have parameters.
            if list(module.children()):
                continue
            h = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(h)

    def _make_hook(self, name):
        def hook(_module, _input, output):
            if isinstance(output, torch.Tensor) and output.numel() > 0:
                self._layer_stds[name] = output.detach().float().std().item()
        return hook

    def _remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Public API ──────────────────────────────────────────────────────

    def step(self, step, loss=None):
        """Called every training step. Runs checks at ``log_every`` intervals."""
        self._last_step = step
        if loss is not None:
            self._rolling_loss.append(loss)

        if step % self.log_every != 0:
            return []

        alerts = []

        # Record loss.
        if loss is not None:
            self._loss_history.append((step, loss))
            alerts.extend(self._check_loss_spike(step, loss))

        # Gradient checks.
        alerts.extend(self._check_gradients(step))

        # Dead neuron check.
        alerts.extend(self._check_dead_neurons(step))

        # Weight divergence.
        alerts.extend(self._check_weight_divergence(step))

        # Layer collapse (uses data from forward hooks).
        alerts.extend(self._check_layer_collapse(step))

        # Filter by alert_on.
        if self.alert_on is not None:
            alerts = [a for a in alerts if a.condition in self.alert_on]

        self._alerts.extend(alerts)

        if alerts and self.log_fn:
            self.log_fn(step, alerts)

        return alerts

    def summary(self):
        """Return a ``MonitorSummary`` with full history."""
        by_cond: dict = {}
        for a in self._alerts:
            by_cond[a.condition] = by_cond.get(a.condition, 0) + 1

        return MonitorSummary(
            total_steps=self._last_step,
            total_alerts=len(self._alerts),
            alerts_by_condition=by_cond,
            gradient_norm_history=list(self._grad_norm_history),
            loss_history=list(self._loss_history),
            dead_neuron_history=list(self._dead_neuron_history),
        )

    def alerts(self):
        """Return all alerts fired so far."""
        return list(self._alerts)

    # ── Detection logic ─────────────────────────────────────────────────

    def _check_gradients(self, step):
        alerts = []
        max_norm = 0.0
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            norm = p.grad.data.norm().item()
            max_norm = max(max_norm, norm)

        self._grad_norm_history.append((step, max_norm))
        self._rolling_grad.append(max_norm)

        # Explosion: absolute threshold.
        if max_norm > _GRAD_EXPLOSION_ABS:
            alerts.append(MonitorAlert(
                step=step,
                condition="gradient_explosion",
                severity="ERROR",
                details={"max_grad_norm": max_norm, "threshold": _GRAD_EXPLOSION_ABS},
                message=f"Gradient explosion: max norm {max_norm:.2f} > {_GRAD_EXPLOSION_ABS}",
            ))
        # Explosion: relative to rolling average.
        elif len(self._rolling_grad) >= 2:
            avg = sum(list(self._rolling_grad)[:-1]) / (len(self._rolling_grad) - 1)
            if avg > 0 and max_norm > _GRAD_EXPLOSION_REL * avg:
                alerts.append(MonitorAlert(
                    step=step,
                    condition="gradient_explosion",
                    severity="WARN",
                    details={"max_grad_norm": max_norm, "rolling_avg": avg},
                    message=f"Gradient spike: {max_norm:.4f} > {_GRAD_EXPLOSION_REL}x avg ({avg:.4f})",
                ))

        # Vanishing.
        if max_norm < _GRAD_VANISHING and max_norm >= 0:
            alerts.append(MonitorAlert(
                step=step,
                condition="gradient_vanishing",
                severity="WARN",
                details={"max_grad_norm": max_norm},
                message=f"Vanishing gradients: max norm {max_norm:.2e}",
            ))

        return alerts

    def _check_dead_neurons(self, step):
        total = 0
        dead = 0
        for name, p in self.model.named_parameters():
            if p.data.dim() < 2:
                continue
            rows = p.data.shape[0]
            total += rows
            weight_norms = p.data.view(rows, -1).norm(dim=1)
            if p.grad is not None:
                grad_norms = p.grad.data.view(rows, -1).norm(dim=1)
                dead_mask = (weight_norms < _NEURON_DEATH_THRESH) & (grad_norms < _NEURON_DEATH_THRESH)
            else:
                dead_mask = weight_norms < _NEURON_DEATH_THRESH
            dead += dead_mask.sum().item()

        self._dead_neuron_history.append((step, dead))

        if total > 0 and dead / total > _NEURON_DEATH_FRAC:
            return [MonitorAlert(
                step=step,
                condition="neuron_death",
                severity="ERROR",
                details={"dead": dead, "total": total, "fraction": dead / total},
                message=f"Neuron death: {dead}/{total} ({dead/total:.1%}) neurons dead",
            )]
        return []

    def _check_loss_spike(self, step, loss):
        if len(self._rolling_loss) < 2:
            return []
        # Average of everything except the current value.
        past = list(self._rolling_loss)[:-1]
        avg = sum(past) / len(past)
        if avg > 0 and loss > _LOSS_SPIKE_FACTOR * avg:
            return [MonitorAlert(
                step=step,
                condition="loss_spike",
                severity="WARN",
                details={"loss": loss, "rolling_avg": avg},
                message=f"Loss spike: {loss:.4f} > {_LOSS_SPIKE_FACTOR}x avg ({avg:.4f})",
            )]
        return []

    def _check_weight_divergence(self, step):
        alerts = []
        for name, p in self.model.named_parameters():
            init_norm = self._initial_norms.get(name, 0)
            if init_norm < 1e-12:
                continue
            current = p.data.norm().item()
            ratio = current / init_norm
            if ratio > _WEIGHT_DIVERGENCE or (ratio < 1.0 / _WEIGHT_DIVERGENCE and ratio > 0):
                alerts.append(MonitorAlert(
                    step=step,
                    condition="weight_divergence",
                    severity="WARN",
                    details={"param": name, "initial_norm": init_norm, "current_norm": current, "ratio": ratio},
                    message=f"Weight divergence in {name}: norm ratio {ratio:.2f}x",
                ))
        return alerts

    def _check_layer_collapse(self, step):
        alerts = []
        for name, std in self._layer_stds.items():
            if std < _LAYER_COLLAPSE_STD:
                alerts.append(MonitorAlert(
                    step=step,
                    condition="layer_collapse",
                    severity="ERROR",
                    details={"layer": name, "std": std},
                    message=f"Layer collapse in {name}: output std {std:.2e}",
                ))
        self._layer_stds.clear()
        return alerts

    def __del__(self):
        self._remove_hooks()
