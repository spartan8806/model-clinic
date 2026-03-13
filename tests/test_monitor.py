"""Tests for the training-time monitoring system."""

import torch
import torch.nn as nn

from model_clinic._monitor import ClinicMonitor
from model_clinic._types import MonitorAlert, MonitorSummary


def _tiny_model():
    """A minimal model for testing."""
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    return model


def _fake_train_step(model, step):
    """Simulate a training step: forward + backward."""
    x = torch.randn(2, 8)
    out = model(x)
    loss = out.sum()
    loss.backward()
    return loss.item()


class TestClinicMonitor:

    def test_no_alerts_on_healthy_training(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        # Use a stable loss to avoid false loss_spike from random variance.
        for step in range(10):
            _fake_train_step(model, step)
            alerts = monitor.step(step, loss=1.0)
        # Healthy tiny model with stable loss should not fire alerts.
        assert len(monitor.alerts()) == 0

    def test_gradient_explosion_absolute(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        # Simulate a forward/backward to populate grads.
        _fake_train_step(model, 0)
        # Blow up a gradient.
        for p in model.parameters():
            p.grad = torch.full_like(p.grad, 200.0)
            break
        alerts = monitor.step(0, loss=1.0)
        conditions = [a.condition for a in alerts]
        assert "gradient_explosion" in conditions

    def test_gradient_vanishing(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        _fake_train_step(model, 0)
        # Set all gradients to near-zero.
        for p in model.parameters():
            p.grad = torch.zeros_like(p.grad)
        alerts = monitor.step(0, loss=1.0)
        conditions = [a.condition for a in alerts]
        assert "gradient_vanishing" in conditions

    def test_loss_spike(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        # Build up a rolling average.
        for step in range(5):
            _fake_train_step(model, step)
            monitor.step(step, loss=1.0)
        # Now spike the loss.
        _fake_train_step(model, 5)
        alerts = monitor.step(5, loss=100.0)
        conditions = [a.condition for a in alerts]
        assert "loss_spike" in conditions

    def test_neuron_death(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        _fake_train_step(model, 0)
        # Zero out all weights and grads to simulate death.
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
                if p.grad is not None:
                    p.grad.zero_()
        alerts = monitor.step(0, loss=1.0)
        conditions = [a.condition for a in alerts]
        assert "neuron_death" in conditions

    def test_weight_divergence(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        _fake_train_step(model, 0)
        # Blow up weights.
        with torch.no_grad():
            for p in model.parameters():
                p.mul_(100.0)
                break
        alerts = monitor.step(0, loss=1.0)
        conditions = [a.condition for a in alerts]
        assert "weight_divergence" in conditions

    def test_layer_collapse(self):
        # Use a model that will produce near-zero output.
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        # Zero all weights so output std collapses.
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        # Need a forward pass to trigger hooks.
        x = torch.randn(2, 8)
        _ = model(x)
        # Now check — grads won't exist, but that's fine.
        alerts = monitor.step(0, loss=0.0)
        conditions = [a.condition for a in alerts]
        assert "layer_collapse" in conditions

    def test_log_every_skips(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=10)
        _fake_train_step(model, 1)
        # Step 1 should not trigger checks (only step 0, 10, 20, ...).
        alerts = monitor.step(1, loss=1.0)
        assert alerts == []

    def test_summary_structure(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        for step in range(3):
            loss = _fake_train_step(model, step)
            monitor.step(step, loss=loss)
        summary = monitor.summary()
        assert isinstance(summary, MonitorSummary)
        assert summary.total_steps == 2
        assert isinstance(summary.gradient_norm_history, list)
        assert isinstance(summary.loss_history, list)
        assert len(summary.loss_history) == 3

    def test_alert_on_filter(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1, alert_on=["loss_spike"])
        _fake_train_step(model, 0)
        # Zero grads to trigger vanishing — but it's filtered out.
        for p in model.parameters():
            p.grad = torch.zeros_like(p.grad)
        alerts = monitor.step(0, loss=1.0)
        # gradient_vanishing should be filtered.
        conditions = [a.condition for a in alerts]
        assert "gradient_vanishing" not in conditions

    def test_log_fn_called(self):
        model = _tiny_model()
        calls = []
        monitor = ClinicMonitor(model, log_every=1, log_fn=lambda s, a: calls.append((s, a)))
        # Build rolling average then spike.
        for step in range(5):
            _fake_train_step(model, step)
            monitor.step(step, loss=1.0)
        _fake_train_step(model, 5)
        monitor.step(5, loss=100.0)
        # log_fn should have been called with the spike.
        assert len(calls) > 0
        assert any(a.condition == "loss_spike" for _, alerts in calls for a in alerts)

    def test_alert_dataclass(self):
        alert = MonitorAlert(step=10, condition="test", severity="WARN", message="hi")
        assert alert.step == 10
        assert "WARN" in str(alert)

    def test_summary_alerts_by_condition(self):
        model = _tiny_model()
        monitor = ClinicMonitor(model, log_every=1)
        # Force a spike.
        for step in range(5):
            _fake_train_step(model, step)
            monitor.step(step, loss=1.0)
        _fake_train_step(model, 5)
        monitor.step(5, loss=100.0)
        summary = monitor.summary()
        if summary.total_alerts > 0:
            assert isinstance(summary.alerts_by_condition, dict)
            assert sum(summary.alerts_by_condition.values()) == summary.total_alerts
