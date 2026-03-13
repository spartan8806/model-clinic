"""Tests for the plugin discovery and loading system."""

import warnings
from unittest.mock import MagicMock, patch

import torch
import pytest

from model_clinic._plugins import load_plugins, list_plugins, reset_plugins_loaded, plugins_loaded
from model_clinic._types import Finding
from model_clinic.clinic import ConditionRegistry, diagnose


class TestLoadPlugins:
    """Test load_plugins() behavior."""

    def setup_method(self):
        reset_plugins_loaded()

    def test_no_plugins_installed(self):
        """load_plugins() returns empty list when no plugins are installed."""
        registry = ConditionRegistry()
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[]):
            loaded = load_plugins(registry)
        assert loaded == []

    def test_load_with_mock_entry_point(self):
        """load_plugins() calls register(registry) for each discovered plugin."""
        registry = ConditionRegistry()

        # Create a mock entry point
        mock_ep = MagicMock()
        mock_ep.name = "test_plugin"

        def fake_register(reg):
            def my_detector(name, tensor, ctx):
                if tensor.dim() >= 2 and tensor.float().max().item() > 999:
                    return [Finding("my_custom_check", "WARN", name, {})]
                return []
            reg.register("my_custom_check", my_detector)

        mock_ep.load.return_value = fake_register

        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[mock_ep]):
            loaded = load_plugins(registry)

        assert loaded == ["test_plugin"]
        assert "my_custom_check" in registry._detectors

    def test_load_plugin_that_raises(self):
        """load_plugins() warns but continues when a plugin raises."""
        registry = ConditionRegistry()

        mock_ep = MagicMock()
        mock_ep.name = "broken_plugin"
        mock_ep.load.return_value = MagicMock(side_effect=RuntimeError("plugin init failed"))

        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[mock_ep]):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                loaded = load_plugins(registry)

        assert loaded == []
        assert len(w) == 1
        assert "broken_plugin" in str(w[0].message)

    def test_load_sets_plugins_loaded_flag(self):
        """After load_plugins(), plugins_loaded() returns True."""
        registry = ConditionRegistry()
        assert not plugins_loaded()
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[]):
            load_plugins(registry)
        assert plugins_loaded()

    def test_load_uses_global_registry_when_none(self):
        """load_plugins(None) uses the global REGISTRY."""
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[]):
            loaded = load_plugins(None)
        assert loaded == []

    def test_multiple_plugins(self):
        """load_plugins() loads multiple plugins in order."""
        registry = ConditionRegistry()

        def make_ep(ep_name, condition):
            ep = MagicMock()
            ep.name = ep_name
            def reg_fn(r):
                r.register(condition, lambda n, t, c: [])
            ep.load.return_value = reg_fn
            return ep

        eps = [make_ep("plugin_a", "check_a"), make_ep("plugin_b", "check_b")]
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=eps):
            loaded = load_plugins(registry)

        assert loaded == ["plugin_a", "plugin_b"]
        assert "check_a" in registry._detectors
        assert "check_b" in registry._detectors


class TestListPlugins:
    """Test list_plugins() behavior."""

    def test_no_plugins_returns_empty(self):
        """list_plugins() returns empty list when none installed."""
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[]):
            result = list_plugins()
        assert result == []

    def test_list_returns_plugin_info(self):
        """list_plugins() returns dicts with name and module."""
        mock_ep = MagicMock()
        mock_ep.name = "my_plugin"
        mock_ep.value = "my_package.clinic_plugin:register"
        mock_ep.dist = None

        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[mock_ep]):
            result = list_plugins()

        assert len(result) == 1
        assert result[0]["name"] == "my_plugin"
        assert result[0]["module"] == "my_package.clinic_plugin:register"


class TestDiagnosePluginKwarg:
    """Test that diagnose() accepts plugins=False."""

    def setup_method(self):
        reset_plugins_loaded()

    def test_diagnose_plugins_false(self):
        """diagnose(plugins=False) skips plugin loading."""
        sd = {"w": torch.randn(4, 4)}
        with patch("model_clinic._plugins.load_plugins") as mock_load:
            findings = diagnose(sd, plugins=False)
            mock_load.assert_not_called()

    def test_diagnose_plugins_true_default(self):
        """diagnose() with default plugins=True calls load_plugins once."""
        sd = {"w": torch.randn(4, 4)}
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[]):
            findings = diagnose(sd, plugins=True)
        # After first call, plugins_loaded is True
        assert plugins_loaded()

    def test_diagnose_plugins_only_loads_once(self):
        """Repeated diagnose() calls only load plugins the first time."""
        sd = {"w": torch.randn(4, 4)}
        with patch("model_clinic._plugins.importlib.metadata.entry_points", return_value=[]) as mock_eps:
            diagnose(sd, plugins=True)
            diagnose(sd, plugins=True)
        # entry_points should only be called once
        assert mock_eps.call_count == 1
