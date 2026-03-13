"""Tests for dashboard features: gate evolution plot and interactive HTML report."""

import os
import subprocess
import sys
import tempfile

import torch

# Windows subprocess helpers (match pattern from test_cli.py)
_SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
_RUN_KWARGS = {"stdin": subprocess.DEVNULL, "env": _SUBPROCESS_ENV}

from model_clinic._report import (
    _build_gate_evolution_svg,
    generate_report,
)
from model_clinic._types import Finding, HealthScore, ModelMeta


def _valid_svg(svg_str):
    """Check that a string is a well-formed SVG element."""
    assert svg_str.strip().startswith("<svg"), f"Expected SVG start, got: {svg_str[:60]}"
    assert svg_str.strip().endswith("</svg>"), f"Expected SVG end, got: {svg_str[-60:]}"


# ---------------------------------------------------------------------------
# _build_gate_evolution_svg
# ---------------------------------------------------------------------------

class TestBuildGateEvolutionSvg:

    def _gate_sd(self, **kw):
        """Build a minimal state dict with scalar gate tensors."""
        return {k: torch.tensor(v) for k, v in kw.items()}

    def test_returns_valid_svg_with_gates(self):
        before = self._gate_sd(**{"module/gate": -8.0, "module/pre_memory_gate": 0.5})
        after = self._gate_sd(**{"module/gate": -3.0, "module/pre_memory_gate": 0.8})
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)

    def test_contains_rect_elements(self):
        before = {"gate": torch.tensor(-6.0)}
        after = {"gate": torch.tensor(2.0)}
        svg = _build_gate_evolution_svg(before, after)
        assert "rect" in svg

    def test_empty_returns_empty_string(self):
        # No gate tensors — should return ""
        before = {"embed.weight": torch.randn(10, 8)}
        after = {"embed.weight": torch.randn(10, 8)}
        result = _build_gate_evolution_svg(before, after)
        assert result == ""

    def test_both_empty_returns_empty_string(self):
        assert _build_gate_evolution_svg({}, {}) == ""

    def test_red_color_for_stuck_closed(self):
        """Gates with value < -5 should have red color in the SVG."""
        before = {"gate": torch.tensor(-9.0)}
        after = {"gate": torch.tensor(-7.0)}
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)
        assert "#f44336" in svg  # red

    def test_orange_color_for_stuck_open(self):
        """Gates with value > 5 should have orange color in the SVG."""
        before = {"gate": torch.tensor(8.0)}
        after = {"gate": torch.tensor(6.0)}
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)
        assert "#ff9800" in svg  # orange

    def test_green_color_for_healthy(self):
        """Gates in (-5, 5) should have green color in the SVG."""
        before = {"gate": torch.tensor(0.0)}
        after = {"gate": torch.tensor(1.5)}
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)
        assert "#4caf50" in svg  # green

    def test_ignores_non_scalar_tensors_named_gate(self):
        """Tensors named *gate* with numel() > 1 should be ignored."""
        before = {"gate_proj.weight": torch.randn(64, 64)}  # numel=4096
        after = {"gate_proj.weight": torch.randn(64, 64)}
        result = _build_gate_evolution_svg(before, after)
        assert result == ""

    def test_custom_dimensions(self):
        before = {"gate": torch.tensor(-3.0)}
        after = {"gate": torch.tensor(1.0)}
        svg = _build_gate_evolution_svg(before, after, width=800, height=300)
        _valid_svg(svg)
        assert 'width="800"' in svg

    def test_gate_only_in_before(self):
        """Gate present in before but not after should still render."""
        before = {"gate": torch.tensor(-2.0)}
        after = {}
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)

    def test_gate_only_in_after(self):
        """Gate present in after but not before should still render."""
        before = {}
        after = {"gate": torch.tensor(3.0)}
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)

    def test_multiple_gates(self):
        """Multiple scalar gates should all appear in the chart."""
        before = {
            "block0/gate": torch.tensor(-6.0),
            "block1/gate": torch.tensor(0.5),
            "block2/gate": torch.tensor(7.0),
        }
        after = {
            "block0/gate": torch.tensor(-2.0),
            "block1/gate": torch.tensor(1.0),
            "block2/gate": torch.tensor(3.0),
        }
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)
        # Should have content for all three
        assert "block0" in svg
        assert "block1" in svg
        assert "block2" in svg

    def test_legend_present(self):
        """The SVG should include a legend."""
        before = {"gate": torch.tensor(0.0)}
        after = {"gate": torch.tensor(1.0)}
        svg = _build_gate_evolution_svg(before, after)
        _valid_svg(svg)
        # Legend text for each zone
        assert "Healthy" in svg
        assert "closed" in svg  # "Stuck closed"
        assert "open" in svg    # "Stuck open"


# ---------------------------------------------------------------------------
# generate_report with interactive=True
# ---------------------------------------------------------------------------

class TestGenerateReportInteractive:

    def _make_report(self, findings=None, interactive=True):
        """Helper to generate a report and return HTML content."""
        findings = findings or []
        hs = HealthScore(overall=75, categories={"weights": 80, "stability": 70}, grade="C")
        meta = ModelMeta(source="test_model", num_params=1000, num_tensors=10,
                         hidden_size=64, num_layers=2)
        sd = {
            "embed.weight": torch.randn(50, 64),
            "norm.weight": torch.ones(64),
        }
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(sd, findings, [], hs, meta, path, interactive=interactive)
            with open(path, encoding="utf-8") as f:
                return f.read()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_interactive_true_includes_script_tag(self):
        html = self._make_report(interactive=True)
        assert "<script>" in html

    def test_interactive_false_no_script_tag(self):
        html = self._make_report(interactive=False)
        assert "<script>" not in html

    def test_filter_js_present_in_interactive_report(self):
        html = self._make_report(interactive=True)
        # The word "filter" should appear in the JS
        assert "filter" in html.lower()

    def test_filter_buttons_present(self):
        html = self._make_report(interactive=True)
        assert "data-filter-sev" in html
        assert "data-filter-cat" in html

    def test_search_box_present(self):
        html = self._make_report(interactive=True)
        assert "param-search" in html
        assert "Search tensor name" in html

    def test_sort_select_present(self):
        html = self._make_report(interactive=True)
        assert "sort-select" in html

    def test_findings_tbody_id_present(self):
        html = self._make_report(interactive=True)
        assert 'id="findings-tbody"' in html

    def test_finding_rows_have_data_attributes(self):
        findings = [
            Finding("norm_drift", "WARN", "layers.0.norm.weight", {"mean": 2.5}),
            Finding("dead_neurons", "ERROR", "layers.0.mlp.weight",
                    {"dim": "rows", "pct": 15.0}),
        ]
        html = self._make_report(findings=findings, interactive=True)
        assert 'data-severity="WARN"' in html
        assert 'data-severity="ERROR"' in html
        assert 'data-category=' in html
        assert 'data-param=' in html

    def test_severity_filter_buttons_all_severities(self):
        html = self._make_report(interactive=True)
        assert 'data-filter-sev="ALL"' in html
        assert 'data-filter-sev="ERROR"' in html
        assert 'data-filter-sev="WARN"' in html
        assert 'data-filter-sev="INFO"' in html

    def test_category_filter_buttons_present(self):
        html = self._make_report(interactive=True)
        assert 'data-filter-cat="weights"' in html
        assert 'data-filter-cat="stability"' in html

    def test_report_still_valid_html(self):
        html = self._make_report(interactive=True)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html
        assert "<svg" in html  # gauge at minimum

    def test_mcfilter_function_defined(self):
        """The JS filter function name should appear in the output."""
        html = self._make_report(interactive=True)
        assert "mcFilter" in html

    def test_mcsort_function_defined(self):
        """The JS sort function name should appear in the output."""
        html = self._make_report(interactive=True)
        assert "mcSort" in html


# ---------------------------------------------------------------------------
# Gate evolution in compare report
# ---------------------------------------------------------------------------

class TestGateEvolutionInCompareReport:

    def test_compare_data_with_gate_sds_includes_evolution(self):
        """When compare_data has before_sd/after_sd with gates, chart appears."""
        hs = HealthScore(overall=80, categories={"weights": 85, "stability": 75}, grade="B")
        meta = ModelMeta(source="test", num_params=500, num_tensors=5,
                         hidden_size=32, num_layers=1)
        sd = {"embed.weight": torch.randn(20, 32)}

        class CompareData:
            before = HealthScore(overall=60,
                                 categories={"weights": 55, "stability": 65}, grade="D")
            after = HealthScore(overall=80,
                                categories={"weights": 85, "stability": 75}, grade="B")
            before_sd = {"gate": torch.tensor(-8.0)}
            after_sd = {"gate": torch.tensor(-2.0)}

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(sd, [], [], hs, meta, path,
                            compare_data=CompareData())
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "Gate Evolution" in html
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def test_compare_data_without_gate_sds_no_gate_section(self):
        """When compare_data lacks before_sd/after_sd, no gate section."""
        hs = HealthScore(overall=80, categories={"weights": 85}, grade="B")
        meta = ModelMeta(source="test", num_params=500, num_tensors=5,
                         hidden_size=32, num_layers=1)
        sd = {"embed.weight": torch.randn(20, 32)}

        class CompareData:
            before = HealthScore(overall=60, categories={"weights": 55}, grade="D")
            after = HealthScore(overall=80, categories={"weights": 85}, grade="B")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(sd, [], [], hs, meta, path,
                            compare_data=CompareData())
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "Gate Evolution" not in html
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Dashboard CLI registered in help
# ---------------------------------------------------------------------------

class TestDashboardCliHelp:

    def test_dashboard_in_help_output(self):
        """model-clinic --help should mention the dashboard command."""
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "--help"],
            capture_output=True, text=True, timeout=120, **_RUN_KWARGS,
        )
        output = result.stdout + result.stderr
        assert "dashboard" in output.lower(), (
            f"'dashboard' not found in CLI help output:\n{output}"
        )

    def test_dashboard_subcommand_help(self):
        """model-clinic dashboard --help should show dashboard-specific help."""
        result = subprocess.run(
            [sys.executable, "-m", "model_clinic.cli", "dashboard", "--help"],
            capture_output=True, text=True, timeout=120, **_RUN_KWARGS,
        )
        output = result.stdout + result.stderr
        # Should mention port and model
        assert "--port" in output or "port" in output.lower(), (
            f"Dashboard help missing port info:\n{output}"
        )
