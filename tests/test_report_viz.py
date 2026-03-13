"""Tests for SVG visualizations in the HTML report."""

import os
import tempfile

import torch

from model_clinic._report import (
    _svg_histogram,
    _svg_norm_bars,
    _svg_dead_neuron_grid,
    _svg_gauge,
    _svg_attention_entropy_heatmap,
    _svg_before_after_bars,
    _svg_neuron_activation_histogram,
    _extract_norm_data,
    _extract_dead_neuron_data,
    _sample_weights,
    generate_report,
)
from model_clinic._types import Finding, HealthScore, ModelMeta


def _valid_svg(svg_str):
    """Check that a string is a well-formed SVG element."""
    assert svg_str.strip().startswith("<svg"), f"Expected SVG start, got: {svg_str[:60]}"
    assert svg_str.strip().endswith("</svg>"), f"Expected SVG end, got: {svg_str[-60:]}"


# ---------------------------------------------------------------------------
# _svg_histogram
# ---------------------------------------------------------------------------

class TestSvgHistogram:

    def test_basic(self):
        svg = _svg_histogram([1.0, 2.0, 3.0, 4.0, 5.0])
        _valid_svg(svg)
        assert "rect" in svg

    def test_empty_returns_empty(self):
        assert _svg_histogram([]) == ""

    def test_constant_values(self):
        svg = _svg_histogram([3.0] * 100)
        _valid_svg(svg)
        assert "constant" in svg

    def test_large_dataset(self):
        vals = torch.randn(10000).tolist()
        svg = _svg_histogram(vals, bins=50)
        _valid_svg(svg)

    def test_custom_dimensions(self):
        svg = _svg_histogram([1.0, 2.0], width=300, height=100, bins=10)
        _valid_svg(svg)
        assert 'width="300"' in svg
        assert 'height="100"' in svg

    def test_single_value(self):
        svg = _svg_histogram([42.0])
        _valid_svg(svg)


# ---------------------------------------------------------------------------
# _svg_norm_bars
# ---------------------------------------------------------------------------

class TestSvgNormBars:

    def test_basic(self):
        data = [("layer.0.norm.weight", 1.0), ("layer.1.norm.weight", 1.5)]
        svg = _svg_norm_bars(data)
        _valid_svg(svg)
        assert "rect" in svg

    def test_empty_returns_empty(self):
        assert _svg_norm_bars([]) == ""

    def test_single_entry(self):
        svg = _svg_norm_bars([("norm", 0.95)])
        _valid_svg(svg)

    def test_drifted_values(self):
        data = [("a", 0.1), ("b", 1.0), ("c", 2.5)]
        svg = _svg_norm_bars(data)
        _valid_svg(svg)
        # Should contain the reference line at 1.0
        assert "1.0" in svg

    def test_long_names_truncated(self):
        name = "model.layers.999.self_attn.q_proj.layernorm.weight"
        svg = _svg_norm_bars([(name, 1.0)])
        _valid_svg(svg)


# ---------------------------------------------------------------------------
# _svg_dead_neuron_grid
# ---------------------------------------------------------------------------

class TestSvgDeadNeuronGrid:

    def test_basic(self):
        data = [("layer.0", 5.0), ("layer.1", 50.0), ("layer.2", 0.0)]
        svg = _svg_dead_neuron_grid(data)
        _valid_svg(svg)
        assert "rect" in svg

    def test_empty_returns_empty(self):
        assert _svg_dead_neuron_grid([]) == ""

    def test_single_layer(self):
        svg = _svg_dead_neuron_grid([("only_layer", 25.0)])
        _valid_svg(svg)

    def test_many_layers(self):
        data = [(f"layer.{i}", i * 2.0) for i in range(50)]
        svg = _svg_dead_neuron_grid(data, width=600)
        _valid_svg(svg)

    def test_tooltip_present(self):
        svg = _svg_dead_neuron_grid([("my_layer", 33.3)])
        assert "33.3% dead" in svg


# ---------------------------------------------------------------------------
# _svg_gauge
# ---------------------------------------------------------------------------

class TestSvgGauge:

    def test_basic(self):
        svg = _svg_gauge(75)
        _valid_svg(svg)
        assert "75" in svg

    def test_zero(self):
        svg = _svg_gauge(0)
        _valid_svg(svg)
        assert "0" in svg

    def test_hundred(self):
        svg = _svg_gauge(100)
        _valid_svg(svg)
        assert "100" in svg

    def test_clamps_above_100(self):
        svg = _svg_gauge(150)
        _valid_svg(svg)
        assert "100" in svg

    def test_clamps_below_0(self):
        svg = _svg_gauge(-10)
        _valid_svg(svg)
        assert "0" in svg

    def test_custom_size(self):
        svg = _svg_gauge(50, size=200)
        _valid_svg(svg)
        assert 'width="200"' in svg

    def test_contains_health_score_label(self):
        svg = _svg_gauge(80)
        assert "Health Score" in svg


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

class TestExtractNormData:

    def test_finds_norm_layers(self):
        sd = {
            "layers.0.attn_norm.weight": torch.ones(64),
            "layers.0.ffn_norm.weight": torch.full((64,), 1.5),
            "layers.0.q_proj.weight": torch.randn(64, 64),
        }
        norms = _extract_norm_data(sd)
        assert len(norms) == 2
        assert norms[0][0] == "layers.0.attn_norm.weight"
        assert abs(norms[0][1] - 1.0) < 1e-5
        assert abs(norms[1][1] - 1.5) < 1e-5

    def test_empty_state_dict(self):
        assert _extract_norm_data({}) == []

    def test_ignores_2d_tensors(self):
        sd = {"layernorm.weight": torch.randn(32, 32)}
        assert _extract_norm_data(sd) == []


class TestExtractDeadNeuronData:

    def test_extracts_dead_rows(self):
        findings = [
            Finding("dead_neurons", "ERROR", "layer.0.weight",
                    {"dim": "rows", "pct": 15.0}),
            Finding("dead_neurons", "WARN", "layer.1.weight",
                    {"dim": "cols", "pct": 5.0}),  # not rows
            Finding("norm_drift", "WARN", "norm.weight", {}),
        ]
        data = _extract_dead_neuron_data(findings)
        assert len(data) == 1
        assert data[0] == ("layer.0.weight", 15.0)

    def test_empty_findings(self):
        assert _extract_dead_neuron_data([]) == []


class TestSampleWeights:

    def test_basic(self):
        tensors = [torch.randn(100, 100)]
        vals = _sample_weights({}, tensors, max_samples=500)
        assert len(vals) <= 500
        assert len(vals) > 0

    def test_empty(self):
        assert _sample_weights({}, []) == []

    def test_small_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        vals = _sample_weights({}, [t], max_samples=10000)
        assert len(vals) == 3


# ---------------------------------------------------------------------------
# Full report generation still works
# ---------------------------------------------------------------------------

class TestGenerateReportWithViz:

    def test_report_generates_valid_html(self, tiny_state_dict):
        hs = HealthScore(overall=82, categories={"weights": 85, "stability": 80}, grade="B")
        meta = ModelMeta(source="test", num_params=1000, num_tensors=10,
                         hidden_size=64, num_layers=2)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(tiny_state_dict, [], [], hs, meta, path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
            assert "<svg" in html  # gauge at minimum
            assert "Health Score" in html
            assert "</html>" in html
        finally:
            os.unlink(path)

    def test_report_with_findings(self, tiny_state_dict):
        findings = [
            Finding("dead_neurons", "ERROR", "layers.0.mlp.down_proj.weight",
                    {"dim": "rows", "pct": 10.0, "count": 3}),
            Finding("norm_drift", "WARN", "final_norm.weight",
                    {"mean": 2.5}),
        ]
        hs = HealthScore(overall=55, categories={"weights": 50, "stability": 60}, grade="D")
        meta = ModelMeta(source="test", num_params=1000, num_tensors=10,
                         hidden_size=64, num_layers=2)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(tiny_state_dict, findings, [], hs, meta, path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "Dead Neuron Map" in html
            assert "Layer Norm Drift" in html
            assert html.count("<svg") >= 3  # gauge + histograms + norm + dead
        finally:
            os.unlink(path)

    def test_report_no_findings(self, tiny_state_dict):
        hs = HealthScore(overall=100, categories={"weights": 100}, grade="A")
        meta = ModelMeta(source="test", num_params=500, num_tensors=5,
                         hidden_size=64, num_layers=1)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(tiny_state_dict, [], [], hs, meta, path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            # Should not have dead neuron or norm drift sections
            assert "Dead Neuron Map" not in html
            # Norm section depends on whether state_dict has norm layers
            assert "<!DOCTYPE html>" in html
        finally:
            os.unlink(path)

    def test_report_with_empty_state_dict(self):
        hs = HealthScore(overall=100, categories={}, grade="A")
        meta = ModelMeta(source="empty", num_params=0, num_tensors=0)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report({}, [], [], hs, meta, path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "<!DOCTYPE html>" in html
        finally:
            os.unlink(path)

    def test_report_with_attention_findings(self, tiny_state_dict):
        findings = [
            Finding("attention_imbalance", "WARN", "layers.0.attention",
                    {"entropy": 0.3, "heads": 4}),
        ]
        hs = HealthScore(overall=70, categories={"attention": 60}, grade="C")
        meta = ModelMeta(source="test", num_params=1000, num_tensors=10,
                         hidden_size=64, num_layers=2)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(tiny_state_dict, findings, [], hs, meta, path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "Attention Health" in html
        finally:
            os.unlink(path)

    def test_report_with_compare_data(self, tiny_state_dict):
        hs = HealthScore(overall=82, categories={"weights": 85, "stability": 80}, grade="B")
        meta = ModelMeta(source="test", num_params=1000, num_tensors=10,
                         hidden_size=64, num_layers=2)

        class CompareData:
            before = HealthScore(overall=55, categories={"weights": 50, "stability": 60}, grade="D")
            after = HealthScore(overall=82, categories={"weights": 85, "stability": 80}, grade="B")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(tiny_state_dict, [], [], hs, meta, path,
                            compare_data=CompareData())
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "Before / After Treatment" in html
            assert "Before" in html
            assert "After" in html
        finally:
            os.unlink(path)

    def test_report_with_neuron_histogram(self, tiny_state_dict):
        findings = [
            Finding("dead_neurons", "ERROR", "layer.0.weight",
                    {"dim": "rows", "pct": 15.0}),
            Finding("dead_neurons", "WARN", "layer.1.weight",
                    {"dim": "rows", "pct": 5.0}),
            Finding("dead_neurons", "WARN", "layer.2.weight",
                    {"dim": "cols", "pct": 8.0}),
        ]
        hs = HealthScore(overall=60, categories={"neurons": 50}, grade="D")
        meta = ModelMeta(source="test", num_params=1000, num_tensors=10,
                         hidden_size=64, num_layers=2)
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            generate_report(tiny_state_dict, findings, [], hs, meta, path)
            with open(path, encoding="utf-8") as f:
                html = f.read()
            assert "Neuron Health Distribution" in html
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# _svg_attention_entropy_heatmap
# ---------------------------------------------------------------------------

class TestSvgAttentionEntropyHeatmap:

    def test_basic(self):
        findings = [
            Finding("attention_imbalance", "WARN", "layers.0.attention",
                    {"entropy": 0.3, "heads": 4}),
            Finding("head_redundancy", "ERROR", "layers.1.attention",
                    {"redundant_heads": 2}),
            Finding("attention_imbalance", "INFO", "layers.2.attention",
                    {"entropy": 1.5}),
        ]
        svg = _svg_attention_entropy_heatmap(findings)
        _valid_svg(svg)
        assert "rect" in svg
        assert "title" in svg

    def test_empty_returns_empty(self):
        assert _svg_attention_entropy_heatmap([]) == ""

    def test_no_relevant_findings_returns_empty(self):
        findings = [
            Finding("dead_neurons", "ERROR", "layer.0.weight",
                    {"dim": "rows", "pct": 10.0}),
        ]
        assert _svg_attention_entropy_heatmap(findings) == ""

    def test_single_finding(self):
        findings = [
            Finding("attention_imbalance", "WARN", "layers.0.attn",
                    {"entropy": 0.5}),
        ]
        svg = _svg_attention_entropy_heatmap(findings)
        _valid_svg(svg)

    def test_all_severities_colored(self):
        findings = [
            Finding("attention_imbalance", "ERROR", "l.0", {}),
            Finding("attention_imbalance", "WARN", "l.1", {}),
            Finding("head_redundancy", "INFO", "l.2", {}),
        ]
        svg = _svg_attention_entropy_heatmap(findings)
        _valid_svg(svg)
        # Red for ERROR, orange for WARN, green for INFO
        assert "#f44336" in svg
        assert "#ff9800" in svg
        assert "#4caf50" in svg

    def test_custom_dimensions(self):
        findings = [
            Finding("attention_imbalance", "WARN", "l.0", {}),
        ]
        svg = _svg_attention_entropy_heatmap(findings, width=600, height=300)
        _valid_svg(svg)
        assert 'width="600"' in svg


# ---------------------------------------------------------------------------
# _svg_before_after_bars
# ---------------------------------------------------------------------------

class TestSvgBeforeAfterBars:

    def test_basic(self):
        before = {"weights": 50, "stability": 60, "neurons": 40}
        after = {"weights": 85, "stability": 75, "neurons": 90}
        svg = _svg_before_after_bars(before, after)
        _valid_svg(svg)
        assert "rect" in svg
        assert "Before" in svg
        assert "After" in svg

    def test_empty_before_returns_empty(self):
        assert _svg_before_after_bars({}, {"weights": 80}) == ""

    def test_empty_after_returns_empty(self):
        assert _svg_before_after_bars({"weights": 80}, {}) == ""

    def test_both_empty_returns_empty(self):
        assert _svg_before_after_bars({}, {}) == ""

    def test_single_category(self):
        svg = _svg_before_after_bars({"weights": 50}, {"weights": 80})
        _valid_svg(svg)
        assert "+30" in svg

    def test_negative_delta(self):
        svg = _svg_before_after_bars({"weights": 80}, {"weights": 75})
        _valid_svg(svg)
        assert "-5" in svg

    def test_zero_delta(self):
        svg = _svg_before_after_bars({"weights": 80}, {"weights": 80})
        _valid_svg(svg)
        assert "+0" in svg

    def test_mismatched_keys(self):
        before = {"weights": 50, "stability": 60}
        after = {"weights": 80, "neurons": 70}
        svg = _svg_before_after_bars(before, after)
        _valid_svg(svg)
        # Should contain all three categories
        assert "neurons" in svg
        assert "stability" in svg
        assert "weights" in svg

    def test_custom_dimensions(self):
        svg = _svg_before_after_bars({"a": 50}, {"a": 80}, width=600, height=300)
        _valid_svg(svg)
        assert 'width="600"' in svg
        assert 'height="300"' in svg


# ---------------------------------------------------------------------------
# _svg_neuron_activation_histogram
# ---------------------------------------------------------------------------

class TestSvgNeuronActivationHistogram:

    def test_basic(self):
        findings = [
            Finding("dead_neurons", "ERROR", "l.0", {"pct": 5.0}),
            Finding("dead_neurons", "ERROR", "l.1", {"pct": 15.0}),
            Finding("dead_neurons", "WARN", "l.2", {"pct": 2.0}),
            Finding("dead_neurons", "WARN", "l.3", {"pct": 25.0}),
            Finding("dead_neurons", "WARN", "l.4", {"pct": 8.0}),
        ]
        svg = _svg_neuron_activation_histogram(findings)
        _valid_svg(svg)
        assert "rect" in svg

    def test_empty_returns_empty(self):
        assert _svg_neuron_activation_histogram([]) == ""

    def test_no_dead_neurons_returns_empty(self):
        findings = [
            Finding("norm_drift", "WARN", "norm.weight", {"mean": 2.5}),
        ]
        assert _svg_neuron_activation_histogram(findings) == ""

    def test_single_finding(self):
        findings = [
            Finding("dead_neurons", "ERROR", "l.0", {"pct": 10.0}),
        ]
        svg = _svg_neuron_activation_histogram(findings)
        _valid_svg(svg)

    def test_all_same_values(self):
        findings = [
            Finding("dead_neurons", "WARN", f"l.{i}", {"pct": 5.0})
            for i in range(10)
        ]
        svg = _svg_neuron_activation_histogram(findings)
        _valid_svg(svg)
        assert "constant" in svg

    def test_ignores_findings_without_pct(self):
        findings = [
            Finding("dead_neurons", "WARN", "l.0", {"dim": "rows"}),  # no pct
            Finding("dead_neurons", "WARN", "l.1", {"pct": 10.0}),
        ]
        svg = _svg_neuron_activation_histogram(findings)
        _valid_svg(svg)

    def test_custom_dimensions(self):
        findings = [
            Finding("dead_neurons", "ERROR", "l.0", {"pct": 5.0}),
            Finding("dead_neurons", "ERROR", "l.1", {"pct": 15.0}),
        ]
        svg = _svg_neuron_activation_histogram(findings, width=500, height=150)
        _valid_svg(svg)
        assert 'width="500"' in svg
