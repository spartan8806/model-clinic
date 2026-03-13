"""HTML report generator for model-clinic.

Produces a self-contained HTML file with inline CSS and inline SVG
visualizations. No external dependencies, no JavaScript — just a
static diagnostic report.
"""

import math
import re
from collections import defaultdict
from datetime import datetime

import torch

from model_clinic._references import get_references


def _score_color(score):
    """Return CSS color for a health score."""
    if score >= 90:
        return "#4caf50"
    if score >= 65:
        return "#ffeb3b"
    if score >= 50:
        return "#ff9800"
    return "#f44336"


def _severity_color(severity):
    """Return CSS color for a finding severity."""
    return {"ERROR": "#f44336", "WARN": "#ffeb3b", "INFO": "#64b5f6"}.get(
        severity, "#888"
    )


def _risk_color(risk):
    """Return CSS color for a prescription risk level."""
    return {"high": "#f44336", "medium": "#ff9800", "low": "#4caf50"}.get(
        risk, "#888"
    )


def _fmt(n, decimals=2):
    """Format a number for display."""
    if isinstance(n, int):
        return f"{n:,}"
    if isinstance(n, float):
        if abs(n) < 0.01 and n != 0:
            return f"{n:.2e}"
        return f"{n:,.{decimals}f}"
    return str(n)


def _esc(text):
    """Escape HTML entities."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# SVG visualization functions
# ---------------------------------------------------------------------------

def _svg_histogram(values, width=200, height=80, bins=30, color="#4caf50"):
    """Generate an inline SVG histogram from a list of values.

    Args:
        values: list of numeric values to plot.
        width: SVG width in pixels.
        height: SVG height in pixels.
        bins: number of histogram bins.
        color: fill color for the bars.

    Returns:
        An SVG string, or an empty string if values is empty.
    """
    if not values:
        return ""

    v_min = min(values)
    v_max = max(values)

    # Degenerate case: all values identical
    if v_max == v_min:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
            f'<rect x="0" y="0" width="{width}" height="{height}" '
            f'fill="{color}" opacity="0.3" rx="2"/>'
            f'<text x="{width // 2}" y="{height // 2 + 4}" '
            f'text-anchor="middle" fill="#aaa" font-size="10" '
            f'font-family="sans-serif">constant</text>'
            f'</svg>'
        )

    # Build histogram counts
    counts = [0] * bins
    bin_width = (v_max - v_min) / bins
    for v in values:
        idx = int((v - v_min) / bin_width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1

    max_count = max(counts) if counts else 1
    bar_w = width / bins
    padding_top = 4
    usable_h = height - padding_top

    bars = []
    for i, c in enumerate(counts):
        bar_h = (c / max_count) * usable_h if max_count > 0 else 0
        x = i * bar_w
        y = height - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" opacity="0.8"/>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        + "".join(bars)
        + '</svg>'
    )


def _svg_norm_bars(norm_data, width=400, height=None):
    """Bar chart of LayerNorm weight means.

    Args:
        norm_data: list of (name, mean_value) tuples.
        width: SVG width in pixels.
        height: SVG height in pixels (auto-calculated if None).

    Returns:
        An SVG string, or an empty string if norm_data is empty.
    """
    if not norm_data:
        return ""

    bar_h = 18
    gap = 4
    label_w = 180
    bar_area_w = width - label_w - 40  # 40px for value label
    row_h = bar_h + gap
    if height is None:
        height = row_h * len(norm_data) + gap

    max_val = max(abs(v) for _, v in norm_data) if norm_data else 1.0
    if max_val == 0:
        max_val = 1.0
    # Scale so that 1.0 occupies half the bar area
    scale_ref = max(max_val, 2.0)

    rows = []
    for i, (name, val) in enumerate(norm_data):
        y = i * row_h + gap
        bar_len = (abs(val) / scale_ref) * bar_area_w

        # Color: green near 1.0, yellow when drifting, red when far
        drift = abs(val - 1.0)
        if drift < 0.1:
            bar_color = "#4caf50"
        elif drift < 0.5:
            bar_color = "#ffeb3b"
        else:
            bar_color = "#f44336"

        # Truncate label
        short = name if len(name) <= 28 else "..." + name[-25:]
        rows.append(
            f'<text x="{label_w - 4}" y="{y + bar_h - 4}" '
            f'text-anchor="end" fill="#aaa" font-size="10" '
            f'font-family="Consolas,monospace">{_esc(short)}</text>'
            f'<rect x="{label_w}" y="{y}" width="{bar_len:.1f}" '
            f'height="{bar_h}" fill="{bar_color}" opacity="0.8" rx="2"/>'
            f'<text x="{label_w + bar_len + 4}" y="{y + bar_h - 4}" '
            f'fill="#aaa" font-size="10" font-family="sans-serif">'
            f'{val:.3f}</text>'
        )

    # Reference line at 1.0
    ref_x = label_w + (1.0 / scale_ref) * bar_area_w
    ref_line = (
        f'<line x1="{ref_x:.1f}" y1="0" x2="{ref_x:.1f}" y2="{height}" '
        f'stroke="#4caf50" stroke-width="1" stroke-dasharray="3,3" opacity="0.6"/>'
        f'<text x="{ref_x:.1f}" y="{height - 2}" text-anchor="middle" '
        f'fill="#4caf50" font-size="9" font-family="sans-serif" opacity="0.7">'
        f'1.0</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        + ref_line
        + "".join(rows)
        + '</svg>'
    )


def _svg_dead_neuron_grid(dead_data, width=400, height=60):
    """Grid heatmap of dead neuron percentages per layer.

    Args:
        dead_data: list of (layer_name, dead_pct) tuples.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        An SVG string, or an empty string if dead_data is empty.
    """
    if not dead_data:
        return ""

    n = len(dead_data)
    # Compute grid layout: fit cells into width
    cell_size = min(40, max(12, (width - 20) // n))
    cols = max(1, (width - 20) // cell_size)
    grid_rows = math.ceil(n / cols)
    label_h = 14
    total_h = max(height, grid_rows * cell_size + label_h + 8)

    cells = []
    for i, (name, pct) in enumerate(dead_data):
        row = i // cols
        col = i % cols
        x = col * cell_size + 4
        y = row * cell_size + 4

        # Color intensity: 0% dead = dark/transparent, 100% dead = bright red
        intensity = min(pct / 100.0, 1.0)
        if intensity < 0.05:
            fill = "#2a2a4a"
        elif intensity < 0.3:
            fill = "#ff9800"
        else:
            fill = "#f44336"
        opacity = max(0.3, intensity)

        # Tooltip via <title>
        short = name.split(".")[-1] if "." in name else name
        cells.append(
            f'<rect x="{x}" y="{y}" width="{cell_size - 2}" '
            f'height="{cell_size - 2}" fill="{fill}" opacity="{opacity:.2f}" '
            f'rx="2"><title>{_esc(name)}: {pct:.1f}% dead</title></rect>'
        )

    # Legend
    legend_y = grid_rows * cell_size + 8
    legend = (
        f'<text x="4" y="{legend_y + 10}" fill="#666" font-size="9" '
        f'font-family="sans-serif">'
        f'{n} layers | dark = healthy, orange/red = dead neurons</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {total_h}" width="{width}" height="{total_h}">'
        + "".join(cells)
        + legend
        + '</svg>'
    )


def _svg_attention_entropy_heatmap(findings, width=400, height=200):
    """Heatmap grid of attention head health across layers.

    Visualizes attention heads colored by health status, extracted from
    findings with condition "head_redundancy" or "attention_imbalance".

    Args:
        findings: list of Finding objects.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        An SVG string, or an empty string if no relevant findings exist.
    """
    # Extract relevant findings
    attn_findings = [
        f for f in findings
        if f.condition in ("head_redundancy", "attention_imbalance")
    ]
    if not attn_findings:
        return ""

    n = len(attn_findings)
    padding = 4
    label_h = 16

    # Grid layout
    cell_size = min(36, max(14, (width - 2 * padding) // max(1, int(math.sqrt(n)))))
    cols = max(1, (width - 2 * padding) // cell_size)
    grid_rows = math.ceil(n / cols)
    total_h = max(height, grid_rows * cell_size + label_h + 2 * padding)

    cells = []
    for i, f in enumerate(attn_findings):
        row = i // cols
        col = i % cols
        x = col * cell_size + padding
        y = row * cell_size + padding

        # Determine health color from severity
        if f.severity == "ERROR":
            fill = "#f44336"
            opacity = 0.9
        elif f.severity == "WARN":
            fill = "#ff9800"
            opacity = 0.75
        else:
            fill = "#4caf50"
            opacity = 0.6

        # Build tooltip text
        short = f.param_name.split(".")[-1] if "." in f.param_name else f.param_name
        detail_parts = [f"{k}: {v}" for k, v in f.details.items()
                        if not isinstance(v, (list, dict))]
        tip = f"{f.param_name}: {f.condition}"
        if detail_parts:
            tip += " (" + ", ".join(detail_parts) + ")"

        cells.append(
            f'<rect x="{x}" y="{y}" width="{cell_size - 2}" '
            f'height="{cell_size - 2}" fill="{fill}" opacity="{opacity:.2f}" '
            f'rx="2"><title>{_esc(tip)}</title></rect>'
        )

    # Legend
    legend_y = grid_rows * cell_size + padding + 4
    legend = (
        f'<text x="{padding}" y="{legend_y + 10}" fill="#666" font-size="9" '
        f'font-family="sans-serif">'
        f'{n} attention findings | green = info, orange = warn, red = error</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {total_h}" width="{width}" height="{total_h}">'
        + "".join(cells)
        + legend
        + '</svg>'
    )


def _svg_before_after_bars(before_scores, after_scores, width=400, height=200):
    """Side-by-side bar chart comparing category scores before and after treatment.

    Args:
        before_scores: dict of category name -> score (0-100).
        after_scores: dict of category name -> score (0-100).
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        An SVG string, or an empty string if either dict is empty.
    """
    if not before_scores or not after_scores:
        return ""

    # Use union of keys, sorted
    categories = sorted(set(before_scores) | set(after_scores))
    if not categories:
        return ""

    n = len(categories)
    padding_left = 100
    padding_right = 50
    padding_top = 20
    padding_bottom = 10
    bar_area_w = width - padding_left - padding_right
    usable_h = height - padding_top - padding_bottom

    group_h = usable_h / n
    bar_h = max(4, (group_h - 8) / 2)  # Two bars per group

    elements = []
    for i, cat in enumerate(categories):
        y_base = padding_top + i * group_h
        before_val = before_scores.get(cat, 0)
        after_val = after_scores.get(cat, 0)
        delta = after_val - before_val

        # Category label
        elements.append(
            f'<text x="{padding_left - 6}" y="{y_base + group_h / 2 + 3}" '
            f'text-anchor="end" fill="#aaa" font-size="10" '
            f'font-family="sans-serif">{_esc(cat)}</text>'
        )

        # Before bar (grey)
        bw = max(0, (before_val / 100.0) * bar_area_w)
        elements.append(
            f'<rect x="{padding_left}" y="{y_base + 2}" '
            f'width="{bw:.1f}" height="{bar_h:.1f}" '
            f'fill="#666" opacity="0.6" rx="2">'
            f'<title>Before: {before_val}</title></rect>'
        )

        # After bar (colored by score)
        aw = max(0, (after_val / 100.0) * bar_area_w)
        after_color = _score_color(after_val)
        elements.append(
            f'<rect x="{padding_left}" y="{y_base + bar_h + 4}" '
            f'width="{aw:.1f}" height="{bar_h:.1f}" '
            f'fill="{after_color}" opacity="0.8" rx="2">'
            f'<title>After: {after_val}</title></rect>'
        )

        # Delta label
        delta_color = "#4caf50" if delta >= 0 else "#f44336"
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        label_x = padding_left + max(bw, aw) + 6
        elements.append(
            f'<text x="{label_x:.1f}" y="{y_base + group_h / 2 + 3}" '
            f'fill="{delta_color}" font-size="10" font-weight="600" '
            f'font-family="sans-serif">{delta_str}</text>'
        )

    # Legend
    elements.append(
        f'<rect x="{padding_left}" y="{height - 8}" width="8" height="8" '
        f'fill="#666" opacity="0.6" rx="1"/>'
        f'<text x="{padding_left + 12}" y="{height - 1}" fill="#888" '
        f'font-size="9" font-family="sans-serif">Before</text>'
        f'<rect x="{padding_left + 60}" y="{height - 8}" width="8" height="8" '
        f'fill="#4caf50" opacity="0.8" rx="1"/>'
        f'<text x="{padding_left + 72}" y="{height - 1}" fill="#888" '
        f'font-size="9" font-family="sans-serif">After</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        + "".join(elements)
        + '</svg>'
    )


def _svg_neuron_activation_histogram(findings, width=300, height=100):
    """Histogram of dead neuron percentages across layers.

    Extracts dead_pct values from dead_neurons findings and renders
    a histogram using :func:`_svg_histogram`.

    Args:
        findings: list of Finding objects.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        An SVG string, or an empty string if no dead_neurons findings exist.
    """
    dead_pcts = [
        f.details.get("pct", 0)
        for f in findings
        if f.condition == "dead_neurons" and "pct" in f.details
    ]
    if not dead_pcts:
        return ""

    return _svg_histogram(dead_pcts, width=width, height=height,
                          bins=min(20, len(dead_pcts)), color="#f44336")


def _build_gate_evolution_svg(before_sd: dict, after_sd: dict, width=600, height=200) -> str:
    """SVG chart showing gate value changes between two checkpoints.

    Finds all tensors named *gate* with numel() == 1 (scalar gates).
    Shows before/after as a grouped bar chart.
    Color: red for stuck_closed (< -5), green for healthy (-5 to 5),
    orange for stuck_open (> 5).

    Args:
        before_sd: state dict of the before checkpoint.
        after_sd: state dict of the after checkpoint.
        width: SVG width in pixels.
        height: SVG height in pixels.

    Returns:
        An SVG string, or an empty string if no scalar gates found.
    """
    def _gate_color(val):
        if val < -5:
            return "#f44336"   # red — stuck closed
        if val > 5:
            return "#ff9800"   # orange — stuck open
        return "#4caf50"       # green — healthy

    # Collect all scalar gates present in either checkpoint
    all_gate_names = set()
    for name, tensor in before_sd.items():
        if isinstance(tensor, torch.Tensor) and "gate" in name.lower() and tensor.numel() == 1:
            all_gate_names.add(name)
    for name, tensor in after_sd.items():
        if isinstance(tensor, torch.Tensor) and "gate" in name.lower() and tensor.numel() == 1:
            all_gate_names.add(name)

    if not all_gate_names:
        return ""

    gates = sorted(all_gate_names)
    n = len(gates)

    padding_left = 200
    padding_right = 60
    padding_top = 20
    padding_bottom = 24  # room for legend
    bar_area_w = width - padding_left - padding_right
    usable_h = height - padding_top - padding_bottom

    group_h = usable_h / n
    bar_h = max(4, (group_h - 6) / 2)

    # Reference lines at -5 and +5 (stuck thresholds)
    # Map gate value to x position: clamp to [-10, 10] for display
    display_min = -10.0
    display_max = 10.0
    display_range = display_max - display_min

    def _val_to_x(val):
        clamped = max(display_min, min(display_max, val))
        frac = (clamped - display_min) / display_range
        return padding_left + frac * bar_area_w

    elements = []

    # Background grid lines at -5, 0, +5
    for marker in (-5, 0, 5):
        gx = _val_to_x(marker)
        elements.append(
            f'<line x1="{gx:.1f}" y1="{padding_top}" x2="{gx:.1f}" '
            f'y2="{height - padding_bottom}" stroke="#333" stroke-width="1" '
            f'stroke-dasharray="3,3" opacity="0.7"/>'
            f'<text x="{gx:.1f}" y="{padding_top - 4}" text-anchor="middle" '
            f'fill="#555" font-size="9" font-family="sans-serif">{marker}</text>'
        )

    # Zero center line (bolder)
    zero_x = _val_to_x(0)
    elements.append(
        f'<line x1="{zero_x:.1f}" y1="{padding_top}" x2="{zero_x:.1f}" '
        f'y2="{height - padding_bottom}" stroke="#444" stroke-width="1.5" opacity="0.9"/>'
    )

    for i, name in enumerate(gates):
        y_base = padding_top + i * group_h

        before_t = before_sd.get(name)
        after_t = after_sd.get(name)
        before_val = before_t.item() if isinstance(before_t, torch.Tensor) else None
        after_val = after_t.item() if isinstance(after_t, torch.Tensor) else None

        # Truncate label
        short = name if len(name) <= 32 else "..." + name[-29:]
        elements.append(
            f'<text x="{padding_left - 6}" y="{y_base + group_h / 2 + 3}" '
            f'text-anchor="end" fill="#aaa" font-size="10" '
            f'font-family="Consolas,monospace">{_esc(short)}</text>'
        )

        # Before bar (grey outline + colored dot at value)
        if before_val is not None:
            bx = _val_to_x(before_val)
            bc = _gate_color(before_val)
            elements.append(
                f'<rect x="{min(zero_x, bx):.1f}" y="{y_base + 2}" '
                f'width="{abs(bx - zero_x):.1f}" height="{bar_h:.1f}" '
                f'fill="#555" opacity="0.5" rx="2">'
                f'<title>Before: {before_val:.3f}</title></rect>'
                f'<circle cx="{bx:.1f}" cy="{y_base + 2 + bar_h / 2:.1f}" r="4" '
                f'fill="{bc}" opacity="0.85">'
                f'<title>Before: {before_val:.3f}</title></circle>'
            )

        # After bar (colored)
        if after_val is not None:
            ax = _val_to_x(after_val)
            ac = _gate_color(after_val)
            y_after = y_base + bar_h + 4
            elements.append(
                f'<rect x="{min(zero_x, ax):.1f}" y="{y_after:.1f}" '
                f'width="{abs(ax - zero_x):.1f}" height="{bar_h:.1f}" '
                f'fill="{ac}" opacity="0.7" rx="2">'
                f'<title>After: {after_val:.3f}</title></rect>'
                f'<circle cx="{ax:.1f}" cy="{y_after + bar_h / 2:.1f}" r="4" '
                f'fill="{ac}" opacity="0.95">'
                f'<title>After: {after_val:.3f}</title></circle>'
            )

        # Value labels at right edge
        label_parts = []
        if before_val is not None:
            label_parts.append(f"B:{before_val:.2f}")
        if after_val is not None:
            label_parts.append(f"A:{after_val:.2f}")
        label_text = "  ".join(label_parts)
        elements.append(
            f'<text x="{padding_left + bar_area_w + 4}" '
            f'y="{y_base + group_h / 2 + 3}" '
            f'fill="#aaa" font-size="9" font-family="Consolas,monospace">'
            f'{_esc(label_text)}</text>'
        )

    # Legend at bottom
    legend_y = height - padding_bottom + 6
    elements.append(
        f'<circle cx="{padding_left + 8}" cy="{legend_y + 4}" r="4" fill="#4caf50"/>'
        f'<text x="{padding_left + 16}" y="{legend_y + 8}" fill="#888" '
        f'font-size="9" font-family="sans-serif">Healthy (-5 to 5)</text>'
        f'<circle cx="{padding_left + 120}" cy="{legend_y + 4}" r="4" fill="#f44336"/>'
        f'<text x="{padding_left + 128}" y="{legend_y + 8}" fill="#888" '
        f'font-size="9" font-family="sans-serif">Stuck closed (&lt;-5)</text>'
        f'<circle cx="{padding_left + 240}" cy="{legend_y + 4}" r="4" fill="#ff9800"/>'
        f'<text x="{padding_left + 248}" y="{legend_y + 8}" fill="#888" '
        f'font-size="9" font-family="sans-serif">Stuck open (&gt;5)</text>'
        f'<rect x="{padding_left + 340}" y="{legend_y}" width="8" height="4" '
        f'fill="#555" opacity="0.5" rx="1"/>'
        f'<text x="{padding_left + 352}" y="{legend_y + 8}" fill="#888" '
        f'font-size="9" font-family="sans-serif">Before</text>'
    )

    total_h = max(height, padding_top + n * group_h + padding_bottom)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {total_h:.0f}" width="{width}" height="{total_h:.0f}">'
        + "".join(elements)
        + '</svg>'
    )


def _svg_gauge(score, size=120):
    """Circular gauge showing health score 0-100.

    Draws a 240-degree arc that fills based on the score, with color
    transitioning from red (0) through yellow to green (100).

    Args:
        score: numeric health score, 0-100.
        size: SVG width and height in pixels.

    Returns:
        An SVG string.
    """
    score = max(0, min(100, score))
    cx = size / 2
    cy = size / 2
    r = size * 0.38
    stroke_w = size * 0.08

    # Arc spans 240 degrees, from 150 to 390 (or -210 to 30)
    start_angle = 150
    sweep = 240

    def _polar(angle_deg):
        rad = math.radians(angle_deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    # Background arc (full 240 degrees)
    bg_end_angle = start_angle + sweep
    bg_x1, bg_y1 = _polar(start_angle)
    bg_x2, bg_y2 = _polar(bg_end_angle)
    large_flag = 1 if sweep > 180 else 0
    bg_path = (
        f'M {bg_x1:.2f} {bg_y1:.2f} '
        f'A {r:.2f} {r:.2f} 0 {large_flag} 1 {bg_x2:.2f} {bg_y2:.2f}'
    )

    # Foreground arc (proportional to score)
    fg_sweep = sweep * (score / 100.0)
    fg_end_angle = start_angle + fg_sweep
    fg_x1, fg_y1 = _polar(start_angle)
    fg_x2, fg_y2 = _polar(fg_end_angle)
    fg_large = 1 if fg_sweep > 180 else 0

    # Handle zero score (no arc to draw)
    if score == 0:
        fg_path_el = ""
    else:
        fg_path = (
            f'M {fg_x1:.2f} {fg_y1:.2f} '
            f'A {r:.2f} {r:.2f} 0 {fg_large} 1 {fg_x2:.2f} {fg_y2:.2f}'
        )
        fg_path_el = (
            f'<path d="{fg_path}" fill="none" '
            f'stroke="{_score_color(score)}" '
            f'stroke-width="{stroke_w:.1f}" stroke-linecap="round"/>'
        )

    # Score text
    font_size = size * 0.28
    grade_size = size * 0.12

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {size} {size}" width="{size}" height="{size}">'
        f'<path d="{bg_path}" fill="none" stroke="#2a2a4a" '
        f'stroke-width="{stroke_w:.1f}" stroke-linecap="round"/>'
        + fg_path_el
        + f'<text x="{cx}" y="{cy + font_size * 0.15}" text-anchor="middle" '
        f'fill="{_score_color(score)}" font-size="{font_size:.0f}" '
        f'font-weight="700" font-family="sans-serif">{score}</text>'
        f'<text x="{cx}" y="{cy + font_size * 0.15 + grade_size + 4}" '
        f'text-anchor="middle" fill="#aaa" font-size="{grade_size:.0f}" '
        f'font-family="sans-serif">Health Score</text>'
        + '</svg>'
    )


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_norm_data(state_dict):
    """Find all norm layers and their mean values.

    Args:
        state_dict: dict of name -> Tensor.

    Returns:
        List of (name, mean_value) tuples for norm weight tensors.
    """
    norms = []
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() == 1 and any(
            kw in name.lower()
            for kw in ["norm.weight", "layernorm", "rmsnorm"]
        ):
            norms.append((name, tensor.float().mean().item()))
    return norms


def _extract_dead_neuron_data(findings):
    """Extract dead neuron percentages from findings.

    Args:
        findings: list of Finding objects.

    Returns:
        List of (param_name, dead_pct) tuples.
    """
    data = []
    for f in findings:
        if f.condition == "dead_neurons" and f.details.get("dim") == "rows":
            data.append((f.param_name, f.details.get("pct", 0)))
    return data


def _sample_weights(state_dict, tensors, max_samples=10000):
    """Sample weight values from a list of tensors.

    Args:
        state_dict: not used directly (kept for API consistency).
        tensors: list of Tensor objects to sample from.
        max_samples: maximum number of values to return.

    Returns:
        List of float values sampled from the tensors.
    """
    all_vals = []
    total = sum(t.numel() for t in tensors)
    if total == 0:
        return []

    for t in tensors:
        flat = t.detach().float().flatten()
        # Proportional sampling
        n_take = max(1, int(max_samples * t.numel() / total))
        if flat.numel() <= n_take:
            all_vals.extend(flat.tolist())
        else:
            indices = torch.randperm(flat.numel())[:n_take]
            all_vals.extend(flat[indices].tolist())

    # Final cap
    if len(all_vals) > max_samples:
        all_vals = all_vals[:max_samples]
    return all_vals


# ---------------------------------------------------------------------------
# Existing report helpers
# ---------------------------------------------------------------------------

def weight_summary(state_dict):
    """Group params by layer pattern and compute basic stats.

    Returns list of dicts with keys:
        group, count, total_elements, mean_norm, max_norm, tensors
    """
    groups = defaultdict(list)

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        # Group by first two dotted segments (e.g. "model.layers")
        # or by slash-separated prefix for composite checkpoints
        parts = re.split(r"[./]", name)
        if len(parts) >= 2:
            # Collapse numeric layer indices
            prefix = parts[0]
            second = parts[1] if not parts[1].isdigit() else "*"
            group = f"{prefix}.{second}"
        else:
            group = parts[0]
        groups[group].append(tensor)

    rows = []
    for group, tensors in sorted(groups.items()):
        norms = []
        for t in tensors:
            n = t.float().norm().item()
            if math.isfinite(n):
                norms.append(n)
        rows.append({
            "group": group,
            "count": len(tensors),
            "total_elements": sum(t.numel() for t in tensors),
            "mean_norm": sum(norms) / len(norms) if norms else 0,
            "max_norm": max(norms) if norms else 0,
            "tensors": tensors,
        })
    return rows


def _build_suggested_fixes(findings, prescriptions, model_path="<model>"):
    """Build a suggested-actions card grouped by priority.

    Returns an HTML string for the card, or empty string if nothing to show.
    Groups by condition type, not per-tensor, so 19 norm_drift findings become
    one action item: "Reset 19 norm layer weights".
    """
    if not findings:
        return (
            '<div class="suggest-box suggest-ok">'
            '<div class="suggest-title">All Clear</div>'
            '<p style="color:#aaa;margin:0">No issues detected. Model looks healthy.</p>'
            '</div>'
        )

    # Separate advisories (no real fix) from actionable prescriptions
    advisory_conds = set()
    actionable = {}  # condition -> {count, severity, desc, risk, cmd_flag}
    for rx in prescriptions:
        cond = rx.finding.condition
        is_advisory = rx.name.startswith("advisory_")
        if is_advisory:
            advisory_conds.add(cond)
        else:
            if cond not in actionable:
                actionable[cond] = {
                    "count": 0,
                    "severity": rx.finding.severity,
                    "desc": rx.description,
                    "risk": rx.risk,
                    "name": rx.name,
                }
            actionable[cond]["count"] += 1

    # Count findings per condition and severity
    from collections import Counter
    cond_severity = {}
    cond_count = Counter()
    for f in findings:
        cond_count[f.condition] += 1
        # Keep worst severity per condition
        prev = cond_severity.get(f.condition, "INFO")
        if f.severity == "ERROR" or (f.severity == "WARN" and prev == "INFO"):
            cond_severity[f.condition] = f.severity

    # Separate by priority
    errors = [(c, actionable[c]) for c in actionable if cond_severity.get(c) == "ERROR"]
    warns  = [(c, actionable[c]) for c in actionable if cond_severity.get(c) == "WARN"]
    advisories = [(c, cond_count[c]) for c in advisory_conds if c in cond_count]

    # CLI command hints
    cli_conservative = f'model-clinic treat "{model_path}" --conservative --save fixed.pt'
    cli_full         = f'model-clinic treat "{model_path}" --save fixed.pt'

    def _item(label, count, risk, desc):
        risk_color = {"low": "#4caf50", "medium": "#ff9800", "high": "#f44336"}.get(risk, "#888")
        count_str = f" &times;{count}" if count > 1 else ""
        return (
            f'<div class="suggest-item">'
            f'<span class="suggest-badge" style="background:{risk_color}">{risk.upper()}</span>'
            f'<span class="suggest-label">{_esc(label)}{count_str}</span>'
            f'<span class="suggest-desc">{_esc(desc[:80])}{"..." if len(desc) > 80 else ""}</span>'
            f'</div>'
        )

    html = '<div class="suggest-box">'

    if errors:
        html += '<div class="suggest-title suggest-critical">Critical — Fix Before Use</div>'
        for cond, info in sorted(errors, key=lambda x: -x[1]["count"]):
            html += _item(cond.replace("_", " ").title(), info["count"], info["risk"], info["desc"])

    if warns:
        html += '<div class="suggest-title suggest-warn">Recommended Fixes</div>'
        for cond, info in sorted(warns, key=lambda x: -x[1]["count"]):
            html += _item(cond.replace("_", " ").title(), info["count"], info["risk"], info["desc"])

    if advisories:
        html += '<div class="suggest-title suggest-info">Monitor / Manual Review</div>'
        for cond, count in sorted(advisories, key=lambda x: -x[1]):
            count_str = f" &times;{count}" if count > 1 else ""
            html += (
                f'<div class="suggest-item suggest-advisory">'
                f'<span class="suggest-badge" style="background:#607d8b">ADVISORY</span>'
                f'<span class="suggest-label">{_esc(cond.replace("_", " ").title())}{count_str}</span>'
                f'<span class="suggest-desc">No automatic fix — review manually</span>'
                f'</div>'
            )

    # CLI commands
    has_medium_or_high = any(
        info["risk"] in ("medium", "high")
        for info in actionable.values()
    )
    html += '<div class="suggest-cli">'
    if warns or errors:
        if has_medium_or_high:
            html += (
                f'<div class="suggest-cmd-label">Conservative (low-risk only):</div>'
                f'<code class="suggest-cmd">{_esc(cli_conservative)}</code>'
                f'<div class="suggest-cmd-label" style="margin-top:.5rem">All fixes:</div>'
                f'<code class="suggest-cmd">{_esc(cli_full)}</code>'
            )
        else:
            html += (
                f'<div class="suggest-cmd-label">Apply all fixes:</div>'
                f'<code class="suggest-cmd">{_esc(cli_conservative)}</code>'
            )
    else:
        html += '<span style="color:#888;font-size:.85rem">No automated fixes available for these findings.</span>'
    html += '</div>'

    html += '</div>'
    return html


def _svg_mri_rank_heatmap(mri_results, width=600, cell_size=18):
    """Generate an inline SVG grid showing rank utilization per layer.

    Each cell represents one weight matrix. Color encodes rank utilization:
    dark red (0%) -> yellow (50%) -> green (100%).
    """
    if not mri_results:
        return ""

    n = len(mri_results)
    cols = max(1, width // (cell_size + 2))
    rows_count = (n + cols - 1) // cols
    height = rows_count * (cell_size + 2) + 30

    def _ru_color(ru):
        if ru >= 0.8:
            return "#4caf50"
        if ru >= 0.5:
            t = (ru - 0.5) / 0.3
            r = int(255 - t * 179)
            g = int(235 + t * (175 - 235))
            b = int(59 + t * (80 - 59))
            return f"#{r:02x}{g:02x}{b:02x}"
        if ru >= 0.1:
            t = (ru - 0.1) / 0.4
            r = int(244 + t * (255 - 244))
            g = int(67 + t * (235 - 67))
            b = int(54 + t * (59 - 54))
            return f"#{r:02x}{g:02x}{b:02x}"
        return "#b71c1c"

    rects = []
    for i, lr in enumerate(mri_results):
        col = i % cols
        row = i // cols
        x = col * (cell_size + 2)
        y = row * (cell_size + 2)
        color = _ru_color(lr.rank_utilization)
        title = (f"{lr.name}: rank {lr.numerical_rank}/{min(lr.shape)}, "
                 f"util={lr.rank_utilization:.0%}")
        rects.append(
            f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
            f'fill="{color}" rx="2"><title>{title}</title></rect>'
        )

    legend_y = rows_count * (cell_size + 2) + 5
    legend_items = [("#b71c1c", "&lt;10%"), ("#f44336", "10%"),
                    ("#ffeb3b", "50%"), ("#4caf50", "80%+")]
    legend_parts = []
    lx = 0
    for lc, lt in legend_items:
        legend_parts.append(
            f'<rect x="{lx}" y="{legend_y}" width="12" height="12" fill="{lc}" rx="1"/>'
            f'<text x="{lx + 15}" y="{legend_y + 10}" '
            f'font-size="10" fill="#aaa">{lt}</text>'
        )
        lx += 60

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        + "\n".join(rects) + "\n" + "\n".join(legend_parts) + "\n</svg>"
    )


def generate_report(state_dict, findings, prescriptions, health_score,
                    meta, output_path, compare_data=None, debug=False,
                    interactive=False, mri_results=None):
    """Generate a self-contained HTML report file.

    Args:
        state_dict: dict of name -> Tensor
        findings: list of Finding objects
        prescriptions: list of Prescription objects
        health_score: HealthScore object (overall, categories, grade)
        meta: ModelMeta object
        output_path: path to write the HTML file
        compare_data: optional object with .before and .after HealthScore attrs
        debug: if True, include raw tensor stats for verification
        interactive: if True, inject JavaScript interactivity layer (filter, search, sort)
    """
    from model_clinic import __version__

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    model_name = _esc(meta.source if meta.source != "unknown" else "Model")
    overall = health_score.overall
    grade = _esc(health_score.grade)
    color = _score_color(overall)

    # SVG gauge replaces the old score-box
    gauge_svg = _svg_gauge(overall, size=120)

    # Category bars
    cat_html = ""
    for cat_name, cat_score in health_score.categories.items():
        cc = _score_color(cat_score)
        cat_html += (
            f'<div class="cat-row">'
            f'<span class="cat-label">{_esc(cat_name)}</span>'
            f'<div class="cat-bar"><div class="cat-fill" '
            f'style="width:{cat_score}%;background:{cc}"></div></div>'
            f'<span class="cat-score" style="color:{cc}">{cat_score}</span>'
            f'</div>\n'
        )

    # Suggested fixes card
    model_path_hint = getattr(meta, "source", "<model>")
    suggested_fixes_html = _build_suggested_fixes(findings, prescriptions, model_path_hint)

    # Norm drift chart (only if norm layers exist)
    norm_data = _extract_norm_data(state_dict)
    norm_section = ""
    if norm_data:
        norm_svg = _svg_norm_bars(norm_data, width=600)
        norm_section = (
            f'\n<h2>Layer Norm Drift</h2>\n'
            f'<div class="viz-container">{norm_svg}</div>\n'
        )

    # Dead neuron grid (only if dead_neuron findings exist)
    dead_data = _extract_dead_neuron_data(findings)
    dead_section = ""
    if dead_data:
        dead_svg = _svg_dead_neuron_grid(dead_data, width=500)
        dead_section = (
            f'\n<h2>Dead Neuron Map</h2>\n'
            f'<div class="viz-container">{dead_svg}</div>\n'
        )

    # Attention health heatmap
    attn_svg = _svg_attention_entropy_heatmap(findings, width=500)
    attn_section = ""
    if attn_svg:
        attn_section = (
            f'\n<h2>Attention Health</h2>\n'
            f'<div class="viz-container">{attn_svg}</div>\n'
        )

    # Neuron health distribution histogram
    neuron_hist_svg = _svg_neuron_activation_histogram(findings, width=400, height=120)
    neuron_hist_section = ""
    if neuron_hist_svg:
        neuron_hist_section = (
            f'\n<h2>Neuron Health Distribution</h2>\n'
            f'<div class="viz-container">{neuron_hist_svg}</div>\n'
        )

    # Before/after comparison (only if compare_data provided)
    compare_section = ""
    if compare_data is not None:
        before = getattr(compare_data, 'before', None)
        after = getattr(compare_data, 'after', None)
        before_cats = before.categories if before else {}
        after_cats = after.categories if after else {}
        ba_svg = _svg_before_after_bars(before_cats, after_cats, width=500, height=250)
        if ba_svg:
            compare_section = (
                f'\n<h2>Before / After Treatment</h2>\n'
                f'<div class="viz-container">{ba_svg}</div>\n'
            )

        # Gate evolution chart (needs before/after state dicts on compare_data)
        before_sd = getattr(compare_data, 'before_sd', None)
        after_sd = getattr(compare_data, 'after_sd', None)
        if before_sd is not None and after_sd is not None:
            gate_svg = _build_gate_evolution_svg(before_sd, after_sd, width=600)
            if gate_svg:
                gate_h = max(200, 60 * sum(
                    1 for name, t in {**before_sd, **after_sd}.items()
                    if isinstance(t, torch.Tensor) and "gate" in name.lower() and t.numel() == 1
                ) + 44)
                gate_svg = _build_gate_evolution_svg(before_sd, after_sd, width=600, height=gate_h)
                compare_section += (
                    f'\n<h2>Gate Evolution</h2>\n'
                    f'<div class="viz-container">{gate_svg}</div>\n'
                )

    # MRI rank utilization heatmap (only if mri_results provided)
    mri_section = ""
    if mri_results:
        from model_clinic._mri import mri_summary as _mri_summary
        mri_svg = _svg_mri_rank_heatmap(mri_results, width=600)
        ms = _mri_summary(mri_results)
        mri_stats = (
            f'<div style="font-size:.85rem;color:#aaa;margin-top:.75rem">'
            f'Analyzed {ms["analyzed_layers"]} weight matrices. '
            f'Mean rank utilization: {ms["mean_rank_utilization"]:.0%}. '
            f'Low-rank: {ms["n_low_rank"]}. '
            f'Degenerate: {ms["n_degenerate"]}. '
            f'Information score: {ms["information_score"]}/100.'
            f'</div>'
        )
        mri_section = (
            f'\n<h2>MRI Analysis</h2>\n'
            f'<div class="viz-container">{mri_svg}{mri_stats}</div>\n'
        )

    # Findings table rows
    # Import here to avoid circular at module level
    from model_clinic._health_score import _categorize as _cat_fn
    findings_rows = ""
    for f in sorted(findings, key=lambda x: {"ERROR": 0, "WARN": 1, "INFO": 2}.get(x.severity, 3)):
        sc = _severity_color(f.severity)
        details = ", ".join(f"{k}: {_fmt(v) if isinstance(v, (int, float)) else _esc(v)}"
                           for k, v in f.details.items()
                           if not isinstance(v, (list, dict)))
        refs = get_references(f.condition)
        ref_links = " ".join(
            f'<a href="{_esc(r["url"])}" style="color:#64b5f6;font-size:.8rem" '
            f'target="_blank" title="{_esc(r.get("note", ""))}">{_esc(r["title"])}</a>'
            for r in refs if r.get("url")
        )
        f_category = _cat_fn(f.condition)
        findings_rows += (
            f'<tr class="finding-row" '
            f'data-severity="{_esc(f.severity)}" '
            f'data-category="{_esc(f_category)}" '
            f'data-param="{_esc(f.param_name.lower())}">'
            f'<td><span class="badge" style="background:{sc}">{_esc(f.severity)}</span></td>'
            f'<td>{_esc(f.condition)}</td>'
            f'<td class="mono">{_esc(f.param_name)}</td>'
            f'<td>{details}</td>'
            f'<td>{ref_links if ref_links else "&mdash;"}</td>'
            f'</tr>\n'
        )

    if not findings_rows:
        findings_rows = '<tr><td colspan="5" class="empty">No issues found.</td></tr>'

    # Prescriptions table rows
    rx_rows = ""
    for rx in prescriptions:
        rc = _risk_color(rx.risk)
        rx_rows += (
            f'<tr>'
            f'<td>{_esc(rx.name)}</td>'
            f'<td><span class="badge" style="background:{rc}">{_esc(rx.risk)}</span></td>'
            f'<td>{_esc(rx.description)}</td>'
            f'<td class="mono">{_esc(rx.finding.param_name)}</td>'
            f'</tr>\n'
        )

    if not rx_rows:
        rx_rows = '<tr><td colspan="4" class="empty">No treatments recommended.</td></tr>'

    # Weight summary with inline histograms
    ws = weight_summary(state_dict)
    ws_rows = ""
    for row in ws:
        sampled = _sample_weights(state_dict, row["tensors"])
        hist_svg = _svg_histogram(sampled, width=160, height=40, bins=25)
        ws_rows += (
            f'<tr>'
            f'<td class="mono">{_esc(row["group"])}</td>'
            f'<td>{_fmt(row["count"])}</td>'
            f'<td>{_fmt(row["total_elements"])}</td>'
            f'<td>{_fmt(row["mean_norm"])}</td>'
            f'<td>{_fmt(row["max_norm"])}</td>'
            f'<td>{hist_svg}</td>'
            f'</tr>\n'
        )

    # Debug section: raw tensor stats for verification
    # Process one tensor at a time to avoid OOM on large models
    debug_section = ""
    if debug:
        import gc as _gc
        debug_lines = []
        debug_lines.append("RAW TENSOR STATS (for verification)")
        debug_lines.append("=" * 70)
        n_tensors = sum(1 for v in state_dict.values() if isinstance(v, torch.Tensor))
        debug_lines.append(f"Total tensors: {n_tensors}")
        debug_lines.append("")
        for name, tensor in sorted(state_dict.items()):
            if not isinstance(tensor, torch.Tensor):
                continue
            stats = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
            # Use in-place checks to minimize memory — avoid full .float() copy
            with torch.no_grad():
                has_nan = bool(torch.isnan(tensor).any().item())
                has_inf = bool(torch.isinf(tensor).any().item())
                stats["has_nan"] = has_nan
                stats["has_inf"] = has_inf
                # Compute stats on a sample for large tensors to avoid OOM
                t_flat = tensor.reshape(-1)
                if t_flat.numel() > 500_000:
                    idx = torch.randperm(t_flat.numel())[:500_000]
                    sample = t_flat[idx].float()
                    stats["sampled"] = True
                else:
                    sample = t_flat.float()
                finite = sample[torch.isfinite(sample)]
                if finite.numel() > 0:
                    stats["min"] = f"{finite.min().item():.6f}"
                    stats["max"] = f"{finite.max().item():.6f}"
                    stats["mean"] = f"{finite.mean().item():.6f}"
                    stats["std"] = f"{finite.std().item():.6f}" if finite.numel() > 1 else "N/A"
                    stats["norm"] = f"{finite.norm().item():.4f}"
                    if tensor.dim() >= 2:
                        # Sample rows for dead-row check on large tensors
                        n_rows = tensor.shape[0]
                        if n_rows > 1000:
                            row_sample = tensor[:1000].float()
                        else:
                            row_sample = tensor.float()
                        dead_rows = int((row_sample.norm(dim=1) < 1e-7).sum().item())
                        stats["dead_rows"] = f"{dead_rows}/{n_rows}" + (" (sampled 1K)" if n_rows > 1000 else "")
                        del row_sample
                else:
                    stats["note"] = "all NaN/Inf"
                if has_nan:
                    stats["nan_count"] = int(torch.isnan(sample).sum().item())
                if has_inf:
                    stats["inf_count"] = int(torch.isinf(sample).sum().item())
                del sample, finite
            debug_lines.append(f"{name}")
            for k, v in stats.items():
                debug_lines.append(f"  {k}: {v}")
            debug_lines.append("")

        _gc.collect()
        debug_lines.append("")
        debug_lines.append("FINDING DETAILS (raw)")
        debug_lines.append("=" * 70)
        for f in findings:
            debug_lines.append(f"[{f.severity}] {f.condition}: {f.param_name}")
            for k, v in f.details.items():
                debug_lines.append(f"  {k}: {v}")
            debug_lines.append("")

        debug_text = _esc("\n".join(debug_lines))
        debug_section = (
            f'\n<h2>Debug: Raw Tensor Stats</h2>\n'
            f'<div class="debug">{debug_text}</div>\n'
        )

    # Interactive JS toolbar for findings table
    interactive_controls_html = ""
    interactive_js_html = ""
    interactive_css = ""
    if interactive:
        interactive_css = (
            ".toolbar{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap;"
            "margin-bottom:1rem;padding:.6rem .75rem;background:#16213e;"
            "border-radius:8px;border:1px solid #2a2a4a}"
            ".toolbar-label{color:#888;font-size:.8rem;flex-shrink:0}"
            ".filter-btn{padding:.25rem .65rem;border-radius:4px;border:1px solid #333;"
            "background:#1a1a2e;color:#aaa;font-size:.78rem;cursor:pointer;"
            "transition:all .15s}"
            ".filter-btn:hover{border-color:#64b5f6;color:#64b5f6}"
            ".filter-btn.active{background:#0d2040;border-color:#64b5f6;color:#64b5f6;"
            "font-weight:600}"
            ".filter-btn.sev-error.active{border-color:#f44336;color:#f44336;"
            "background:#2a0a0a}"
            ".filter-btn.sev-warn.active{border-color:#ffeb3b;color:#ffeb3b;"
            "background:#1a1700}"
            ".filter-btn.sev-info.active{border-color:#64b5f6;color:#64b5f6;"
            "background:#0a1525}"
            ".search-box{flex:1;min-width:160px;max-width:280px;padding:.25rem .6rem;"
            "border-radius:4px;border:1px solid #333;background:#1a1a2e;color:#e0e0e0;"
            "font-size:.82rem;outline:none}"
            ".search-box:focus{border-color:#64b5f6}"
            ".sort-select{padding:.25rem .5rem;border-radius:4px;border:1px solid #333;"
            "background:#1a1a2e;color:#aaa;font-size:.78rem;cursor:pointer}"
            ".finding-row.hidden{display:none}"
        )
        interactive_controls_html = (
            '<div class="toolbar" id="findings-toolbar">'
            '<span class="toolbar-label">Filter:</span>'
            '<button class="filter-btn active sev-all" data-filter-sev="ALL">All</button>'
            '<button class="filter-btn sev-error" data-filter-sev="ERROR">ERROR</button>'
            '<button class="filter-btn sev-warn" data-filter-sev="WARN">WARN</button>'
            '<button class="filter-btn sev-info" data-filter-sev="INFO">INFO</button>'
            '<span class="toolbar-label" style="margin-left:.5rem">Category:</span>'
            '<button class="filter-btn active cat-all" data-filter-cat="ALL">All</button>'
            '<button class="filter-btn" data-filter-cat="weights">Weights</button>'
            '<button class="filter-btn" data-filter-cat="stability">Stability</button>'
            '<button class="filter-btn" data-filter-cat="output">Output</button>'
            '<button class="filter-btn" data-filter-cat="activations">Activations</button>'
            '<input class="search-box" id="param-search" type="text" '
            'placeholder="Search tensor name..." oninput="mcFilter()">'
            '<span class="toolbar-label" style="margin-left:.5rem">Sort:</span>'
            '<select class="sort-select" id="sort-select" onchange="mcSort()">'
            '<option value="severity">Severity</option>'
            '<option value="param">Tensor name</option>'
            '<option value="category">Category</option>'
            '</select>'
            '</div>'
        )
        interactive_js_html = (
            "\n<script>\n"
            "(function(){\n"
            "var activeSev='ALL',activeCat='ALL';\n"
            "function applyFilter(){\n"
            "  var search=(document.getElementById('param-search')||{}).value||'';\n"
            "  search=search.toLowerCase();\n"
            "  var rows=document.querySelectorAll('.finding-row');\n"
            "  rows.forEach(function(row){\n"
            "    var sev=row.getAttribute('data-severity')||'';\n"
            "    var cat=row.getAttribute('data-category')||'';\n"
            "    var param=row.getAttribute('data-param')||'';\n"
            "    var ok=(activeSev==='ALL'||sev===activeSev)&&\n"
            "           (activeCat==='ALL'||cat===activeCat)&&\n"
            "           (!search||param.indexOf(search)!==-1);\n"
            "    row.classList.toggle('hidden',!ok);\n"
            "  });\n"
            "}\n"
            "window.mcFilter=applyFilter;\n"
            "window.mcSort=function(){\n"
            "  var by=(document.getElementById('sort-select')||{}).value||'severity';\n"
            "  var tbody=document.getElementById('findings-tbody');\n"
            "  if(!tbody)return;\n"
            "  var rows=Array.from(tbody.querySelectorAll('.finding-row'));\n"
            "  var sevOrd={'ERROR':0,'WARN':1,'INFO':2};\n"
            "  rows.sort(function(a,b){\n"
            "    if(by==='severity'){\n"
            "      return (sevOrd[a.getAttribute('data-severity')]||3)-"
            "(sevOrd[b.getAttribute('data-severity')]||3);\n"
            "    }\n"
            "    var ka=a.getAttribute('data-'+by)||'';\n"
            "    var kb=b.getAttribute('data-'+by)||'';\n"
            "    return ka.localeCompare(kb);\n"
            "  });\n"
            "  rows.forEach(function(r){tbody.appendChild(r);});\n"
            "};\n"
            "document.addEventListener('DOMContentLoaded',function(){\n"
            "  document.querySelectorAll('[data-filter-sev]').forEach(function(btn){\n"
            "    btn.addEventListener('click',function(){\n"
            "      activeSev=btn.getAttribute('data-filter-sev');\n"
            "      document.querySelectorAll('[data-filter-sev]').forEach(function(b){\n"
            "        b.classList.remove('active');\n"
            "      });\n"
            "      btn.classList.add('active');\n"
            "      applyFilter();\n"
            "    });\n"
            "  });\n"
            "  document.querySelectorAll('[data-filter-cat]').forEach(function(btn){\n"
            "    btn.addEventListener('click',function(){\n"
            "      activeCat=btn.getAttribute('data-filter-cat');\n"
            "      document.querySelectorAll('[data-filter-cat]').forEach(function(b){\n"
            "        b.classList.remove('active');\n"
            "      });\n"
            "      btn.classList.add('active');\n"
            "      applyFilter();\n"
            "    });\n"
            "  });\n"
            "});\n"
            "})();\n"
            "</script>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>model-clinic report — {model_name}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#1a1a2e;color:#e0e0e0;padding:2rem;line-height:1.6}}
.container{{max-width:960px;margin:0 auto}}
h1{{font-size:1.6rem;color:#fff;margin-bottom:.25rem}}
.model-name{{font-size:1.2rem;color:#64b5f6;font-family:"Fira Code",Consolas,monospace;
  margin:.25rem 0 .5rem;word-break:break-all}}
h2{{font-size:1.1rem;color:#aaa;border-bottom:1px solid #333;padding-bottom:.5rem;
  margin:2rem 0 1rem}}
.header{{display:flex;justify-content:space-between;align-items:center;
  margin-bottom:2rem;flex-wrap:wrap;gap:1rem}}
.header-left{{flex:1}}
.meta-line{{color:#888;font-size:.85rem}}
.score-box{{text-align:center;background:#16213e;border-radius:12px;
  padding:1rem 1.5rem;min-width:140px}}
.score-grade{{font-size:1rem;color:#aaa;margin-top:.25rem}}
.cat-row{{display:flex;align-items:center;gap:.75rem;margin:.4rem 0}}
.cat-label{{width:120px;text-align:right;font-size:.85rem;color:#aaa}}
.cat-bar{{flex:1;height:18px;background:#2a2a4a;border-radius:9px;overflow:hidden}}
.cat-fill{{height:100%;border-radius:9px;transition:width .3s}}
.cat-score{{width:30px;font-size:.85rem;font-weight:600}}
table{{width:100%;border-collapse:collapse;margin:.5rem 0 1.5rem;font-size:.85rem}}
th{{text-align:left;padding:.5rem .75rem;background:#16213e;color:#aaa;
  font-weight:600;border-bottom:2px solid #333}}
td{{padding:.45rem .75rem;border-bottom:1px solid #2a2a4a;vertical-align:middle}}
tr:hover td{{background:#16213e}}
.mono{{font-family:"Fira Code",Consolas,monospace;font-size:.8rem;word-break:break-all}}
.badge{{display:inline-block;padding:.15rem .5rem;border-radius:4px;font-size:.75rem;
  font-weight:600;color:#000}}
.empty{{text-align:center;color:#666;padding:1.5rem;font-style:italic}}
.debug{{background:#0d1117;border:1px solid #333;border-radius:6px;padding:1rem;
  font-family:"Fira Code",Consolas,monospace;font-size:.75rem;color:#8b949e;
  overflow-x:auto;white-space:pre;max-height:600px;overflow-y:auto}}
.viz-container{{margin:.5rem 0 1.5rem;padding:.5rem 0}}
.footer{{margin-top:3rem;padding-top:1rem;border-top:1px solid #333;
  color:#555;font-size:.75rem;text-align:center}}
.suggest-box{{background:#16213e;border:1px solid #2a2a4a;border-radius:10px;
  padding:1.25rem 1.5rem;margin:1.5rem 0 2rem}}
.suggest-box.suggest-ok{{border-color:#2e7d32;background:#0a1f0a}}
.suggest-title{{font-size:.8rem;font-weight:700;letter-spacing:.08em;
  text-transform:uppercase;margin:.75rem 0 .4rem;padding-bottom:.25rem;
  border-bottom:1px solid #2a2a4a}}
.suggest-title:first-child{{margin-top:0}}
.suggest-critical{{color:#f44336}}
.suggest-warn{{color:#ff9800}}
.suggest-info{{color:#64b5f6}}
.suggest-item{{display:flex;align-items:center;gap:.6rem;padding:.3rem 0;
  font-size:.85rem;flex-wrap:wrap}}
.suggest-advisory{{opacity:.75}}
.suggest-badge{{display:inline-block;padding:.1rem .45rem;border-radius:3px;
  font-size:.7rem;font-weight:700;color:#000;white-space:nowrap;flex-shrink:0}}
.suggest-label{{font-weight:600;color:#e0e0e0;min-width:160px}}
.suggest-desc{{color:#888;font-size:.8rem;flex:1}}
.suggest-cli{{margin-top:1rem;padding-top:.75rem;border-top:1px solid #2a2a4a}}
.suggest-cmd-label{{font-size:.75rem;color:#666;margin-bottom:.2rem}}
.suggest-cmd{{display:block;background:#0d1117;border:1px solid #333;
  border-radius:4px;padding:.4rem .75rem;font-family:"Fira Code",Consolas,monospace;
  font-size:.78rem;color:#79c0ff;word-break:break-all;margin-bottom:.25rem}}
{interactive_css}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <div class="header-left">
    <h1>Model Clinic Report</h1>
    <div class="model-name">{model_name}</div>
    <div class="meta-line">{now}</div>
    <div class="meta-line">{_fmt(meta.num_params)} params &middot; \
{_fmt(meta.num_tensors)} tensors &middot; \
{meta.num_layers} layers &middot; hidden {_fmt(meta.hidden_size)}</div>
  </div>
  <div class="score-box">
    {gauge_svg}
    <div class="score-grade">Grade: {grade}</div>
  </div>
</div>

<h2>Health Score Breakdown</h2>
{cat_html}

<h2>Suggested Actions</h2>
{suggested_fixes_html}
{norm_section}
{dead_section}
{attn_section}
{neuron_hist_section}
{compare_section}
{mri_section}

<h2>Findings ({len(findings)})</h2>
{interactive_controls_html}
<table>
<thead><tr><th>Severity</th><th>Condition</th><th>Parameter</th><th>Details</th><th>References</th></tr></thead>
<tbody id="findings-tbody">
{findings_rows}
</tbody>
</table>

<h2>Prescriptions ({len(prescriptions)})</h2>
<table>
<tr><th>Treatment</th><th>Risk</th><th>Description</th><th>Target</th></tr>
{rx_rows}
</table>

<h2>Weight Distribution</h2>
<table>
<tr><th>Group</th><th>Tensors</th><th>Elements</th><th>Mean Norm</th>\
<th>Max Norm</th><th>Distribution</th></tr>
{ws_rows}
</table>

{debug_section}

<div class="footer">
  Generated by model-clinic v{__version__}
</div>

</div>
{interactive_js_html}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
