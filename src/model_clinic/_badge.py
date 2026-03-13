"""HuggingFace Hub health badge generation for model-clinic."""

from urllib.parse import quote

from model_clinic._types import HealthScore

__version__ = "0.3.0"

# ── Color mapping ────────────────────────────────────────────────────────────

GRADE_COLORS = {
    "A": "#4c1",       # bright green
    "B": "#44cc11",    # green
    "C": "#dfb317",    # yellow
    "D": "#fe7d37",    # orange
    "F": "#e05d44",    # red
}

# shields.io color name for URL badges
GRADE_COLOR_NAMES = {
    "A": "brightgreen",
    "B": "green",
    "C": "yellow",
    "D": "orange",
    "F": "red",
}


# ── SVG generation ───────────────────────────────────────────────────────────

def _text_width(text: str) -> int:
    """Estimate pixel width of text at ~7px/char (Verdana 11px approx)."""
    return len(text) * 7


def generate_badge_svg(health: HealthScore) -> str:
    """Generate a shields.io-compatible flat-style SVG badge.

    Returns an inline SVG string. The badge reads:
        [ model-clinic | 76/100 C ]
    Color on the right side reflects grade.
    """
    label = "model-clinic"
    message = f"{health.overall}/100 {health.grade}"
    color = GRADE_COLORS.get(health.grade, "#e05d44")

    label_w = _text_width(label) + 10   # padding
    message_w = _text_width(message) + 10
    total_w = label_w + message_w

    # Text x-centers for each half
    label_x = label_w // 2 + 1
    message_x = label_w + message_w // 2 - 1

    svg = f"""\
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{total_w}" height="20">
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="{total_w}" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="{label_w}" height="20" fill="#555"/>
    <rect x="{label_w}" width="{message_w}" height="20" fill="{color}"/>
    <rect width="{total_w}" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110">
    <text x="{label_x * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(label_w - 10) * 10}" lengthAdjust="spacing">{label}</text>
    <text x="{label_x * 10}" y="140" transform="scale(.1)" textLength="{(label_w - 10) * 10}" lengthAdjust="spacing">{label}</text>
    <text x="{message_x * 10}" y="150" fill="#010101" fill-opacity=".3" transform="scale(.1)" textLength="{(message_w - 10) * 10}" lengthAdjust="spacing">{message}</text>
    <text x="{message_x * 10}" y="140" transform="scale(.1)" textLength="{(message_w - 10) * 10}" lengthAdjust="spacing">{message}</text>
  </g>
</svg>"""
    return svg


# ── URL generation ────────────────────────────────────────────────────────────

def generate_badge_url(health: HealthScore) -> str:
    """Generate a shields.io URL for a dynamic badge.

    Returns a URL of the form:
        https://img.shields.io/badge/model--clinic-76%2F100%20C-green
    """
    color_name = GRADE_COLOR_NAMES.get(health.grade, "red")
    label = "model--clinic"  # shields.io uses double-dash for literal hyphen
    message = f"{health.overall}/100 {health.grade}"
    encoded_message = quote(message, safe="")
    return f"https://img.shields.io/badge/{label}-{encoded_message}-{color_name}"


# ── Model card markdown ───────────────────────────────────────────────────────

def generate_model_card_snippet(
    health: HealthScore,
    findings: list,
    model_name: str = "",
) -> str:
    """Generate markdown suitable for pasting into a HuggingFace model card.

    Includes:
    - Shields.io badge
    - Category score table
    - Top-5 findings summary (by severity)
    - Link to model-clinic on PyPI
    """
    badge_url = generate_badge_url(health)
    alt_text = f"model-clinic score: {health.overall}/100 {health.grade}"

    lines = []
    lines.append("## Model Health")
    lines.append("")

    if model_name:
        lines.append(f"**Model:** {model_name}")
        lines.append("")

    lines.append(f"![model-clinic]({badge_url})")
    lines.append("")
    lines.append(f"**Overall:** {health.overall}/100 &nbsp; **Grade:** {health.grade}")
    lines.append("")
    lines.append(f"*{health.summary}*")
    lines.append("")

    # Category table
    lines.append("| Category | Score |")
    lines.append("|----------|-------|")
    for cat in ["weights", "stability", "output", "activations"]:
        val = health.categories.get(cat, 100)
        lines.append(f"| {cat} | {int(val)}/100 |")
    lines.append("")

    # Top findings
    if findings:
        severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
        sorted_findings = sorted(
            findings,
            key=lambda f: (severity_order.get(f.severity, 3), f.condition),
        )
        top = sorted_findings[:5]

        lines.append("**Top findings:**")
        lines.append("")
        for f in top:
            lines.append(f"- `{f.severity}` {f.condition} — `{f.param_name}`")
        lines.append("")

    lines.append(
        f"*Analyzed with [model-clinic](https://pypi.org/project/model-clinic/) v{__version__}*"
    )
    lines.append("")

    return "\n".join(lines)


# ── File I/O ─────────────────────────────────────────────────────────────────

def save_badge_svg(health: HealthScore, path: str) -> None:
    """Save badge SVG to a file."""
    svg = generate_badge_svg(health)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(svg)
