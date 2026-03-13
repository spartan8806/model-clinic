"""Health score computation — a 0-100 score from diagnostic findings."""

import sys

from model_clinic._types import Finding, HealthScore


# ── Category mappings ────────────────────────────────────────────────────

CATEGORY_MAP = {
    "weights": {
        "dead_neurons", "vanishing_norm", "exploding_norm",
        "heavy_tails", "saturated_weights", "identical_rows",
        "weight_corruption", "head_redundancy",
        "moe_router_collapse", "lora_merge_artifacts",
        "quantization_degradation", "model_aging",
        "causal_outlier", "layer_isolation",
    },
    "stability": {
        "nan_inf", "norm_drift", "stuck_gate_closed", "stuck_gate_open",
        "positional_encoding_issues",
        "gradient_noise", "representation_drift",
    },
    "output": {
        "generation_collapse", "low_coherence", "low_entropy",
        "response_uniformity", "token_collapse",
    },
    "activations": {
        "activation_nan", "activation_inf", "activation_explosion",
        "activation_collapse", "residual_explosion", "residual_collapse",
    },
}

RUNTIME_CATEGORIES = {"output", "activations"}

SEVERITY_PENALTY = {
    "ERROR": 25,
    "WARN": 10,
    "INFO": 2,
}

BASE_WEIGHTS = {
    "weights": 0.40,
    "stability": 0.30,
    "output": 0.20,
    "activations": 0.10,
}


# ── Core logic ───────────────────────────────────────────────────────────

def _categorize(condition: str) -> str:
    for cat, conditions in CATEGORY_MAP.items():
        if condition in conditions:
            return cat
    return "weights"  # uncategorized conditions default to weights


def _grade(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 65:
        return "C"
    if score >= 50:
        return "D"
    return "F"


def _summary(grade: str, categories: dict) -> str:
    if grade == "A":
        return "Model is healthy — no significant issues detected."
    worst = min(categories, key=lambda k: categories[k])
    return f"Grade {grade} — worst category: {worst} ({categories[worst]}/100)."


def compute_health_score(findings: list) -> HealthScore:
    cat_scores = {cat: 100 for cat in CATEGORY_MAP}
    cat_has_findings = {cat: False for cat in CATEGORY_MAP}

    # Per-condition penalty caps: no single condition can tank a whole category.
    # This prevents 20 norm_drift WARNs from zeroing out stability.
    CONDITION_CAPS = {
        "ERROR": 35,   # One condition type can deduct at most 35pts (category)
        "WARN": 15,    # One condition type can deduct at most 15pts
        "INFO": 5,     # One condition type can deduct at most 5pts
    }

    # Track total penalty applied per (category, condition) pair
    applied = {}

    for f in findings:
        cat = _categorize(f.condition)
        base_penalty = SEVERITY_PENALTY.get(f.severity, 0)
        cap = CONDITION_CAPS.get(f.severity, 0)

        key = (cat, f.condition)
        already_applied = applied.get(key, 0.0)
        remaining_cap = max(0.0, cap - already_applied)
        penalty = min(base_penalty, remaining_cap)

        applied[key] = already_applied + penalty
        cat_scores[cat] = max(0, cat_scores[cat] - penalty)
        cat_has_findings[cat] = True

    # Check if any runtime findings exist
    has_runtime = any(cat_has_findings[c] for c in RUNTIME_CATEGORIES)

    if has_runtime:
        weights = dict(BASE_WEIGHTS)
    else:
        # Redistribute runtime weight proportionally to static categories
        static_total = sum(BASE_WEIGHTS[c] for c in BASE_WEIGHTS if c not in RUNTIME_CATEGORIES)
        weights = {}
        for cat in BASE_WEIGHTS:
            if cat in RUNTIME_CATEGORIES:
                weights[cat] = 0.0
            else:
                weights[cat] = BASE_WEIGHTS[cat] / static_total

    overall = round(sum(cat_scores[cat] * weights[cat] for cat in cat_scores))
    g = _grade(overall)

    return HealthScore(
        overall=overall,
        categories=dict(cat_scores),
        grade=g,
        summary=_summary(g, cat_scores),
    )


# ── Display ──────────────────────────────────────────────────────────────

_GRADE_COLORS = {
    "A": "\033[92m",  # green
    "B": "\033[96m",  # cyan
    "C": "\033[93m",  # yellow
    "D": "\033[33m",  # orange
    "F": "\033[91m",  # red
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"


def _bar(score: int, width: int = 20) -> str:
    filled = round(score / 100 * width)
    empty = width - filled
    if score >= 80:
        color = "\033[92m"
    elif score >= 50:
        color = "\033[93m"
    else:
        color = "\033[91m"
    return f"{color}{'█' * filled}{_DIM}{'░' * empty}{_RESET}"


def print_health_score(score: HealthScore, file=None):
    out = file or sys.stdout
    color = _GRADE_COLORS.get(score.grade, "")

    print(f"\n{_BOLD}Model Health Score{_RESET}", file=out)
    print(f"{'─' * 45}", file=out)
    print(f"  Overall: {color}{_BOLD}{score.overall}/100  {score.grade}{_RESET}", file=out)
    print(file=out)

    order = ["weights", "stability", "output", "activations"]
    for cat in order:
        val = score.categories.get(cat, 100)
        label = f"  {cat:<14s}"
        print(f"{label} {_bar(int(val))} {int(val):>3d}/100", file=out)

    print(file=out)
    print(f"  {_DIM}{score.summary}{_RESET}", file=out)
    print(file=out)
