"""Autopsy — deep forensic analysis for dead or severely broken models.

Usage:
    model-clinic autopsy dead_model.pt
    model-clinic autopsy dead_model.pt --json
    model-clinic autopsy dead_model.pt --safetensors
"""

import argparse
import json
import sys
from collections import defaultdict

import torch


# Category labels for the damage assessment section
_CAT_LABEL = {
    "weights": "Weights",
    "stability": "Stability",
    "output": "Output",
    "activations": "Activations",
}

# Which conditions map to which category (mirrors _health_score.py)
_CATEGORY_MAP = {
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

# Conditions that have an automatic fix available (action is not "advisory")
_AUTO_FIXABLE = {
    "dead_neurons",
    "stuck_gate_closed",
    "stuck_gate_open",
    "exploding_norm",
    "vanishing_norm",
    "heavy_tails",
    "norm_drift",
    "saturated_weights",
    "nan_inf",
    "identical_rows",
}


def _categorize(condition):
    for cat, conds in _CATEGORY_MAP.items():
        if condition in conds:
            return cat
    return "weights"


def _infer_cause_of_death(findings):
    """Pick the primary and secondary cause of death from findings.

    Primary = highest-severity finding group, broken ties by count.
    Secondary = next distinct condition group.
    Returns (primary_condition, primary_count, secondary_condition, secondary_count).
    """
    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    by_condition = defaultdict(list)
    for f in findings:
        by_condition[f.condition].append(f)

    # Sort by: worst severity first, then count descending
    ranked = sorted(
        by_condition.items(),
        key=lambda kv: (
            severity_order.get(min(f.severity for f in kv[1]), 2),
            -len(kv[1]),
        ),
    )

    primary = ranked[0] if len(ranked) >= 1 else None
    secondary = ranked[1] if len(ranked) >= 2 else None
    return primary, secondary


def _salvageable_assessment(findings, prescriptions):
    """Return (auto_fixable_count, total_findings, manual_list)."""
    total = len(findings)
    if total == 0:
        return 0, 0, []

    auto_fixable = sum(
        1 for f in findings if f.condition in _AUTO_FIXABLE
    )
    manual = [
        f for f in findings
        if f.condition not in _AUTO_FIXABLE and f.severity in ("ERROR", "WARN")
    ]
    return auto_fixable, total, manual


def _forensics_tensors(findings, state_dict, top_n=5):
    """Return detailed tensor-level breakdown for ERROR findings.

    Returns list of dicts with: name, severity, condition, shape, detail.
    """
    # Score each unique tensor by severity and condition priority
    priority = {"nan_inf": 100, "exploding_norm": 80, "weight_corruption": 70,
                "dead_neurons": 50, "vanishing_norm": 40}
    sev_score = {"ERROR": 30, "WARN": 10, "INFO": 1}

    tensor_info = {}
    for f in findings:
        if f.severity not in ("ERROR", "WARN"):
            continue
        name = f.param_name
        score = sev_score.get(f.severity, 1) + priority.get(f.condition, 5)
        if name not in tensor_info or score > tensor_info[name]["score"]:
            t = state_dict.get(name)
            shape = list(t.shape) if isinstance(t, torch.Tensor) else None
            detail = _format_forensic_detail(f, t)
            tensor_info[name] = {
                "name": name,
                "severity": f.severity,
                "condition": f.condition,
                "shape": shape,
                "detail": detail,
                "score": score,
            }

    ranked = sorted(tensor_info.values(), key=lambda x: -x["score"])
    return ranked[:top_n]


def _format_forensic_detail(f, tensor):
    """Format a one-line forensic detail string for a finding."""
    d = f.details
    c = f.condition
    if c == "nan_inf":
        positions = ""
        if isinstance(tensor, torch.Tensor):
            nan_mask = torch.isnan(tensor)
            if nan_mask.any():
                idx = nan_mask.nonzero(as_tuple=False)[:3]
                positions = " at positions " + ", ".join(str(tuple(i.tolist())) for i in idx)
        return f"{d.get('nan_count', 0)} NaN / {d.get('inf_count', 0)} Inf values{positions}"
    if c == "exploding_norm":
        return f"per-elem norm: {d.get('per_elem_norm', '?'):.3f} (healthy: <10.0)"
    if c == "dead_neurons":
        return f"{d.get('dead_count', '?')}/{d.get('total', '?')} dead {d.get('dim', '')} ({d.get('pct', 0):.1%})"
    if c == "weight_corruption":
        return f"reason: {d.get('reason', '?')}"
    if c == "vanishing_norm":
        return f"per-elem norm: {d.get('per_elem_norm', '?'):.2e}"
    if c == "norm_drift":
        return f"norm weight mean: {d.get('mean', '?'):.3f} (expected ~1.0)"
    return str(d)[:80]


def autopsy(state_dict, score_threshold=50):
    """Run autopsy analysis on a state dict.

    Args:
        state_dict: model state dict (dict of str -> torch.Tensor)
        score_threshold: models scoring >= this are redirected to exam. Default 50.

    Returns:
        dict with keys:
            score (int), grade (str), redirect (bool),
            findings (list[Finding]), prescriptions (list[Prescription]),
            primary_cause (dict|None), secondary_cause (dict|None),
            damage (dict), salvageable (str), recovery_plan (list[str]),
            forensics (list[dict]), auto_fixable (int), total_findings (int)
    """
    from model_clinic.clinic import diagnose, prescribe
    from model_clinic._health_score import compute_health_score

    findings = diagnose(state_dict)
    health = compute_health_score(findings)
    prescriptions = prescribe(findings)

    # Redirect healthy models
    if health.overall >= score_threshold:
        return {
            "score": health.overall,
            "grade": health.grade,
            "redirect": True,
            "findings": findings,
            "prescriptions": prescriptions,
            "health": health,
        }

    primary, secondary = _infer_cause_of_death(findings)
    auto_fixable, total, manual_intervention = _salvageable_assessment(findings, prescriptions)

    # Damage assessment per category
    damage = {}
    for cat in ["weights", "stability", "output"]:
        cat_findings = [f for f in findings if _categorize(f.condition) == cat]
        errors = sum(1 for f in cat_findings if f.severity == "ERROR")
        warns = sum(1 for f in cat_findings if f.severity == "WARN")
        count = len(cat_findings)
        if errors > 0:
            status = "FAILED"
        elif warns > 0:
            status = "DEGRADED"
        else:
            status = "OK"
        damage[cat] = {"status": status, "count": count, "errors": errors, "warns": warns}

    # Salvageability verdict
    if total == 0:
        salvageable = "N/A"
    elif auto_fixable == total:
        salvageable = "Yes — all findings are auto-fixable"
    elif auto_fixable > 0:
        salvageable = f"Partially — {auto_fixable}/{total} fixable automatically"
    else:
        salvageable = "No — all findings require manual intervention"

    # Recovery plan
    recovery_plan = []
    if auto_fixable > 0:
        conservative_fixable = sum(
            1 for f in findings
            if f.condition in {"dead_neurons", "norm_drift", "vanishing_norm", "identical_rows"}
        )
        if conservative_fixable > 0:
            recovery_plan.append(
                "model-clinic treat model.pt --conservative  "
                f"(fixes: norm_drift, dead_neurons, vanishing_norm)"
            )
        non_conservative = sum(
            1 for f in findings
            if f.condition in {"exploding_norm", "saturated_weights", "heavy_tails",
                               "nan_inf", "stuck_gate_closed", "stuck_gate_open"}
        )
        if non_conservative > 0:
            recovery_plan.append(
                "model-clinic treat model.pt  (applies all available fixes)"
            )

    for f in manual_intervention[:3]:
        recovery_plan.append(
            f"Inspect {f.param_name} manually ({f.condition}) — "
            "re-initialize or remove this tensor"
        )

    recovery_plan.append(
        "model-clinic exam model.pt  (re-evaluate after treatment)"
    )

    forensics = _forensics_tensors(findings, state_dict, top_n=5)

    return {
        "score": health.overall,
        "grade": health.grade,
        "redirect": False,
        "findings": findings,
        "prescriptions": prescriptions,
        "health": health,
        "primary_cause": (
            {"condition": primary[0], "count": len(primary[1]),
             "severity": min(f.severity for f in primary[1]),
             "detail": _format_forensic_detail(primary[1][0], None)}
            if primary else None
        ),
        "secondary_cause": (
            {"condition": secondary[0], "count": len(secondary[1]),
             "severity": min(f.severity for f in secondary[1]),
             "detail": _format_forensic_detail(secondary[1][0], None)}
            if secondary else None
        ),
        "damage": damage,
        "salvageable": salvageable,
        "recovery_plan": recovery_plan,
        "forensics": forensics,
        "auto_fixable": auto_fixable,
        "total_findings": total,
    }


def _print_autopsy(result, model_path):
    """Print a formatted autopsy report."""
    SEP = "=" * 60
    errors = sum(1 for f in result["findings"] if f.severity == "ERROR")
    warns = sum(1 for f in result["findings"] if f.severity == "WARN")

    print(f"\nmodel-clinic AUTOPSY REPORT")
    print(SEP)
    print(f"Model: {model_path}")
    status_str = f"CRITICAL -- {errors} ERROR finding(s), {warns} WARNING(s)"
    print(f"Status: {status_str}")
    print(f"Score:  {result['score']}/100  {result['grade']}")

    # Cause of death
    print()
    print("CAUSE OF DEATH (most likely):")
    if result["primary_cause"]:
        p = result["primary_cause"]
        print(f"  Primary:   {p['condition']} -- {p['count']} instance(s)")
    else:
        print("  Primary:   (no dominant cause identified)")
    if result["secondary_cause"]:
        s = result["secondary_cause"]
        print(f"  Secondary: {s['condition']} -- {s['count']} instance(s)")

    # Damage assessment
    print()
    print("DAMAGE ASSESSMENT:")
    cat_order = ["weights", "stability", "output"]
    for cat in cat_order:
        if cat not in result["damage"]:
            continue
        d = result["damage"][cat]
        status = d["status"]
        count = d["count"]
        label = _CAT_LABEL.get(cat, cat)
        print(f"  {label:<12s} [{status}]  {count} finding(s)")

    # Salvageable
    print()
    print(f"SALVAGEABLE? {result['salvageable']}")
    if result["auto_fixable"] > 0 or result["total_findings"] > 0:
        auto = result["auto_fixable"]
        total = result["total_findings"]
        print(f"  Conservative treatment: {auto}/{total} fixable automatically")

    # Recovery plan
    if result["recovery_plan"]:
        print()
        print("RECOVERY PLAN:")
        for i, step in enumerate(result["recovery_plan"], 1):
            print(f"  {i}. {step}")

    # Forensics
    if result["forensics"]:
        print()
        print(f"FORENSICS -- top {len(result['forensics'])} most damaged tensors:")
        for item in result["forensics"]:
            shape_str = f"[{'x'.join(str(d) for d in item['shape'])}]" if item["shape"] else "[scalar]"
            line = f"  {item['name']:<40s} {shape_str:<12s}  {item['detail']}"
            print(line)

    print()


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic autopsy",
        description="Deep forensic analysis for dead or severely broken models",
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--safetensors", action="store_true",
                        help="Force safetensors loading")
    parser.add_argument("--threshold", type=int, default=50,
                        help="Health score below this triggers autopsy (default: 50)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--export", type=str, default=None,
                        help="Export JSON report to this path")
    args = parser.parse_args()

    from model_clinic._loader import load_state_dict, build_meta

    print(f"Loading: {args.model}")
    state_dict, meta_dict = load_state_dict(args.model, hf=args.hf)
    print(f"Loaded {len(state_dict)} tensors")

    result = autopsy(state_dict, score_threshold=args.threshold)

    if result["redirect"]:
        print(
            f"\nModel scores {result['score']}/100 ({result['grade']}) -- "
            f"above autopsy threshold ({args.threshold})."
        )
        print("Use 'model-clinic exam' for standard diagnosis.")
        return

    if args.json:
        out = {
            "model": args.model,
            "score": result["score"],
            "grade": result["grade"],
            "total_findings": result["total_findings"],
            "auto_fixable": result["auto_fixable"],
            "salvageable": result["salvageable"],
            "primary_cause": result["primary_cause"],
            "secondary_cause": result["secondary_cause"],
            "damage": result["damage"],
            "recovery_plan": result["recovery_plan"],
            "forensics": [
                {k: v for k, v in item.items() if k != "score"}
                for item in result["forensics"]
            ],
            "findings": [
                {"condition": f.condition, "severity": f.severity,
                 "param": f.param_name, "details": f.details}
                for f in result["findings"]
            ],
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        _print_autopsy(result, args.model)

    if args.export:
        out = {
            "model": args.model,
            "score": result["score"],
            "grade": result["grade"],
            "total_findings": result["total_findings"],
            "auto_fixable": result["auto_fixable"],
            "salvageable": result["salvageable"],
            "primary_cause": result["primary_cause"],
            "secondary_cause": result["secondary_cause"],
            "damage": result["damage"],
            "recovery_plan": result["recovery_plan"],
            "forensics": [
                {k: v for k, v in item.items() if k != "score"}
                for item in result["forensics"]
            ],
        }
        with open(args.export, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()
