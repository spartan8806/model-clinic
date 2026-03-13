"""Compare: Health impact comparison between two checkpoints.

Unlike `model-diff` which shows parameter-by-parameter differences,
`model-clinic compare` shows the *health impact* of changes: score deltas,
resolved/new findings, and parameter-level change summary.

Usage:
    model-clinic compare before.pt after.pt
    model-clinic compare original.pt treated.pt --hf
    model-clinic compare a.pt b.pt --json
"""

import argparse
import json
import sys
from collections import Counter

import torch

from model_clinic._loader import load_state_dict, build_meta
from model_clinic._health_score import compute_health_score, _grade
from model_clinic.clinic import diagnose


# ── Display constants ────────────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"


def _delta_color(delta):
    """Color for a numeric delta."""
    if delta > 0:
        return _GREEN
    if delta < 0:
        return _RED
    return _DIM


def _delta_str(delta):
    """Format a delta with sign."""
    if delta > 0:
        return f"+{delta}"
    return str(delta)


# ── Core logic ───────────────────────────────────────────────────────────

def _finding_key(f):
    """Unique key for a finding (condition + param)."""
    return (f.condition, f.param_name)


def compare_models(sd_before, meta_before, sd_after, meta_after):
    """Compare two state dicts and return structured comparison result.

    Returns a dict with health scores, findings delta, and parameter changes.
    """
    # Diagnose both
    findings_before = diagnose(sd_before, meta_before)
    findings_after = diagnose(sd_after, meta_after)

    health_before = compute_health_score(findings_before)
    health_after = compute_health_score(findings_after)

    # Findings delta
    keys_before = set(_finding_key(f) for f in findings_before)
    keys_after = set(_finding_key(f) for f in findings_after)

    resolved_keys = keys_before - keys_after
    new_keys = keys_after - keys_before
    unchanged_keys = keys_before & keys_after

    resolved_by_condition = Counter(k[0] for k in resolved_keys)
    new_by_condition = Counter(k[0] for k in new_keys)
    unchanged_by_condition = Counter(k[0] for k in unchanged_keys)

    # Parameter changes
    all_keys = set(list(sd_before.keys()) + list(sd_after.keys()))
    modified = []
    only_before = []
    only_after = []

    for name in sorted(all_keys):
        if name not in sd_before:
            only_after.append(name)
            continue
        if name not in sd_after:
            only_before.append(name)
            continue

        a = sd_before[name]
        b = sd_after[name]

        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            continue

        if a.shape != b.shape:
            modified.append({"name": name, "delta_norm": float("inf"), "shape_changed": True})
            continue

        delta = (b.float() - a.float()).norm().item()
        if delta > 1e-10:
            modified.append({"name": name, "delta_norm": delta, "shape_changed": False})

    # Sort by delta_norm descending
    modified.sort(key=lambda x: x["delta_norm"], reverse=True)

    max_delta = modified[0]["delta_norm"] if modified else 0.0
    max_delta_name = modified[0]["name"] if modified else ""
    avg_delta = (
        sum(m["delta_norm"] for m in modified if not m["shape_changed"]) /
        max(len([m for m in modified if not m["shape_changed"]]), 1)
    )

    return {
        "health_before": {
            "overall": health_before.overall,
            "grade": health_before.grade,
            "categories": health_before.categories,
        },
        "health_after": {
            "overall": health_after.overall,
            "grade": health_after.grade,
            "categories": health_after.categories,
        },
        "health_delta": health_after.overall - health_before.overall,
        "findings_delta": {
            "resolved": dict(resolved_by_condition),
            "resolved_total": len(resolved_keys),
            "new": dict(new_by_condition),
            "new_total": len(new_keys),
            "unchanged": dict(unchanged_by_condition),
            "unchanged_total": len(unchanged_keys),
        },
        "param_changes": {
            "modified_count": len(modified),
            "only_before": only_before,
            "only_after": only_after,
            "max_delta_norm": max_delta,
            "max_delta_name": max_delta_name,
            "avg_delta_norm": avg_delta,
            "details": modified,
        },
    }


# ── Display ──────────────────────────────────────────────────────────────

def print_compare(result, path_before, path_after, file=None):
    """Print formatted comparison report."""
    out = file or sys.stdout

    print(f"\n{_BOLD}Comparing:{_RESET} {path_before} vs {path_after}\n", file=out)

    # Health score
    hb = result["health_before"]
    ha = result["health_after"]
    delta = result["health_delta"]
    dc = _delta_color(delta)

    print(f"{_BOLD}Health Score:{_RESET}", file=out)
    print(f"  before: {hb['overall']}/100 ({hb['grade']})  "
          f"->  after: {ha['overall']}/100 ({ha['grade']})  "
          f"{dc}{_BOLD}{_delta_str(delta):>4s}{_RESET}", file=out)
    print(file=out)

    # Category breakdown
    print(f"{_BOLD}Category Breakdown:{_RESET}", file=out)
    for cat in ["weights", "stability", "output", "activations"]:
        bv = hb["categories"].get(cat, 100)
        av = ha["categories"].get(cat, 100)
        cd = av - bv
        dc = _delta_color(cd)
        print(f"  {cat:<14s} {bv:>3d} -> {av:>3d}  ({dc}{_delta_str(cd):>4s}{_RESET})", file=out)
    print(file=out)

    # Findings delta
    fd = result["findings_delta"]
    print(f"{_BOLD}Findings Delta:{_RESET}", file=out)

    resolved_detail = ", ".join(f"{c}: {n}" for c, n in sorted(fd["resolved"].items()))
    new_detail = ", ".join(f"{c}: {n}" for c, n in sorted(fd["new"].items()))
    unchanged_detail = ", ".join(f"{c}: {n}" for c, n in sorted(fd["unchanged"].items()))

    print(f"  Resolved:  {fd['resolved_total']}"
          + (f" ({resolved_detail})" if resolved_detail else ""), file=out)
    print(f"  New:       {fd['new_total']}"
          + (f" ({new_detail})" if new_detail else ""), file=out)
    print(f"  Unchanged: {fd['unchanged_total']}"
          + (f" ({unchanged_detail})" if unchanged_detail else ""), file=out)
    print(file=out)

    # Parameter changes
    pc = result["param_changes"]
    print(f"{_BOLD}Parameter Changes:{_RESET}", file=out)
    print(f"  Modified: {pc['modified_count']} tensors", file=out)
    if pc["modified_count"] > 0:
        print(f"  Max delta norm: {pc['max_delta_norm']:.4f} ({pc['max_delta_name']})", file=out)
        print(f"  Avg delta norm: {pc['avg_delta_norm']:.4f}", file=out)

    if pc["only_before"]:
        print(f"  Removed: {len(pc['only_before'])} tensors", file=out)
    if pc["only_after"]:
        print(f"  Added:   {len(pc['only_after'])} tensors", file=out)

    print(file=out)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic compare",
        description="Compare health impact between two checkpoints",
    )
    parser.add_argument("model_before", help="Path to before checkpoint (or HF model)")
    parser.add_argument("model_after", help="Path to after checkpoint (or HF model)")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace models")
    parser.add_argument("--json", action="store_true", help="Output as JSON only")
    parser.add_argument("--export", type=str, default=None, help="Export JSON to file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not args.quiet and not args.json:
        print(f"Loading before: {args.model_before}")
    sd_before, meta_before = load_state_dict(args.model_before, hf=args.hf)

    if not args.quiet and not args.json:
        print(f"Loading after:  {args.model_after}")
    sd_after, meta_after = load_state_dict(args.model_after, hf=args.hf)

    result = compare_models(sd_before, meta_before, sd_after, meta_after)

    if args.json:
        # Strip details for cleaner JSON (they can be large)
        output = dict(result)
        output["param_changes"] = {k: v for k, v in result["param_changes"].items()
                                    if k != "details"}
        print(json.dumps(output, indent=2, default=str))
    else:
        print_compare(result, args.model_before, args.model_after)

    if args.export:
        with open(args.export, "w") as f:
            json.dump(result, f, indent=2, default=str)
        if not args.quiet:
            print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()
