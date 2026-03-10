"""Diff: Compare two checkpoints parameter-by-parameter.

Usage:
    model-diff phase1.pt phase2.pt
    model-diff a.pt b.pt --filter "wrapper"
    model-diff a.pt b.pt --top 20 --sort drift
    model-diff a.pt b.pt --export diff_report.json
"""

import argparse
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from model_clinic._loader import load_state_dict


def compute_diff(name, a, b):
    """Compare two tensors."""
    if a.shape != b.shape:
        return {
            "name": name,
            "status": "shape_changed",
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }

    a_f = a.float().flatten()
    b_f = b.float().flatten()
    delta = b_f - a_f

    result = {
        "name": name,
        "status": "changed",
        "shape": list(a.shape),
        "numel": a.numel(),
        "l2_dist": delta.norm().item(),
        "l1_dist": delta.abs().mean().item(),
        "max_change": delta.abs().max().item(),
        "relative_change": (delta.norm() / max(a_f.norm(), 1e-8)).item(),
        "cosine_sim": F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
            if a.numel() > 1 else 1.0,
        "mean_a": a_f.mean().item(),
        "mean_b": b_f.mean().item(),
        "std_a": a_f.std().item() if a.numel() > 1 else 0.0,
        "std_b": b_f.std().item() if a.numel() > 1 else 0.0,
        "norm_a": a_f.norm().item(),
        "norm_b": b_f.norm().item(),
    }

    if a.dim() == 0:
        result["val_a"] = a.item()
        result["val_b"] = b.item()
        result["val_delta"] = b.item() - a.item()
        if "gate" in name.lower():
            result["sig_a"] = torch.sigmoid(a.float()).item()
            result["sig_b"] = torch.sigmoid(b.float()).item()

    result["is_frozen"] = result["l2_dist"] < 1e-10

    return result


def main():
    parser = argparse.ArgumentParser(description="Compare two model checkpoints")
    parser.add_argument("model_a", help="Path to first checkpoint")
    parser.add_argument("model_b", help="Path to second checkpoint")
    parser.add_argument("--filter", type=str, default=None, help="Filter params by name")
    parser.add_argument("--sort", choices=["name", "drift", "l2", "cosine", "relative"],
                        default="drift", help="Sort order (drift = relative_change)")
    parser.add_argument("--top", type=int, default=0, help="Show only top N")
    parser.add_argument("--frozen", action="store_true", help="Show frozen params too")
    parser.add_argument("--export", type=str, default=None, help="Export to JSON")
    args = parser.parse_args()

    print(f"A: {args.model_a}")
    print(f"B: {args.model_b}")

    params_a, meta_a = load_state_dict(args.model_a)
    params_b, meta_b = load_state_dict(args.model_b)

    if meta_a:
        print(f"  A: phase={meta_a.get('phase', '?')}, step={meta_a.get('step', '?')}")
    if meta_b:
        print(f"  B: phase={meta_b.get('phase', '?')}, step={meta_b.get('step', '?')}")

    all_keys = set(list(params_a.keys()) + list(params_b.keys()))

    diffs = []
    only_a = []
    only_b = []

    for name in sorted(all_keys):
        if args.filter and args.filter not in name:
            continue

        if name not in params_a:
            only_b.append(name)
            continue
        if name not in params_b:
            only_a.append(name)
            continue

        d = compute_diff(name, params_a[name], params_b[name])
        if not args.frozen and d.get("is_frozen"):
            continue
        diffs.append(d)

    sort_key = {
        "name": lambda d: d["name"],
        "drift": lambda d: d.get("relative_change", 0),
        "l2": lambda d: d.get("l2_dist", 0),
        "cosine": lambda d: 1 - d.get("cosine_sim", 1),
        "relative": lambda d: d.get("relative_change", 0),
    }[args.sort]

    diffs.sort(key=sort_key, reverse=(args.sort != "name"))

    if args.top:
        diffs = diffs[:args.top]

    changed = [d for d in diffs if not d.get("is_frozen")]
    frozen = len(all_keys) - len(changed) - len(only_a) - len(only_b)

    print(f"\n{'='*80}")
    print(f"Total params: {len(all_keys):,}")
    print(f"Changed: {len(changed):,}  |  Frozen: {frozen:,}  |  Only A: {len(only_a)}  |  Only B: {len(only_b)}")

    if only_a:
        print(f"\nOnly in A:")
        for name in only_a[:10]:
            print(f"  - {name}")
    if only_b:
        print(f"\nOnly in B:")
        for name in only_b[:10]:
            print(f"  + {name}")

    gate_diffs = [d for d in diffs if "gate" in d["name"].lower() and "val_a" in d]
    if gate_diffs:
        print(f"\nGate changes:")
        for g in gate_diffs:
            sig_a = g.get("sig_a", "?")
            sig_b = g.get("sig_b", "?")
            print(f"  {g['name']}: {g['val_a']:+.4f} -> {g['val_b']:+.4f}"
                  f"  (sig: {sig_a:.6f} -> {sig_b:.6f})")

    print(f"\n{'='*80}")
    print(f"{'Parameter':<55s} {'L2 Dist':>10s} {'Rel Chg':>10s} {'Cosine':>8s} {'Max Chg':>10s}")
    print(f"{'-'*55} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    for d in diffs:
        if d["status"] == "shape_changed":
            print(f"  {d['name']:<55s} SHAPE CHANGED: {d['shape_a']} -> {d['shape_b']}")
            continue

        print(
            f"  {d['name']:<55s}"
            f" {d['l2_dist']:>10.4f}"
            f" {d['relative_change']:>9.2%}"
            f" {d['cosine_sim']:>8.4f}"
            f" {d['max_change']:>10.6f}"
        )

    module_drift = defaultdict(lambda: {"l2_total": 0, "count": 0, "params": 0})
    for d in diffs:
        if d["status"] == "shape_changed":
            continue
        parts = d["name"].split("/")
        module = parts[0] if parts else "root"
        module_drift[module]["l2_total"] += d["l2_dist"] ** 2
        module_drift[module]["count"] += 1
        module_drift[module]["params"] += d["numel"]

    if module_drift:
        print(f"\nModule-level drift:")
        for mod, info in sorted(module_drift.items(), key=lambda x: x[1]["l2_total"], reverse=True):
            total_l2 = info["l2_total"] ** 0.5
            print(f"  {mod:<40s} {info['count']:3d} tensors  {info['params']:>10,} params  L2={total_l2:.4f}")

    if args.export:
        export = {
            "model_a": args.model_a,
            "model_b": args.model_b,
            "meta_a": meta_a,
            "meta_b": meta_b,
            "summary": {
                "total": len(all_keys),
                "changed": len(changed),
                "frozen": frozen,
                "only_a": only_a,
                "only_b": only_b,
            },
            "diffs": diffs,
        }
        with open(args.export, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
