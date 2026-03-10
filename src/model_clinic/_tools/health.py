"""Health: Quick model health check.

Usage:
    model-health checkpoint.pt
    model-health checkpoint.pt --verbose
    model-health Qwen/Qwen2.5-0.5B-Instruct --hf
"""

import argparse
import json
from collections import defaultdict

import torch

from model_clinic._loader import load_state_dict


class HealthReport:
    def __init__(self):
        self.issues = []
        self.stats = {}

    def warn(self, name, msg):
        self.issues.append(("WARN", name, msg))

    def error(self, name, msg):
        self.issues.append(("ERROR", name, msg))

    def info(self, name, msg):
        self.issues.append(("INFO", name, msg))


def check_dead_neurons(name, tensor, report):
    """Check for dead rows/columns in weight matrices."""
    if tensor.dim() < 2:
        return

    t = tensor.float()

    row_norms = t.norm(dim=1)
    dead_rows = (row_norms < 1e-7).sum().item()
    if dead_rows > 0:
        pct = dead_rows / t.shape[0]
        if pct > 0.1:
            report.error(name, f"{dead_rows}/{t.shape[0]} dead output neurons ({pct:.1%})")
        elif pct > 0.01:
            report.warn(name, f"{dead_rows}/{t.shape[0]} dead output neurons ({pct:.1%})")

    col_norms = t.norm(dim=0)
    dead_cols = (col_norms < 1e-7).sum().item()
    if dead_cols > 0:
        pct = dead_cols / t.shape[1]
        if pct > 0.1:
            report.error(name, f"{dead_cols}/{t.shape[1]} dead input neurons ({pct:.1%})")
        elif pct > 0.01:
            report.warn(name, f"{dead_cols}/{t.shape[1]} dead input neurons ({pct:.1%})")


def check_saturation(name, tensor, report):
    """Check for saturated weights."""
    if tensor.dim() < 2:
        return

    t = tensor.float()
    near_boundary = ((t.abs() > 0.99 * t.abs().max()) & (t.abs().max() > 0.1)).float().mean().item()
    if near_boundary > 0.5:
        report.warn(name, f"Saturated: {near_boundary:.1%} of weights near boundary")


def check_norm_health(name, tensor, report, layer_norms):
    """Check for exploding/vanishing norms."""
    t = tensor.float()
    norm = t.norm().item()
    numel = t.numel()

    per_elem_norm = norm / (numel ** 0.5) if numel > 0 else 0

    parts = name.split(".")
    layer_type = parts[-1] if parts else name
    layer_norms[layer_type].append((name, per_elem_norm))

    if per_elem_norm > 10.0:
        report.warn(name, f"High per-element norm: {per_elem_norm:.4f}")
    elif per_elem_norm < 1e-6 and numel > 10:
        report.warn(name, f"Near-zero norm: {per_elem_norm:.4e}")


def check_distribution(name, tensor, report):
    """Check for abnormal distributions."""
    if tensor.numel() < 100:
        return

    t = tensor.float().flatten()
    mean = t.mean().item()
    std = t.std().item()

    if std < 1e-8:
        report.warn(name, f"Zero variance (std={std:.2e})")
        return

    z = (t - mean) / std
    kurtosis = z.pow(4).mean().item()
    if kurtosis > 20:
        report.warn(name, f"Heavy tails: kurtosis={kurtosis:.1f} (normal=3.0)")

    skew = z.pow(3).mean().item()
    if abs(skew) > 3:
        report.warn(name, f"Skewed distribution: skew={skew:+.2f}")


def check_gates(name, tensor, report):
    """Check gate values."""
    if tensor.dim() != 0:
        return
    if "gate" not in name.lower():
        return

    val = tensor.float().item()
    sig = torch.sigmoid(tensor.float()).item()

    report.info(name, f"raw={val:+.4f}, sigmoid={sig:.6f}")

    if sig > 0.95:
        report.warn(name, f"Gate nearly saturated open: {sig:.4f}")
    elif sig < 0.001:
        report.info(name, f"Gate effectively closed: {sig:.6f}")


def check_layernorm(name, tensor, report):
    """Check LayerNorm/RMSNorm weight values (should be near 1.0)."""
    if not any(kw in name.lower() for kw in ["layernorm", "rmsnorm", "norm.weight"]):
        return
    if tensor.dim() != 1:
        return

    t = tensor.float()
    mean_val = t.mean().item()
    std_val = t.std().item()

    if abs(mean_val - 1.0) > 0.5:
        report.warn(name, f"Norm weights drifted from 1.0: mean={mean_val:.4f}")
    if std_val > 0.5:
        report.warn(name, f"Norm weights high variance: std={std_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Model health check")
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all checks")
    parser.add_argument("--export", type=str, default=None, help="Export report to JSON")
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    params, meta = load_state_dict(args.model, hf=args.hf)

    if meta:
        print(f"Phase: {meta.get('phase', '?')}, Step: {meta.get('step', '?')}")

    report = HealthReport()
    layer_norms = defaultdict(list)

    total_params = 0
    total_tensors = 0

    for name, tensor in params.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        total_tensors += 1
        total_params += tensor.numel()

        check_dead_neurons(name, tensor, report)
        check_saturation(name, tensor, report)
        check_norm_health(name, tensor, report, layer_norms)
        check_distribution(name, tensor, report)
        check_gates(name, tensor, report)
        check_layernorm(name, tensor, report)

    for layer_type, norms in layer_norms.items():
        if len(norms) < 3:
            continue
        vals = [n for _, n in norms]
        mean_norm = sum(vals) / len(vals)
        max_norm = max(vals)
        min_norm = min(vals)
        if mean_norm > 0 and max_norm / max(min_norm, 1e-10) > 100:
            worst = max(norms, key=lambda x: x[1])
            report.warn(f"{layer_type} (cross-layer)",
                       f"Norm variance: min={min_norm:.4f}, max={max_norm:.4f}, worst={worst[0]}")

    print(f"\n{'='*80}")
    print(f"HEALTH REPORT")
    print(f"{'='*80}")
    print(f"Tensors: {total_tensors:,}  |  Parameters: {total_params:,}")

    errors = [i for i in report.issues if i[0] == "ERROR"]
    warns = [i for i in report.issues if i[0] == "WARN"]
    infos = [i for i in report.issues if i[0] == "INFO"]

    if errors:
        print(f"\n[ERRORS] ({len(errors)})")
        for sev, name, msg in errors:
            print(f"  {name}")
            print(f"    {msg}")

    if warns:
        print(f"\n[WARNINGS] ({len(warns)})")
        for sev, name, msg in warns:
            print(f"  {name}")
            print(f"    {msg}")

    if args.verbose and infos:
        print(f"\n[INFO] ({len(infos)})")
        for sev, name, msg in infos:
            print(f"  {name}: {msg}")

    if not errors and not warns:
        print(f"\nAll clear. No issues detected.")

    print(f"\n{'='*80}")
    if errors:
        print(f"VERDICT: UNHEALTHY ({len(errors)} errors, {len(warns)} warnings)")
    elif warns:
        print(f"VERDICT: OK with {len(warns)} warnings")
    else:
        print(f"VERDICT: HEALTHY")
    print(f"{'='*80}")

    if args.export:
        export = {
            "model": args.model,
            "meta": meta,
            "total_tensors": total_tensors,
            "total_params": total_params,
            "errors": len(errors),
            "warnings": len(warns),
            "issues": [{"severity": s, "name": n, "message": m} for s, n, m in report.issues],
        }
        with open(args.export, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()
