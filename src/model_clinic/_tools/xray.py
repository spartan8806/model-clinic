"""X-ray: Per-parameter weight inspection.

Usage:
    model-xray checkpoint.pt
    model-xray checkpoint.pt --filter "layer.3"
    model-xray checkpoint.pt --sort norm --top 20
    model-xray Qwen/Qwen2.5-0.5B-Instruct --hf
    model-xray checkpoint.pt --export stats.json
"""

import argparse
import json
from collections import defaultdict

import torch

from model_clinic._loader import load_state_dict


def param_stats(name, tensor):
    """Compute stats for a single parameter."""
    t = tensor.float()
    numel = t.numel()

    stats = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "numel": numel,
        "mean": t.mean().item(),
        "std": t.std().item() if numel > 1 else 0.0,
        "min": t.min().item(),
        "max": t.max().item(),
        "abs_mean": t.abs().mean().item(),
        "norm": t.norm().item(),
        "sparsity": (t.abs() < 1e-8).float().mean().item(),
    }

    if tensor.dim() == 0:
        stats["value"] = t.item()
        if "gate" in name.lower():
            stats["sigmoid"] = torch.sigmoid(t).item()

    return stats


def group_params(all_stats):
    """Group params by module prefix."""
    groups = defaultdict(list)
    for s in all_stats:
        parts = s["name"].split(".")
        if len(parts) >= 3:
            group = ".".join(parts[:3])
        elif len(parts) >= 2:
            group = ".".join(parts[:2])
        else:
            group = parts[0]
        groups[group].append(s)
    return dict(groups)


def format_stats(stats, verbose=False):
    """Format stats for display."""
    name = stats["name"]
    shape = "x".join(str(s) for s in stats["shape"])
    numel = stats["numel"]

    line = f"  {name:<60s} {shape:>20s} ({numel:>10,})"

    if "value" in stats:
        val = stats["value"]
        extra = f"  val={val:+.6f}"
        if "sigmoid" in stats:
            extra += f" (sig={stats['sigmoid']:.6f})"
        line += extra
    else:
        line += (
            f"  mean={stats['mean']:+.4e}  std={stats['std']:.4e}"
            f"  [{stats['min']:+.4e}, {stats['max']:+.4e}]"
        )
        if verbose:
            line += f"  norm={stats['norm']:.4f}  sparsity={stats['sparsity']:.1%}"

    return line


def main():
    parser = argparse.ArgumentParser(description="Inspect model parameters")
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--filter", type=str, default=None, help="Filter params by name substring")
    parser.add_argument("--sort", choices=["name", "norm", "std", "numel", "sparsity"],
                        default="name", help="Sort order")
    parser.add_argument("--top", type=int, default=0, help="Show only top N params")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all stats")
    parser.add_argument("--export", type=str, default=None, help="Export stats to JSON")
    parser.add_argument("--summary", action="store_true", help="Summary only, no per-param")
    args = parser.parse_args()

    print(f"Loading: {args.model}")
    params, meta = load_state_dict(args.model, hf=args.hf)
    print(f"Source: {meta.get('source', 'unknown')}")

    if "extra" in meta:
        extra = meta["extra"]
        if extra:
            print(f"Phase: {extra.get('phase', '?')}, Step: {extra.get('step', '?')}")

    all_stats = []
    for name, tensor in params.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if args.filter and args.filter not in name:
            continue
        all_stats.append(param_stats(name, tensor))

    if not all_stats:
        print("No parameters found matching filter.")
        return

    if args.sort == "norm":
        all_stats.sort(key=lambda s: s["norm"], reverse=True)
    elif args.sort == "std":
        all_stats.sort(key=lambda s: s["std"], reverse=True)
    elif args.sort == "numel":
        all_stats.sort(key=lambda s: s["numel"], reverse=True)
    elif args.sort == "sparsity":
        all_stats.sort(key=lambda s: s["sparsity"], reverse=True)

    if args.top:
        all_stats = all_stats[:args.top]

    total_params = sum(s["numel"] for s in all_stats)
    total_bytes = sum(
        s["numel"] * (2 if "float16" in s["dtype"] or "bfloat16" in s["dtype"] else 4)
        for s in all_stats
    )
    print(f"\nParameters: {len(all_stats):,} tensors, {total_params:,} values ({total_bytes/1024**2:.1f} MB)")

    avg_sparsity = sum(s["sparsity"] * s["numel"] for s in all_stats) / max(total_params, 1)
    print(f"Avg sparsity: {avg_sparsity:.2%}")

    gates = [s for s in all_stats if "gate" in s["name"].lower() and "value" in s]
    if gates:
        print(f"\nGates:")
        for g in gates:
            sig = g.get("sigmoid", "N/A")
            print(f"  {g['name']}: raw={g['value']:+.4f}, sigmoid={sig}")

    if args.summary:
        groups = group_params(all_stats)
        print(f"\nModule groups ({len(groups)}):")
        for group, stats_list in sorted(groups.items()):
            g_params = sum(s["numel"] for s in stats_list)
            g_norm = sum(s["norm"] ** 2 for s in stats_list) ** 0.5
            print(f"  {group:<50s} {len(stats_list):3d} tensors  {g_params:>12,} params  norm={g_norm:.4f}")
    else:
        print(f"\n{'='*120}")
        for s in all_stats:
            print(format_stats(s, verbose=args.verbose))

    if args.export:
        export_data = {
            "model": args.model,
            "meta": {k: str(v) for k, v in meta.items()},
            "total_params": total_params,
            "total_bytes": total_bytes,
            "params": all_stats,
        }
        with open(args.export, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
