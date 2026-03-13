"""MRI: Deep per-layer weight analysis using SVD decomposition.

Usage:
    model-clinic mri checkpoint.pt
    model-clinic mri checkpoint.pt --json
    model-clinic mri checkpoint.pt --top 10
    model-clinic mri model_name --hf
    model-clinic mri checkpoint.pt --max-layers 50
"""

import argparse
import json
import sys

from model_clinic._loader import load_state_dict
from model_clinic._mri import model_mri, mri_summary, LayerMRI


def _status_label(layer: LayerMRI) -> str:
    """Return a short status label for a layer."""
    if layer.is_degenerate:
        return "DEGENERATE"
    if layer.is_low_rank:
        return "LOW RANK"
    if layer.is_heavy_tailed:
        return "HEAVY TAIL"
    if layer.is_sparse:
        return "SPARSE"
    return "healthy"


def _layer_to_dict(layer: LayerMRI) -> dict:
    """Convert LayerMRI to a JSON-serializable dict."""
    return {
        "name": layer.name,
        "shape": list(layer.shape),
        "dtype": layer.dtype,
        "effective_rank": round(layer.effective_rank, 4),
        "numerical_rank": layer.numerical_rank,
        "rank_utilization": round(layer.rank_utilization, 4),
        "top_sv_ratio": round(layer.top_sv_ratio, 4) if layer.top_sv_ratio != float("inf") else None,
        "entropy": round(layer.entropy, 4),
        "stable_rank": round(layer.stable_rank, 4),
        "kurtosis": round(layer.kurtosis, 4),
        "skewness": round(layer.skewness, 4),
        "sparsity": round(layer.sparsity, 4),
        "is_low_rank": layer.is_low_rank,
        "is_degenerate": layer.is_degenerate,
        "is_heavy_tailed": layer.is_heavy_tailed,
        "is_sparse": layer.is_sparse,
        "inferred_role": layer.inferred_role,
    }


def main():
    parser = argparse.ArgumentParser(description="Deep per-layer weight analysis (Model MRI)")
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output in JSON format")
    parser.add_argument("--top", type=int, default=0,
                        help="Show only the N most concerning layers")
    parser.add_argument("--max-layers", type=int, default=None,
                        help="Analyze at most N layers (for speed)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show progress during analysis")
    parser.add_argument("--export", type=str, default=None,
                        help="Export results to JSON file")
    args = parser.parse_args()

    # When --json, send progress to stderr so stdout is pure JSON
    log = (lambda msg: print(msg, file=sys.stderr)) if args.json_output else print

    log(f"Loading: {args.model}")
    state_dict, meta = load_state_dict(args.model, hf=args.hf)
    model_name = meta.get("source", args.model)

    log(f"Model MRI -- {model_name}")

    # Count 2D+ tensors
    import torch
    n_2d = sum(1 for v in state_dict.values()
               if isinstance(v, torch.Tensor) and v.dim() >= 2)
    log(f"Analyzing {n_2d} weight matrices...")
    if args.max_layers:
        log(f"(limited to first {args.max_layers})")

    results = model_mri(state_dict, max_layers=args.max_layers, verbose=args.verbose)
    summary = mri_summary(results)

    if args.json_output:
        output = {
            "model": model_name,
            "summary": summary,
            "layers": [_layer_to_dict(r) for r in results],
        }
        print(json.dumps(output, indent=2))
        return

    # Determine which layers to show
    display_results = results
    if args.top:
        # Sort by most concerning: degenerate first, then low_rank, then lowest rank_utilization
        display_results = sorted(results, key=lambda r: (
            not r.is_degenerate,
            not r.is_low_rank,
            not r.is_heavy_tailed,
            not r.is_sparse,
            r.rank_utilization,  # lower = more concerning
        ))
        display_results = display_results[:args.top]

    # Print layer analysis
    print(f"\nLayer Analysis:")
    for r in display_results:
        shape_str = "x".join(str(s) for s in r.shape)
        k = min(r.shape[0], r.shape[-1]) if len(r.shape) >= 2 else r.shape[0]
        rank_pct = r.rank_utilization * 100
        status = _status_label(r)
        print(f"  {r.name:<55s} [{shape_str:>12s}]  "
              f"rank={r.numerical_rank}/{k} ({rank_pct:4.0f}%)  "
              f"entropy={r.entropy:.2f}  {status}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Mean rank utilization: {summary['mean_rank_utilization']:.0%}")
    print(f"  Low-rank layers: {summary['n_low_rank']}/{summary['analyzed_layers']} "
          f"({summary['n_low_rank']/max(summary['analyzed_layers'],1):.0%})")
    print(f"  Degenerate layers: {summary['n_degenerate']}/{summary['analyzed_layers']} "
          f"({summary['n_degenerate']/max(summary['analyzed_layers'],1):.0%})")
    print(f"  Heavy-tailed: {summary['n_heavy_tailed']}/{summary['analyzed_layers']} "
          f"({summary['n_heavy_tailed']/max(summary['analyzed_layers'],1):.0%})")
    print(f"  Sparse layers: {summary['n_sparse']}/{summary['analyzed_layers']} "
          f"({summary['n_sparse']/max(summary['analyzed_layers'],1):.0%})")
    print(f"  Information score: {summary['information_score']}/100")

    if summary['role_distribution']:
        print(f"\n  Role distribution:")
        for role, count in sorted(summary['role_distribution'].items(),
                                   key=lambda x: -x[1]):
            print(f"    {role:<15s}: {count}")

    # Export if requested
    if args.export:
        output = {
            "model": model_name,
            "summary": summary,
            "layers": [_layer_to_dict(r) for r in results],
        }
        with open(args.export, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nExported to {args.export}")


if __name__ == "__main__":
    main()
