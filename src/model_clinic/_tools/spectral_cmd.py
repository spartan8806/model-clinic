"""Spectral Surgery — SVD-based analysis and denoising of weight matrices.

Usage:
    model-clinic spectral checkpoint.pt
    model-clinic spectral checkpoint.pt --repair
    model-clinic spectral checkpoint.pt --repair --save treated.pt
    model-clinic spectral checkpoint.pt --param layers.0.self_attn.q_proj.weight
    model-clinic spectral checkpoint.pt --json
    model-clinic spectral checkpoint.pt --hf
"""

import argparse
import json
import sys

import torch

from model_clinic._loader import load_state_dict, save_state_dict
from model_clinic._repair.spectral import (
    spectral_analysis,
    spectral_denoise_with_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Spectral Surgery — SVD-based analysis and denoising"
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--repair", action="store_true",
                        help="Apply spectral denoising (without this flag, analysis only)")
    parser.add_argument("--save", "-o", type=str, default=None,
                        help="Save treated checkpoint to this path (requires --repair)")
    parser.add_argument("--param", type=str, default=None,
                        help="Analyze/repair a single parameter by name")
    parser.add_argument("--energy", type=float, default=0.99,
                        help="Energy threshold for rank selection (default: 0.99)")
    parser.add_argument("--max-condition", type=float, default=1000,
                        help="Maximum condition number after denoising (default: 1000)")
    parser.add_argument("--min-rank-ratio", type=float, default=0.1,
                        help="Minimum rank ratio to prevent over-truncation (default: 0.1)")
    parser.add_argument("--json", action="store_true", dest="json_output",
                        help="Output in JSON format")
    args = parser.parse_args()

    log = (lambda msg: print(msg, file=sys.stderr)) if args.json_output else print

    log(f"Loading: {args.model}")
    state_dict, meta = load_state_dict(args.model, hf=args.hf)

    # Filter to 2D+ tensors
    candidates = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.dim() < 2 or tensor.shape[0] < 2 or tensor.shape[1] < 2:
            continue
        if args.param and name != args.param:
            continue
        candidates[name] = tensor

    if args.param and not candidates:
        print(f"Parameter '{args.param}' not found or not a 2D+ tensor.", file=sys.stderr)
        sys.exit(1)

    log(f"Analyzing {len(candidates)} weight matrices...")

    if args.repair:
        _do_repair(state_dict, candidates, args, log)
    else:
        _do_analysis(candidates, args, log)


def _do_analysis(candidates, args, log):
    """Analyze tensors and display SVD spectrum summary."""
    results = []
    for name, tensor in candidates.items():
        info = spectral_analysis(tensor)
        info["name"] = name
        results.append(info)

    # Sort by condition number descending (worst first)
    results.sort(key=lambda r: r["condition_number"]
                 if r["condition_number"] != float("inf") else 1e18,
                 reverse=True)

    if args.json_output:
        output = []
        for r in results:
            output.append({
                "name": r["name"],
                "shape": r["shape"],
                "effective_rank": r["effective_rank"],
                "condition_number": r["condition_number"]
                    if r["condition_number"] != float("inf") else None,
                "total_energy": r["total_energy"],
            })
        print(json.dumps(output, indent=2))
        return

    # Table header
    print(f"\n{'Parameter':<55s} {'Shape':>12s}  {'Eff.Rank':>8s}  {'Cond#':>12s}  Status")
    print("-" * 100)

    n_high_cond = 0
    for r in results:
        shape_str = "x".join(str(s) for s in r["shape"])
        cond = r["condition_number"]
        if cond == float("inf"):
            cond_str = "inf"
            status = "DEGENERATE"
        elif cond > 1_000_000:
            cond_str = f"{cond:.0f}"
            status = "CRITICAL"
            n_high_cond += 1
        elif cond > 10_000:
            cond_str = f"{cond:.0f}"
            status = "HIGH"
            n_high_cond += 1
        elif cond > 1_000:
            cond_str = f"{cond:.0f}"
            status = "elevated"
        else:
            cond_str = f"{cond:.0f}"
            status = "healthy"

        print(f"  {r['name']:<55s} [{shape_str:>12s}]  "
              f"{r['effective_rank']:>8d}  {cond_str:>12s}  {status}")

    print(f"\nSummary: {n_high_cond}/{len(results)} tensors with condition number > 10K")
    if n_high_cond > 0:
        print("Run with --repair to apply spectral denoising.")


def _do_repair(state_dict, candidates, args, log):
    """Apply spectral denoising and optionally save."""
    reports = []
    n_modified = 0

    for name, tensor in candidates.items():
        denoised, report = spectral_denoise_with_report(
            tensor, name,
            energy_threshold=args.energy,
            max_condition=args.max_condition,
            min_rank_ratio=args.min_rank_ratio,
        )
        reports.append(report)

        if report.effective_rank < report.original_rank:
            state_dict[name] = denoised
            n_modified += 1
            log(f"  {name}: rank {report.original_rank} -> {report.effective_rank}, "
                f"cond {report.condition_before:.0f} -> {report.condition_after:.0f}, "
                f"energy {report.energy_retained:.4f}, "
                f"error {report.frobenius_error:.4f}")

    if args.json_output:
        output = []
        for r in reports:
            if r.effective_rank < r.original_rank:
                output.append({
                    "param_name": r.param_name,
                    "original_rank": r.original_rank,
                    "effective_rank": r.effective_rank,
                    "energy_retained": r.energy_retained,
                    "condition_before": r.condition_before,
                    "condition_after": r.condition_after,
                    "frobenius_error": r.frobenius_error,
                })
        print(json.dumps(output, indent=2))
    else:
        log(f"\nModified {n_modified}/{len(candidates)} tensors.")

    if args.save and n_modified > 0:
        save_state_dict(state_dict, args.save)
        log(f"Saved treated checkpoint to {args.save}")
    elif args.save and n_modified == 0:
        log("No tensors needed denoising — nothing saved.")


if __name__ == "__main__":
    main()
