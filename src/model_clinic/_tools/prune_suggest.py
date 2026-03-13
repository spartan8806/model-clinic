"""Prune-suggest — static pruning opportunity analysis.

No forward pass required. Analyses weight tensors for:
  - Low effective rank (SVD-based rank utilization)
  - High sparsity (large fraction of near-zero weights)
  - Identical / redundant rows
  - Redundant attention heads

Usage:
    model-clinic prune-suggest model.pt
    model-clinic prune-suggest model.pt --json
    model-clinic prune-suggest model.pt --min-size 1000
"""

import argparse
import json

import torch


# ── Analysis helpers ──────────────────────────────────────────────────────


def _rank_utilization(tensor, max_dim=256):
    """Estimate rank utilization via effective rank / min(m, n).

    Returns float in [0, 1]. Lower = more prunable.
    Returns None if SVD fails or tensor is too small.
    """
    if tensor.dim() != 2:
        return None
    m, n = tensor.shape
    if min(m, n) < 4:
        return None
    t = tensor.float()
    if m > max_dim:
        t = t[torch.randperm(m)[:max_dim]]
    if t.shape[1] > max_dim:
        t = t[:, torch.randperm(t.shape[1])[:max_dim]]
    try:
        sv = torch.linalg.svdvals(t)
        sv_pos = sv[sv > 1e-10]
        if len(sv_pos) < 2:
            return 0.0
        nuclear = sv_pos.sum().item()
        spectral = sv_pos[0].item()
        eff_rank = nuclear / spectral if spectral > 0 else 0.0
        min_dim = float(min(t.shape))
        return eff_rank / min_dim if min_dim > 0 else 0.0
    except Exception:
        return None


def _sparsity(tensor, near_zero_threshold=1e-3):
    """Fraction of elements with absolute value below threshold.

    Returns float in [0, 1]. Higher = more sparse.
    """
    if tensor.numel() == 0:
        return 0.0
    t = tensor.float()
    near_zero = (t.abs() < near_zero_threshold).float().mean().item()
    return near_zero


def _has_identical_rows(tensor, similarity_threshold=0.999, sample_size=200):
    """Return fraction of near-identical row pairs (upper triangle)."""
    if tensor.dim() != 2 or tensor.shape[0] < 4:
        return 0.0
    n = min(tensor.shape[0], sample_size)
    sample = tensor[:n].float()
    norms = sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = sample / norms
    sims = normed @ normed.T
    triu = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    total_pairs = int(triu.sum().item())
    if total_pairs == 0:
        return 0.0
    dup_pairs = int((sims[triu] > similarity_threshold).sum().item())
    return dup_pairs / total_pairs


def prune_suggestions(state_dict, min_size=100):
    """Analyse a state dict for pruning opportunities.

    Args:
        state_dict: model state dict (dict of str -> torch.Tensor)
        min_size: skip tensors with fewer elements than this (default 100)

    Returns:
        list of dicts, each with keys:
            tensor_name (str)
            reason (str)           -- human-readable explanation
            suggested_amount (float)  -- fraction to prune, 0..1
            risk_level (str)       -- "low", "moderate", "high"
            detail (dict)          -- supporting statistics
    """
    suggestions = []

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not tensor.is_floating_point():
            continue
        if tensor.numel() < min_size:
            continue

        # Skip norm layers (resetting them would break normalisation)
        name_lower = name.lower()
        is_norm = any(kw in name_lower for kw in
                      ["layernorm", "rmsnorm", "norm.weight", "norm.bias"])
        if is_norm:
            continue

        # ── Check 1: low rank utilization ───────────────────────────────
        rank_util = _rank_utilization(tensor)
        if rank_util is not None and rank_util < 0.20:
            # Map rank_util -> suggested prune amount
            # rank_util=0.05 → prune 80%, rank_util=0.15 → prune 60%
            suggested = max(0.50, min(0.90, 1.0 - rank_util * 2.5))
            risk = "low" if rank_util < 0.10 else "moderate"
            suggestions.append({
                "tensor_name": name,
                "reason": f"low rank utilization ({rank_util:.1%}) -- matrix is low-rank",
                "suggested_amount": round(suggested, 2),
                "risk_level": risk,
                "detail": {
                    "rank_utilization": round(rank_util, 4),
                    "shape": list(tensor.shape),
                    "method": "l1_unstructured or structured row/col pruning",
                },
            })
            continue  # don't double-report the same tensor

        # ── Check 2: high sparsity ───────────────────────────────────────
        sparsity = _sparsity(tensor)
        if sparsity > 0.50:
            # High sparsity means many near-zero weights; safe to prune
            risk = "low" if sparsity > 0.70 else "moderate"
            suggestions.append({
                "tensor_name": name,
                "reason": f"{sparsity:.1%} near-zero values -- high weight sparsity",
                "suggested_amount": round(min(sparsity * 0.9, 0.90), 2),
                "risk_level": risk,
                "detail": {
                    "sparsity": round(sparsity, 4),
                    "shape": list(tensor.shape),
                    "method": "l1_unstructured pruning",
                },
            })
            continue

        # ── Check 3: redundant rows ──────────────────────────────────────
        if tensor.dim() == 2:
            dup_frac = _has_identical_rows(tensor)
            if dup_frac > 0.10:
                suggestions.append({
                    "tensor_name": name,
                    "reason": f"{dup_frac:.1%} near-identical row pairs -- redundant rows",
                    "suggested_amount": round(min(dup_frac * 0.8, 0.50), 2),
                    "risk_level": "moderate",
                    "detail": {
                        "duplicate_row_fraction": round(dup_frac, 4),
                        "shape": list(tensor.shape),
                        "method": "structured row pruning",
                    },
                })

    # Sort: low-risk first, then by suggested amount descending
    risk_order = {"low": 0, "moderate": 1, "high": 2}
    suggestions.sort(key=lambda x: (risk_order[x["risk_level"]], -x["suggested_amount"]))
    return suggestions


def _estimate_size_reduction(state_dict, suggestions):
    """Rough estimate of how many parameters would be removed."""
    total_params = sum(
        t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor)
    )
    removed = 0
    for s in suggestions:
        name = s["tensor_name"]
        t = state_dict.get(name)
        if isinstance(t, torch.Tensor):
            removed += int(t.numel() * s["suggested_amount"])
    return total_params, removed


def _print_suggestions(suggestions, state_dict, model_path):
    """Print formatted prune-suggest output."""
    SEP = "=" * 60
    print(f"\nPruning Opportunities ({model_path}):")
    print(SEP)

    if not suggestions:
        print("No significant pruning opportunities detected.")
        print("Model weights appear to be well-utilized.")
        return

    low = [s for s in suggestions if s["risk_level"] == "low"]
    moderate = [s for s in suggestions if s["risk_level"] == "moderate"]
    high = [s for s in suggestions if s["risk_level"] == "high"]

    if low:
        print("\nConservative pruning targets (low risk):")
        for s in low:
            pct = int(s["suggested_amount"] * 100)
            print(f"  {s['tensor_name']}")
            print(f"    {s['reason']}")
            print(f"    Suggested: prune {pct}% of weights")
            print()

    if moderate:
        print("Moderate pruning targets:")
        for s in moderate:
            pct = int(s["suggested_amount"] * 100)
            print(f"  {s['tensor_name']}")
            print(f"    {s['reason']}")
            print(f"    Suggested: prune {pct}% of weights")
            print()

    if high:
        print("High-risk targets (evaluate carefully before pruning):")
        for s in high:
            pct = int(s["suggested_amount"] * 100)
            print(f"  {s['tensor_name']}")
            print(f"    {s['reason']}")
            print(f"    Suggested: prune {pct}% of weights")
            print()

    # Size estimate
    total_params, removed = _estimate_size_reduction(state_dict, suggestions)
    if total_params > 0:
        pct_removed = removed / total_params
        bytes_total = total_params * 4  # fp32
        bytes_after = bytes_total * (1 - pct_removed)
        gb_total = bytes_total / 1e9
        gb_after = bytes_after / 1e9
        print(f"Estimated size reduction: ~{pct_removed:.0%} "
              f"(from {gb_total:.2f}GB to ~{gb_after:.2f}GB at fp32)")
        print()

    # Commands
    print("Commands to apply (PyTorch built-in pruning):")
    for s in (low + moderate)[:3]:
        amt = s["suggested_amount"]
        name = s["tensor_name"]
        # Suggest the method based on detail
        method = s["detail"].get("method", "l1_unstructured")
        if "row" in method:
            print(f"  # {name}")
            print(f"  torch.nn.utils.prune.ln_structured(module, 'weight', amount={amt}, n=1, dim=0)")
        else:
            print(f"  # {name}")
            print(f"  torch.nn.utils.prune.l1_unstructured(module, 'weight', amount={amt})")
    print()
    print("Note: Always evaluate accuracy after pruning. "
          "Use gradual pruning for best results.")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic prune-suggest",
        description="Identify static pruning opportunities in a checkpoint",
    )
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--safetensors", action="store_true",
                        help="Force safetensors loading")
    parser.add_argument("--min-size", type=int, default=100,
                        help="Skip tensors with fewer than N elements (default: 100)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--export", type=str, default=None,
                        help="Export JSON to this path")
    args = parser.parse_args()

    from model_clinic._loader import load_state_dict

    print(f"Loading: {args.model}")
    state_dict, _meta = load_state_dict(args.model, hf=args.hf)
    print(f"Loaded {len(state_dict)} tensors")

    suggestions = prune_suggestions(state_dict, min_size=args.min_size)

    if args.json:
        total_params, removed = _estimate_size_reduction(state_dict, suggestions)
        out = {
            "model": args.model,
            "total_params": total_params,
            "estimated_removed": removed,
            "suggestions": suggestions,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        _print_suggestions(suggestions, state_dict, args.model)

    if args.export:
        total_params, removed = _estimate_size_reduction(state_dict, suggestions)
        out = {
            "model": args.model,
            "total_params": total_params,
            "estimated_removed": removed,
            "suggestions": suggestions,
        }
        with open(args.export, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"Exported to {args.export}")


if __name__ == "__main__":
    main()
