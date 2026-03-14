"""Cross-checkpoint grafting — pick the healthiest version of each parameter.

Given multiple checkpoints of the same architecture, score each parameter
independently and take the best version of each. The result is a merged
state dict that combines the healthiest parts of every checkpoint.

Usage:
    from model_clinic._repair.graft import graft, graft_modules, score_parameter

    merged, manifest = graft(["ckpt_a.pt", "ckpt_b.pt"])
    merged, manifest = graft_modules(["ckpt_a.pt", "ckpt_b.pt"])
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from model_clinic._loader import load_state_dict
from model_clinic._health_score import compute_health_score
from model_clinic.clinic import diagnose


# ── Scoring ──────────────────────────────────────────────────────────────


def score_parameter(key: str, tensor: torch.Tensor) -> float:
    """Compute a health score (0-100) for a single parameter tensor.

    Parameters
    ----------
    key : str
        The parameter name (e.g. "layers.0.attention.weight").
    tensor : torch.Tensor
        The parameter tensor.

    Returns
    -------
    float
        Health score from 0 (broken) to 100 (perfect).
    """
    # Edge cases: scalar, 1D with few elements, very small tensors
    # These can't meaningfully fail diagnostics, so score them perfect.
    if tensor.dim() == 0 or tensor.numel() < 4:
        return 100.0

    # Run diagnosis on a single-tensor state dict
    findings = diagnose({key: tensor}, verbose=False, plugins=False)
    score = compute_health_score(findings)
    return float(score.overall)


# ── Manifest ─────────────────────────────────────────────────────────────


@dataclass
class GraftManifest:
    """Record of which checkpoint each parameter was sourced from."""

    sources: dict = field(default_factory=dict)
    # sources[param_name] = {"source_checkpoint": str, "score": float, "runner_up_score": float|None}

    summary: dict = field(default_factory=dict)
    # summary = {"total_params": int, "params_from_each_source": {path: count},
    #            "overall_score_before": {path: float}, "overall_score_after": float}

    def to_json(self) -> dict:
        """Return a JSON-serializable dict."""
        return {
            "sources": dict(self.sources),
            "summary": dict(self.summary),
        }

    def print_report(self, file=None):
        """Print a human-readable graft report."""
        out = file or sys.stdout
        s = self.summary

        print("\n\033[1mGraft Report\033[0m", file=out)
        print("─" * 50, file=out)
        print(f"  Total parameters: {s.get('total_params', 0)}", file=out)
        print(file=out)

        # Per-source counts
        counts = s.get("params_from_each_source", {})
        before_scores = s.get("overall_score_before", {})
        for path, count in counts.items():
            before = before_scores.get(path, "?")
            print(f"  {path}: {count} params (checkpoint score: {before})", file=out)

        print(file=out)
        after = s.get("overall_score_after", "?")
        print(f"  Merged score: {after}", file=out)
        print(file=out)

        # Show any interesting picks (where runner-up was significantly worse)
        interesting = []
        for key, info in self.sources.items():
            runner_up = info.get("runner_up_score")
            if runner_up is not None and info["score"] - runner_up >= 10:
                interesting.append((key, info["score"], runner_up))

        if interesting:
            interesting.sort(key=lambda x: x[1] - x[2], reverse=True)
            print("  Notable picks (score gap >= 10):", file=out)
            for key, best, runner in interesting[:20]:
                print(f"    {key}: {best:.0f} vs {runner:.0f}", file=out)
            if len(interesting) > 20:
                print(f"    ... and {len(interesting) - 20} more", file=out)
            print(file=out)


# ── Core grafting ────────────────────────────────────────────────────────


def _load_checkpoints(checkpoints):
    """Load all checkpoints, returning list of (path, state_dict) pairs."""
    results = []
    for cp in checkpoints:
        sd, _meta = load_state_dict(str(cp))
        results.append((str(cp), sd))
    return results


def _all_keys(loaded_checkpoints):
    """Collect all parameter keys across all checkpoints."""
    keys = {}
    for _path, sd in loaded_checkpoints:
        for k in sd:
            keys[k] = True
    return list(keys.keys())


def _compute_checkpoint_scores(loaded_checkpoints):
    """Compute overall health score for each full checkpoint."""
    scores = {}
    for path, sd in loaded_checkpoints:
        findings = diagnose(sd, verbose=False, plugins=False)
        hs = compute_health_score(findings)
        scores[path] = float(hs.overall)
    return scores


def graft(checkpoints, strategy="best_per_layer"):
    """Graft the healthiest version of each parameter from multiple checkpoints.

    Parameters
    ----------
    checkpoints : list of str/Path
        Paths to checkpoint files (.pt, .pth, .safetensors, etc.).
    strategy : str
        Currently only "best_per_layer" is supported.

    Returns
    -------
    (merged_state_dict, GraftManifest)
        The merged state dict and a manifest documenting provenance.
    """
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")

    loaded = _load_checkpoints(checkpoints)

    # Single checkpoint: just return it
    if len(loaded) == 1:
        path, sd = loaded[0]
        findings = diagnose(sd, verbose=False, plugins=False)
        hs = compute_health_score(findings)
        manifest = GraftManifest(
            sources={k: {"source_checkpoint": path, "score": score_parameter(k, v), "runner_up_score": None}
                     for k, v in sd.items()},
            summary={
                "total_params": len(sd),
                "params_from_each_source": {path: len(sd)},
                "overall_score_before": {path: float(hs.overall)},
                "overall_score_after": float(hs.overall),
            },
        )
        return dict(sd), manifest

    # Compute full-checkpoint scores for reporting
    checkpoint_scores = _compute_checkpoint_scores(loaded)

    all_keys = _all_keys(loaded)
    merged = {}
    sources = {}
    source_counts = defaultdict(int)

    for key in all_keys:
        # Gather candidates: (path, tensor) for checkpoints that have this key
        candidates = []
        for path, sd in loaded:
            if key in sd:
                candidates.append((path, sd[key]))

        if len(candidates) == 1:
            # Only one checkpoint has this param — take it
            path, tensor = candidates[0]
            s = score_parameter(key, tensor)
            merged[key] = tensor.clone()
            sources[key] = {
                "source_checkpoint": path,
                "score": s,
                "runner_up_score": None,
            }
            source_counts[path] += 1
            continue

        # Score each candidate
        scored = []
        for path, tensor in candidates:
            s = score_parameter(key, tensor)
            scored.append((s, path, tensor))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_path, best_tensor = scored[0]
        runner_up_score = scored[1][0] if len(scored) > 1 else None

        merged[key] = best_tensor.clone()
        sources[key] = {
            "source_checkpoint": best_path,
            "score": best_score,
            "runner_up_score": runner_up_score,
        }
        source_counts[best_path] += 1

    # Compute merged score
    merged_findings = diagnose(merged, verbose=False, plugins=False)
    merged_hs = compute_health_score(merged_findings)

    manifest = GraftManifest(
        sources=sources,
        summary={
            "total_params": len(merged),
            "params_from_each_source": dict(source_counts),
            "overall_score_before": checkpoint_scores,
            "overall_score_after": float(merged_hs.overall),
        },
    )

    return merged, manifest


# ── Module-level grafting ────────────────────────────────────────────────


def _group_keys_by_module(keys, depth=2):
    """Group parameter keys by their module prefix up to a given depth.

    Example with depth=2:
        "layers.0.attention.weight" -> "layers.0"
        "layers.0.attention.bias"   -> "layers.0"
        "embed.weight"              -> "embed"

    Returns dict of {module_prefix: [key1, key2, ...]}.
    """
    groups = defaultdict(list)
    for key in keys:
        parts = key.split(".")
        prefix = ".".join(parts[:depth]) if len(parts) > depth else ".".join(parts[:-1]) if len(parts) > 1 else key
        groups[prefix].append(key)
    return dict(groups)


def _score_module_group(keys, state_dict):
    """Score a group of parameters as a unit by diagnosing them together.

    Parameters
    ----------
    keys : list of str
        Parameter names in this module group.
    state_dict : dict
        Full state dict to pull tensors from.

    Returns
    -------
    float
        Health score for this module group.
    """
    subset = {k: state_dict[k] for k in keys if k in state_dict}
    if not subset:
        return 100.0
    findings = diagnose(subset, verbose=False, plugins=False)
    hs = compute_health_score(findings)
    return float(hs.overall)


def graft_modules(checkpoints, module_level=True, depth=2):
    """Graft at the module level rather than individual parameters.

    Groups parameters by module prefix (e.g. "layers.0") and picks the
    healthiest version of each entire module group. This preserves
    intra-module coherence.

    Parameters
    ----------
    checkpoints : list of str/Path
        Paths to checkpoint files.
    module_level : bool
        If True (default), group by module. If False, falls back to
        per-parameter grafting via ``graft()``.
    depth : int
        How many levels of the parameter name to use for grouping.
        depth=2 groups "layers.0.attn.weight" as "layers.0".

    Returns
    -------
    (merged_state_dict, GraftManifest)
    """
    if not module_level:
        return graft(checkpoints)

    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")

    loaded = _load_checkpoints(checkpoints)

    if len(loaded) == 1:
        return graft(checkpoints)

    checkpoint_scores = _compute_checkpoint_scores(loaded)

    # Collect all keys and group them
    all_keys = _all_keys(loaded)
    groups = _group_keys_by_module(all_keys, depth=depth)

    merged = {}
    sources = {}
    source_counts = defaultdict(int)

    for module_prefix, keys_in_group in groups.items():
        # For each module group, score every checkpoint that has those keys
        scored = []
        for path, sd in loaded:
            # Check which keys this checkpoint has for this group
            available_keys = [k for k in keys_in_group if k in sd]
            if not available_keys:
                continue
            module_score = _score_module_group(available_keys, sd)
            scored.append((module_score, path, available_keys))

        if not scored:
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_path, best_keys = scored[0]
        runner_up_score = scored[1][0] if len(scored) > 1 else None

        # Copy all keys from the winning checkpoint for this module
        best_sd = dict(loaded)[best_path] if False else None
        for p, sd in loaded:
            if p == best_path:
                best_sd = sd
                break

        for key in keys_in_group:
            if key in best_sd:
                merged[key] = best_sd[key].clone()
                sources[key] = {
                    "source_checkpoint": best_path,
                    "score": best_score,
                    "runner_up_score": runner_up_score,
                }
                source_counts[best_path] += 1
            else:
                # Key not in winning checkpoint — find it elsewhere
                for path, sd in loaded:
                    if key in sd:
                        param_score = score_parameter(key, sd[key])
                        merged[key] = sd[key].clone()
                        sources[key] = {
                            "source_checkpoint": path,
                            "score": param_score,
                            "runner_up_score": None,
                        }
                        source_counts[path] += 1
                        break

    # Compute merged score
    merged_findings = diagnose(merged, verbose=False, plugins=False)
    merged_hs = compute_health_score(merged_findings)

    manifest = GraftManifest(
        sources=sources,
        summary={
            "total_params": len(merged),
            "params_from_each_source": dict(source_counts),
            "overall_score_before": checkpoint_scores,
            "overall_score_after": float(merged_hs.overall),
        },
    )

    return merged, manifest
