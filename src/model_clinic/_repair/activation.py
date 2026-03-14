"""Level 5: Activation-Guided Repair.

Runs calibration data through a live model, measures per-layer impact on
representation quality, and repairs layers that are destructive (collapse rank,
explode norms, or destroy entropy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def effective_rank(tensor: torch.Tensor) -> float:
    """SVD-based effective rank: exp(entropy of normalised singular values).

    Works on 2-D+ tensors.  Batch dimensions (all but last two) are flattened.
    For 1-D tensors, rank is 1 if nonzero, else 0.
    """
    if tensor.numel() == 0:
        return 0.0

    t = tensor.detach().float()

    if t.dim() < 2:
        return 1.0 if t.norm().item() > 0 else 0.0

    # Flatten batch dims → (M, N)
    if t.dim() > 2:
        t = t.reshape(-1, t.shape[-1])

    try:
        S = torch.linalg.svdvals(t)
    except Exception:
        return 1.0

    S = S[S > 0]
    if S.numel() == 0:
        return 0.0

    # Normalise to a probability distribution
    p = S / S.sum()
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def token_entropy(tensor: torch.Tensor) -> float:
    """Shannon entropy of the token-level norm distribution.

    Measures how uniformly the model treats different positions. Higher entropy
    means more uniform treatment.  Expects shape (batch, seq_len, hidden) or
    (seq_len, hidden).
    """
    t = tensor.detach().float()

    if t.dim() < 2:
        return 0.0

    # Treat everything-but-last as the token dimension
    if t.dim() == 2:
        norms = t.norm(dim=-1)  # (seq_len,)
    elif t.dim() == 3:
        norms = t.norm(dim=-1).mean(dim=0)  # average across batch → (seq_len,)
    else:
        t = t.reshape(-1, t.shape[-1])
        norms = t.norm(dim=-1)

    total = norms.sum()
    if total.item() == 0:
        return 0.0

    p = norms / total
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    return entropy


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayerStats:
    """Per-layer activation statistics."""
    name: str
    input_rank: float
    output_rank: float
    rank_change: float
    input_entropy: float
    output_entropy: float
    entropy_change: float
    cosine_similarity: float
    norm_ratio: float


@dataclass
class ActivationReport:
    """Summary of an activation audit + optional repair."""
    all_stats: Dict[str, LayerStats]
    destructive_layers: List[str]
    repairs_applied: List[str] = field(default_factory=list)
    strategy_used: Optional[str] = None


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def activation_audit(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: str = "cpu",
) -> Dict[str, LayerStats]:
    """Run calibration data through *model* and measure per-layer impact.

    Args:
        model: A ``torch.nn.Module`` in a loadable state.
        calibration_data: List of input tensors (e.g. token-ID tensors).
        device: Device to run on.

    Returns:
        Dict mapping layer name → :class:`LayerStats`.
    """
    model = model.to(device)
    model.eval()

    # Accumulators: name → list of per-sample dicts
    accum: Dict[str, List[dict]] = {}

    hooks = []

    def _make_hook(name: str):
        def hook_fn(module, inp, out):
            inp_t = inp[0] if isinstance(inp, tuple) else inp
            out_t = out[0] if isinstance(out, tuple) else out

            if not isinstance(inp_t, torch.Tensor) or not isinstance(out_t, torch.Tensor):
                return

            inp_t = inp_t.detach().float()
            out_t = out_t.detach().float()

            inp_flat = inp_t.flatten()
            out_flat = out_t.flatten()

            # Cosine similarity (guard against zero-norm)
            inp_norm = inp_flat.norm()
            out_norm = out_flat.norm()
            if inp_norm.item() > 0 and out_norm.item() > 0:
                # Align lengths — for Embedding, output may differ from input
                if inp_flat.shape[0] != out_flat.shape[0]:
                    cos_sim = 0.0
                else:
                    cos_sim = F.cosine_similarity(inp_flat.unsqueeze(0),
                                                  out_flat.unsqueeze(0)).item()
            else:
                cos_sim = 0.0

            nr = out_norm.item() / (inp_norm.item() + 1e-8)

            entry = {
                "input_rank": effective_rank(inp_t),
                "output_rank": effective_rank(out_t),
                "input_entropy": token_entropy(inp_t),
                "output_entropy": token_entropy(out_t),
                "cosine_similarity": cos_sim,
                "norm_ratio": nr,
            }
            accum.setdefault(name, []).append(entry)
        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            h = module.register_forward_hook(_make_hook(name))
            hooks.append(h)

    # Forward pass over calibration data
    with torch.no_grad():
        for sample in calibration_data:
            sample = sample.to(device)
            if sample.dim() == 1:
                sample = sample.unsqueeze(0)  # add batch dim
            try:
                model(sample)
            except Exception:
                # Some models need different input shapes — skip bad samples
                continue

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average accumulators into LayerStats
    stats: Dict[str, LayerStats] = {}
    for name, entries in accum.items():
        n = len(entries)
        if n == 0:
            continue
        avg = {k: sum(e[k] for e in entries) / n for k in entries[0]}
        rank_change = avg["output_rank"] - avg["input_rank"]
        entropy_change = avg["output_entropy"] - avg["input_entropy"]
        stats[name] = LayerStats(
            name=name,
            input_rank=avg["input_rank"],
            output_rank=avg["output_rank"],
            rank_change=rank_change,
            input_entropy=avg["input_entropy"],
            output_entropy=avg["output_entropy"],
            entropy_change=entropy_change,
            cosine_similarity=avg["cosine_similarity"],
            norm_ratio=avg["norm_ratio"],
        )

    return stats


# ---------------------------------------------------------------------------
# Destructive-layer detection
# ---------------------------------------------------------------------------

def find_destructive_layers(
    stats: Dict[str, LayerStats],
    rank_threshold: float = -0.3,
    norm_low: float = 0.1,
    norm_high: float = 10.0,
    entropy_threshold: float = -0.5,
) -> List[str]:
    """Identify layers that destroy representation quality.

    A layer is *destructive* if any of the following hold:
    - ``rank_change < rank_threshold * input_rank``  (kills too much rank)
    - ``norm_ratio`` outside ``[norm_low, norm_high]``
    - ``entropy_change < entropy_threshold``

    Returns:
        Sorted list of destructive layer names.
    """
    destructive = []
    for name, s in stats.items():
        is_bad = False
        # Rank collapse
        if s.input_rank > 0 and s.rank_change < rank_threshold * s.input_rank:
            is_bad = True
        # Norm explosion / collapse
        if s.norm_ratio < norm_low or s.norm_ratio > norm_high:
            is_bad = True
        # Entropy destruction
        if s.entropy_change < entropy_threshold:
            is_bad = True
        if is_bad:
            destructive.append(name)
    return sorted(destructive)


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _get_module(model: nn.Module, name: str) -> nn.Module:
    """Resolve a dotted name to a sub-module."""
    parts = name.split(".")
    mod = model
    for p in parts:
        mod = getattr(mod, p)
    return mod


def _get_layers_of_same_type(model: nn.Module, target_name: str):
    """Return ordered list of (name, module) pairs sharing the same type as *target_name*."""
    target_mod = _get_module(model, target_name)
    target_type = type(target_mod)
    return [(n, m) for n, m in model.named_modules() if type(m) is target_type]


def activation_repair(
    model: nn.Module,
    destructive_layers: List[str],
    strategy: str = "shrink",
    shrink_factor: float = 0.1,
) -> Tuple[dict, List[str]]:
    """Repair destructive layers in *model*.

    Strategies:

    ``"shrink"``
        Multiply all parameters of the layer by *shrink_factor* (soft bypass).

    ``"passthrough"``
        For ``nn.Linear``, set weight to identity-like and bias to zero.
        Falls back to ``shrink`` for non-Linear modules.

    ``"interpolate"``
        Average weights from neighbouring layers of the same type.
        Falls back to ``shrink`` for edge layers (first/last of their type)
        that have no neighbours on both sides.

    Returns:
        ``(state_dict, repaired_names)`` — the modified state dict and the
        list of layer names that were actually repaired.
    """
    repaired: List[str] = []

    with torch.no_grad():
        for name in destructive_layers:
            try:
                module = _get_module(model, name)
            except AttributeError:
                continue

            if strategy == "shrink":
                _apply_shrink(module, shrink_factor)
                repaired.append(name)

            elif strategy == "passthrough":
                if isinstance(module, nn.Linear):
                    _apply_passthrough_linear(module)
                    repaired.append(name)
                else:
                    # Fallback to shrink for non-Linear
                    _apply_shrink(module, shrink_factor)
                    repaired.append(name)

            elif strategy == "interpolate":
                success = _apply_interpolate(model, name, module)
                if not success:
                    # Edge layer with no valid neighbours — fall back
                    _apply_shrink(module, shrink_factor)
                repaired.append(name)

            else:
                raise ValueError(f"Unknown repair strategy: {strategy!r}")

    return model.state_dict(), repaired


def _apply_shrink(module: nn.Module, factor: float):
    """Scale all parameters of *module* by *factor*."""
    for p in module.parameters():
        p.mul_(factor)


def _apply_passthrough_linear(module: nn.Linear):
    """Set a Linear layer to approximate identity.

    If the layer is square, set weight = I. Otherwise, set weight to the
    top-left identity block and zero the rest.  Bias is zeroed.
    """
    w = module.weight  # (out_features, in_features)
    out_f, in_f = w.shape
    w.zero_()
    min_dim = min(out_f, in_f)
    for i in range(min_dim):
        w[i, i] = 1.0
    if module.bias is not None:
        module.bias.zero_()


def _apply_interpolate(model: nn.Module, target_name: str, target_module: nn.Module) -> bool:
    """Average weights from neighbouring layers of the same type.

    Returns True if interpolation was applied, False if no valid neighbours
    exist (caller should fall back).
    """
    same_type = _get_layers_of_same_type(model, target_name)
    names = [n for n, _ in same_type]

    if target_name not in names:
        return False

    idx = names.index(target_name)
    neighbours = []
    if idx > 0:
        neighbours.append(same_type[idx - 1][1])
    if idx < len(names) - 1:
        neighbours.append(same_type[idx + 1][1])

    if not neighbours:
        return False

    # We need at least one neighbour whose params have the same shape
    target_params = dict(target_module.named_parameters())
    matched = False

    for pname, tp in target_params.items():
        compatible = []
        for nb in neighbours:
            nb_params = dict(nb.named_parameters())
            if pname in nb_params and nb_params[pname].shape == tp.shape:
                compatible.append(nb_params[pname])
        if compatible:
            avg = torch.stack([c.data for c in compatible]).mean(dim=0)
            tp.data.copy_(avg)
            matched = True

    return matched
