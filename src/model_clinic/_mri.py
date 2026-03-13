"""Model MRI -- deep per-layer analysis using weight matrix decomposition.

Answers "What is each layer actually doing?" using static weight analysis only.
No forward pass required. Uses SVD to compute effective rank, entropy,
stable rank, and other information-theoretic metrics per weight matrix.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import math
import statistics

import torch


@dataclass
class LayerMRI:
    """Per-layer MRI analysis result."""
    name: str
    shape: tuple
    dtype: str

    # Rank analysis (via SVD on 2D projection)
    effective_rank: float        # sum(s)/max(s) normalized
    numerical_rank: int          # rank at 1e-6 tolerance
    rank_utilization: float      # numerical_rank / min(shape)
    top_sv_ratio: float          # s[0]/s[1] -- how dominant is top singular vector

    # Information content
    entropy: float               # normalized entropy of singular values
    stable_rank: float           # (||W||_F / ||W||_2)^2

    # Weight distribution
    kurtosis: float
    skewness: float
    sparsity: float              # fraction of near-zero values (< 1e-4)

    # Health flags
    is_low_rank: bool            # rank_utilization < 0.1
    is_degenerate: bool          # numerical_rank < 2
    is_heavy_tailed: bool        # kurtosis > threshold
    is_sparse: bool              # sparsity > 0.5

    # Role inference (heuristic)
    inferred_role: str           # "attention", "mlp_gate", "embedding", "norm", "projection", "unknown"


# Thresholds
_LOW_RANK_THRESHOLD = 0.1
_HEAVY_TAIL_KURTOSIS = 10.0
_SPARSE_THRESHOLD = 0.5
_NEAR_ZERO = 1e-4
_SVD_RANK_TOL = 1e-6
_LARGE_MATRIX_THRESHOLD = 1000


def _infer_role(name: str) -> str:
    """Heuristic role inference from parameter name."""
    nl = name.lower()
    if any(k in nl for k in ("q_proj", "k_proj", "v_proj", "o_proj",
                              "self_attn", "attention")):
        return "attention"
    if any(k in nl for k in ("gate_proj", "gate_up", "mlp.gate")):
        return "mlp_gate"
    if any(k in nl for k in ("embed", "wte", "wpe", "token")):
        return "embedding"
    if any(k in nl for k in ("norm", "layernorm", "rmsnorm", "ln_")):
        return "norm"
    if any(k in nl for k in ("down_proj", "up_proj", "fc1", "fc2",
                              "mlp", "dense", "lm_head", "head")):
        return "projection"
    return "unknown"


def _to_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape tensor to 2D for SVD. Merges leading dims if needed."""
    if tensor.dim() == 2:
        return tensor
    # Merge all but last dim
    return tensor.reshape(-1, tensor.shape[-1])


def _compute_svd(mat: torch.Tensor, use_lowrank: bool = False) -> torch.Tensor:
    """Compute singular values of a 2D matrix.

    Uses torch.svd_lowrank for large matrices for speed.
    Returns singular values tensor (sorted descending).
    """
    m, n = mat.shape
    k = min(m, n)

    if use_lowrank and k > _LARGE_MATRIX_THRESHOLD:
        # Use randomized SVD -- compute top-k singular values
        # q must be <= min(m, n); use at most 300 for speed
        q = min(k, 300)
        try:
            _, s, _ = torch.svd_lowrank(mat, q=q)
            return s
        except Exception:
            # Fallback to full SVD
            pass

    # Full SVD
    try:
        s = torch.linalg.svdvals(mat)
        return s
    except Exception:
        # Last resort: return zeros
        return torch.zeros(k)


def analyze_layer(name: str, tensor: torch.Tensor) -> LayerMRI:
    """Analyze a single weight tensor and return LayerMRI.

    Args:
        name: parameter name
        tensor: weight tensor (must be >= 2D)

    Returns:
        LayerMRI with all computed metrics.
    """
    shape = tuple(tensor.shape)
    dtype = str(tensor.dtype)
    t = tensor.detach().float()
    mat = _to_2d(t)
    m, n = mat.shape
    k = min(m, n)

    # Determine if we should use low-rank SVD
    use_lowrank = max(m, n) > _LARGE_MATRIX_THRESHOLD

    # SVD
    s = _compute_svd(mat, use_lowrank=use_lowrank)

    # Filter out near-zero singular values for ratio calculations
    s_max = s[0].item() if len(s) > 0 else 0.0

    # Effective rank: sum(s) / max(s)
    if s_max > 0:
        effective_rank = (s.sum() / s_max).item()
    else:
        effective_rank = 0.0

    # Numerical rank: count of singular values above tolerance
    if s_max > 0:
        tol = _SVD_RANK_TOL * s_max
        numerical_rank = int((s > tol).sum().item())
    else:
        numerical_rank = 0

    # Rank utilization
    rank_utilization = numerical_rank / k if k > 0 else 0.0

    # Top SV ratio: s[0]/s[1]
    if len(s) >= 2 and s[1].item() > 1e-12:
        top_sv_ratio = (s[0] / s[1]).item()
    else:
        top_sv_ratio = float("inf")

    # Entropy of singular values (normalized)
    if s_max > 0:
        # Normalize to probability distribution
        s_pos = s[s > 0]
        if len(s_pos) > 1:
            p = s_pos / s_pos.sum()
            raw_entropy = -(p * p.log()).sum().item()
            max_entropy = math.log(len(s_pos))
            entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            entropy = 0.0
    else:
        entropy = 0.0

    # Stable rank: (||W||_F / ||W||_2)^2
    fro_norm = mat.norm("fro").item()
    spectral_norm = s_max
    if spectral_norm > 0:
        stable_rank = (fro_norm / spectral_norm) ** 2
    else:
        stable_rank = 0.0

    # Weight distribution stats
    flat = t.reshape(-1)

    # Kurtosis: E[(x-mu)^4] / std^4 - 3
    mu = flat.mean()
    std = flat.std()
    if std.item() > 1e-12:
        centered = flat - mu
        kurtosis = (centered.pow(4).mean() / std.pow(4)).item() - 3.0
    else:
        kurtosis = 0.0

    # Skewness: E[(x-mu)^3] / std^3
    if std.item() > 1e-12:
        skewness = (centered.pow(3).mean() / std.pow(3)).item()
    else:
        skewness = 0.0

    # Sparsity
    sparsity = (flat.abs() < _NEAR_ZERO).float().mean().item()

    # Health flags
    is_low_rank = rank_utilization < _LOW_RANK_THRESHOLD
    is_degenerate = numerical_rank < 2
    is_heavy_tailed = kurtosis > _HEAVY_TAIL_KURTOSIS
    is_sparse = sparsity > _SPARSE_THRESHOLD

    # Role
    inferred_role = _infer_role(name)

    return LayerMRI(
        name=name,
        shape=shape,
        dtype=dtype,
        effective_rank=effective_rank,
        numerical_rank=numerical_rank,
        rank_utilization=rank_utilization,
        top_sv_ratio=top_sv_ratio,
        entropy=entropy,
        stable_rank=stable_rank,
        kurtosis=kurtosis,
        skewness=skewness,
        sparsity=sparsity,
        is_low_rank=is_low_rank,
        is_degenerate=is_degenerate,
        is_heavy_tailed=is_heavy_tailed,
        is_sparse=is_sparse,
        inferred_role=inferred_role,
    )


def model_mri(
    state_dict: dict,
    max_layers: int = None,
    verbose: bool = False,
) -> List[LayerMRI]:
    """Deep per-layer analysis using SVD decomposition.

    Returns list of LayerMRI, one per tensor with >= 2 dimensions.
    Sorted by name.

    Uses torch.svd_lowrank for large matrices (> 1000x1000) for speed.

    Args:
        state_dict: dict mapping parameter names to tensors.
        max_layers: if set, analyze at most this many layers.
        verbose: if True, print progress during analysis.

    Returns:
        List of LayerMRI objects, sorted by parameter name.
    """
    results = []
    candidates = sorted(
        [(k, v) for k, v in state_dict.items()
         if isinstance(v, torch.Tensor) and v.dim() >= 2],
        key=lambda x: x[0],
    )

    if max_layers is not None:
        candidates = candidates[:max_layers]

    total = len(candidates)
    for i, (name, tensor) in enumerate(candidates):
        if verbose:
            print(f"  [{i+1}/{total}] {name} {list(tensor.shape)}")
        results.append(analyze_layer(name, tensor))

    return results


def mri_summary(mri_results: List[LayerMRI]) -> dict:
    """Summarize MRI results across the model.

    Returns dict with:
    - total_layers, analyzed_layers
    - mean_rank_utilization, median_rank_utilization
    - n_low_rank, n_degenerate, n_heavy_tailed, n_sparse
    - role_distribution: {"attention": 12, "mlp_gate": 24, ...}
    - information_score: 0-100 (overall information utilization)
    """
    if not mri_results:
        return {
            "total_layers": 0,
            "analyzed_layers": 0,
            "mean_rank_utilization": 0.0,
            "median_rank_utilization": 0.0,
            "n_low_rank": 0,
            "n_degenerate": 0,
            "n_heavy_tailed": 0,
            "n_sparse": 0,
            "role_distribution": {},
            "information_score": 0,
        }

    n = len(mri_results)
    rank_utils = [r.rank_utilization for r in mri_results]

    role_dist = {}
    for r in mri_results:
        role_dist[r.inferred_role] = role_dist.get(r.inferred_role, 0) + 1

    # Information score: weighted combination of rank utilization and entropy
    # 0-100 scale
    mean_rank_util = statistics.mean(rank_utils)
    mean_entropy = statistics.mean([r.entropy for r in mri_results])
    # Penalize for degenerate and low-rank layers
    n_degenerate = sum(1 for r in mri_results if r.is_degenerate)
    n_low_rank = sum(1 for r in mri_results if r.is_low_rank)
    degen_penalty = (n_degenerate / n) * 30  # up to 30 pts off
    low_rank_penalty = (n_low_rank / n) * 20  # up to 20 pts off

    raw_score = (mean_rank_util * 50 + mean_entropy * 50)
    info_score = max(0, min(100, int(raw_score - degen_penalty - low_rank_penalty)))

    return {
        "total_layers": n,
        "analyzed_layers": n,
        "mean_rank_utilization": round(mean_rank_util, 4),
        "median_rank_utilization": round(statistics.median(rank_utils), 4),
        "n_low_rank": n_low_rank,
        "n_degenerate": n_degenerate,
        "n_heavy_tailed": sum(1 for r in mri_results if r.is_heavy_tailed),
        "n_sparse": sum(1 for r in mri_results if r.is_sparse),
        "role_distribution": role_dist,
        "information_score": info_score,
    }
