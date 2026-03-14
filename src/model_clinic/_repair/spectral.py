"""Spectral Surgery — SVD-based denoising for weight matrices.

Level 2 of the deep repair pipeline. Decomposes a weight matrix via SVD,
identifies noise-carrying singular values (those that inflate condition number
without carrying energy), and truncates them. This directly fixes gradient_noise
findings caused by high condition numbers.

Usage:
    from model_clinic._repair.spectral import spectral_denoise, spectral_analysis

    # Analyze without modifying
    info = spectral_analysis(tensor)

    # Denoise in place
    denoised = spectral_denoise(tensor)
"""

from dataclasses import dataclass

import torch


@dataclass
class SpectralReport:
    """Report from a spectral denoising operation."""
    param_name: str
    original_rank: int
    effective_rank: int
    energy_retained: float
    condition_before: float
    condition_after: float
    frobenius_error: float


def spectral_analysis(tensor):
    """Analyze the singular value spectrum of a tensor without modifying it.

    Args:
        tensor: A PyTorch tensor (must be 2D).

    Returns:
        dict with keys:
            singular_values: 1D tensor of singular values (descending)
            effective_rank: number of SVs needed for 99% energy
            condition_number: ratio of largest to smallest nonzero SV
            energy_distribution: cumulative energy fraction per SV
            total_energy: sum of squared singular values
            shape: original tensor shape
    """
    if tensor.dim() < 2:
        return {
            "singular_values": torch.tensor([]),
            "effective_rank": 0,
            "condition_number": 1.0,
            "energy_distribution": torch.tensor([]),
            "total_energy": 0.0,
            "shape": list(tensor.shape),
        }

    t = tensor.float()
    # For tensors with more than 2 dims, reshape to 2D
    original_shape = t.shape
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)

    sv = torch.linalg.svdvals(t)

    total_energy = (sv ** 2).sum().item()
    if total_energy < 1e-30:
        return {
            "singular_values": sv,
            "effective_rank": 0,
            "condition_number": float("inf"),
            "energy_distribution": torch.zeros_like(sv),
            "total_energy": 0.0,
            "shape": list(original_shape),
        }

    energy_dist = torch.cumsum(sv ** 2, dim=0) / total_energy

    # Effective rank: number of SVs for 99% energy
    effective_rank = (torch.searchsorted(energy_dist, torch.tensor(0.99)) + 1).item()
    effective_rank = min(effective_rank, len(sv))

    # Condition number
    sv_pos = sv[sv > 1e-10]
    if len(sv_pos) >= 2:
        condition_number = (sv_pos[0] / sv_pos[-1]).item()
    elif len(sv_pos) == 1:
        condition_number = 1.0
    else:
        condition_number = float("inf")

    return {
        "singular_values": sv,
        "effective_rank": effective_rank,
        "condition_number": condition_number,
        "energy_distribution": energy_dist,
        "total_energy": total_energy,
        "shape": list(original_shape),
    }


def spectral_denoise(tensor, energy_threshold=0.99, max_condition=1000,
                     min_rank_ratio=0.1):
    """SVD-based denoising. Truncate singular values that carry noise.

    Selects an effective rank by intersecting two criteria:
    1. Energy threshold — keep enough SVs to retain `energy_threshold` of total
       spectral energy (sum of squared singular values).
    2. Condition cap — drop SVs smaller than S[0] / max_condition.

    The more conservative (lower) rank wins, but we never go below
    `min_rank_ratio * min(m, n)` to prevent over-truncation.

    Args:
        tensor: 2D+ PyTorch tensor to denoise.
        energy_threshold: fraction of spectral energy to preserve (default 0.99).
        max_condition: maximum condition number after truncation (default 1000).
        min_rank_ratio: never truncate below this fraction of original rank (default 0.1).

    Returns:
        Denoised tensor (same shape and dtype as input), or the original tensor
        unchanged if it's 1D, too small, or already well-conditioned.
    """
    # Edge cases: skip non-2D and tiny tensors
    if tensor.dim() < 2:
        return tensor
    if tensor.shape[0] < 2 or tensor.shape[1] < 2:
        return tensor

    original_dtype = tensor.dtype
    t = tensor.float()

    # For higher-dim tensors, reshape to 2D
    original_shape = t.shape
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)

    m, n = t.shape
    min_dim = min(m, n)

    # Compute SVD
    try:
        U, S, Vt = torch.linalg.svd(t, full_matrices=False)
    except Exception:
        return tensor

    # Check if already well-conditioned — skip if so
    sv_pos = S[S > 1e-10]
    if len(sv_pos) < 2:
        return tensor
    condition_before = (sv_pos[0] / sv_pos[-1]).item()
    if condition_before <= max_condition:
        return tensor

    # Method 1: Energy-based rank selection
    total_energy = (S ** 2).sum()
    if total_energy < 1e-30:
        return tensor
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
    rank_energy = (torch.searchsorted(cumulative_energy, torch.tensor(energy_threshold)) + 1).item()
    rank_energy = min(rank_energy, len(S))

    # Method 2: Condition number cap
    threshold_sv = S[0] / max_condition
    rank_condition = (S > threshold_sv).sum().item()

    # Take the more conservative (lower) rank
    effective_rank = min(rank_energy, rank_condition)

    # Apply minimum rank floor
    min_rank = max(1, int(min_dim * min_rank_ratio))
    effective_rank = max(effective_rank, min_rank)

    # Don't exceed the number of available singular values
    effective_rank = min(effective_rank, len(S))

    # If we're keeping everything, no-op
    if effective_rank >= len(S):
        return tensor

    # Reconstruct with truncated SVD
    U_trunc = U[:, :effective_rank]
    S_trunc = S[:effective_rank].clone()
    Vt_trunc = Vt[:effective_rank, :]

    # Clamp small SVs so condition number stays within max_condition,
    # even when min_rank_ratio forces us to keep more components
    sv_floor = S_trunc[0] / max_condition
    S_trunc = torch.clamp(S_trunc, min=sv_floor.item())

    reconstructed = U_trunc @ torch.diag(S_trunc) @ Vt_trunc

    # Reshape back if needed
    if len(original_shape) > 2:
        reconstructed = reconstructed.reshape(original_shape)

    return reconstructed.to(original_dtype)


def spectral_denoise_with_report(tensor, param_name, energy_threshold=0.99,
                                 max_condition=1000, min_rank_ratio=0.1):
    """Like spectral_denoise but also returns a SpectralReport.

    Args:
        tensor: 2D+ PyTorch tensor.
        param_name: Name of the parameter (for the report).
        energy_threshold: fraction of spectral energy to preserve.
        max_condition: maximum condition number after truncation.
        min_rank_ratio: never truncate below this fraction of original rank.

    Returns:
        (denoised_tensor, SpectralReport)
    """
    if tensor.dim() < 2 or tensor.shape[0] < 2 or tensor.shape[1] < 2:
        return tensor, SpectralReport(
            param_name=param_name,
            original_rank=min(tensor.shape) if tensor.dim() >= 2 else 0,
            effective_rank=min(tensor.shape) if tensor.dim() >= 2 else 0,
            energy_retained=1.0,
            condition_before=1.0,
            condition_after=1.0,
            frobenius_error=0.0,
        )

    original_dtype = tensor.dtype
    t = tensor.float()

    original_shape = t.shape
    if t.dim() > 2:
        t = t.reshape(t.shape[0], -1)

    m, n = t.shape
    min_dim = min(m, n)

    try:
        U, S, Vt = torch.linalg.svd(t, full_matrices=False)
    except Exception:
        return tensor, SpectralReport(
            param_name=param_name,
            original_rank=min_dim,
            effective_rank=min_dim,
            energy_retained=1.0,
            condition_before=float("inf"),
            condition_after=float("inf"),
            frobenius_error=0.0,
        )

    # Condition before
    sv_pos = S[S > 1e-10]
    original_rank = len(sv_pos)
    if len(sv_pos) < 2:
        return tensor, SpectralReport(
            param_name=param_name,
            original_rank=original_rank,
            effective_rank=original_rank,
            energy_retained=1.0,
            condition_before=1.0,
            condition_after=1.0,
            frobenius_error=0.0,
        )

    condition_before = (sv_pos[0] / sv_pos[-1]).item()

    # If already well-conditioned, return unchanged
    if condition_before <= max_condition:
        return tensor, SpectralReport(
            param_name=param_name,
            original_rank=original_rank,
            effective_rank=original_rank,
            energy_retained=1.0,
            condition_before=condition_before,
            condition_after=condition_before,
            frobenius_error=0.0,
        )

    total_energy = (S ** 2).sum()
    if total_energy < 1e-30:
        return tensor, SpectralReport(
            param_name=param_name,
            original_rank=original_rank,
            effective_rank=0,
            energy_retained=0.0,
            condition_before=condition_before,
            condition_after=condition_before,
            frobenius_error=0.0,
        )

    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy

    rank_energy = (torch.searchsorted(cumulative_energy, torch.tensor(energy_threshold)) + 1).item()
    rank_energy = min(rank_energy, len(S))

    threshold_sv = S[0] / max_condition
    rank_condition = (S > threshold_sv).sum().item()

    effective_rank = min(rank_energy, rank_condition)
    min_rank = max(1, int(min_dim * min_rank_ratio))
    effective_rank = max(effective_rank, min_rank)
    effective_rank = min(effective_rank, len(S))

    if effective_rank >= len(S):
        return tensor, SpectralReport(
            param_name=param_name,
            original_rank=original_rank,
            effective_rank=original_rank,
            energy_retained=1.0,
            condition_before=condition_before,
            condition_after=condition_before,
            frobenius_error=0.0,
        )

    # Reconstruct
    U_trunc = U[:, :effective_rank]
    S_trunc = S[:effective_rank].clone()
    Vt_trunc = Vt[:effective_rank, :]

    # Clamp small SVs so condition number stays within max_condition,
    # even when min_rank_ratio forces us to keep more components
    sv_floor = S_trunc[0] / max_condition
    S_trunc = torch.clamp(S_trunc, min=sv_floor.item())

    reconstructed = U_trunc @ torch.diag(S_trunc) @ Vt_trunc

    # Metrics
    energy_retained = (cumulative_energy[effective_rank - 1]).item()
    condition_after = (S_trunc[0] / S_trunc[-1]).item() if S_trunc[-1] > 1e-10 else float("inf")
    frobenius_error = (t - reconstructed).norm().item() / t.norm().item()

    if len(original_shape) > 2:
        reconstructed = reconstructed.reshape(original_shape)

    report = SpectralReport(
        param_name=param_name,
        original_rank=original_rank,
        effective_rank=effective_rank,
        energy_retained=energy_retained,
        condition_before=condition_before,
        condition_after=condition_after,
        frobenius_error=frobenius_error,
    )

    return reconstructed.to(original_dtype), report
