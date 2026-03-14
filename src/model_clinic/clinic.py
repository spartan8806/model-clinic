"""Model Clinic — Automated diagnosis and treatment.

Runs health checks, maps findings to known prescriptions, applies fixes
one at a time with before/after testing, rolls back if things get worse.

health.py is the blood test. clinic.py is the doctor.

Usage:
    model-clinic exam checkpoint.pt
    model-clinic treat checkpoint.pt --save treated.pt
    model-clinic treat checkpoint.pt --test --save treated.pt
    model-clinic treat checkpoint.pt --conservative --save treated.pt
    model-clinic treat checkpoint.pt --runtime --save treated.pt
    model-clinic exam Qwen/Qwen2.5-0.5B-Instruct --hf
"""

import argparse
import json
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

from model_clinic._types import (
    Finding, Prescription, TreatmentResult, ExamReport,
    ExamResult, PipelineResult,
)
from model_clinic._loader import load_state_dict, load_model, build_meta, save_state_dict
from model_clinic._eval import (
    eval_coherence, eval_perplexity, eval_logit_entropy, eval_diversity,
    DEFAULT_PROMPTS, EXAMPLE_RUNTIME_PROMPTS,
)
from model_clinic._utils import safe_str, to_float
from model_clinic._health_score import compute_health_score, print_health_score
from model_clinic._references import format_references


# ── Thresholds & constants ────────────────────────────────────────────────

DEAD_NEURON_THRESHOLD = 1e-7        # Norm below this = dead
GATE_CLOSED_THRESHOLD = 0.01       # sigmoid below this = stuck closed
GATE_OPEN_THRESHOLD = 0.99         # sigmoid above this = stuck open
EXPLODING_NORM_THRESHOLD = 10.0    # Per-element norm above this = exploding
VANISHING_NORM_THRESHOLD = 1e-6    # Per-element norm below this = vanishing
KURTOSIS_THRESHOLD = 50            # Kurtosis above this = heavy tails
NORM_DRIFT_THRESHOLD = 1.5         # |mean - 1.0| above this = drifted (pretraining shifts norms naturally)
SATURATION_THRESHOLD = 0.3         # Fraction near max above this = saturated
SIMILARITY_THRESHOLD = 0.999       # Cosine sim above this = duplicate rows
MAX_TENSOR_ELEMENTS = 50_000_000   # Skip tensors larger than this for expensive ops
MAX_TENSOR_PAIRWISE = 10_000_000   # Skip tensors larger than this for pairwise ops
KURTOSIS_SAMPLE_SIZE = 1_000_000   # Sample size for kurtosis estimation
ROW_SAMPLE_SIZE = 200              # Max rows to sample for duplicate detection
CLAMP_SIGMA = 4.0                  # Default sigma for outlier clamping


# ── Metadata filter ───────────────────────────────────────────────────────

# Growth/tracking tensor keywords — these are counters, indices, and EMA
# accumulators from neural growth systems, NOT model weights.
_METADATA_KEYWORDS = frozenset([
    "neuron_age", "total_steps", "loss_idx", "gradient_sq_ema",
    "original_size", "growth_mask", "growth_score", "neuron_mask",
    "step_count", "update_count", "activation_count",
    "last_growth_step", "neurons_born", "gradient_ema",
    "activation_ema", "usefulness_score",
])


def _is_metadata_tensor(name, tensor):
    """Return True if this tensor is a growth/tracking metadata param, not a model weight."""
    name_lower = name.lower()
    # Check for known metadata keywords in any segment of the param name
    for kw in _METADATA_KEYWORDS:
        if kw in name_lower:
            return True
    return False


# ── Condition Registry ─────────────────────────────────────────────────────

class ConditionRegistry:
    """Registry of diagnosable conditions and their prescriptions."""

    def __init__(self):
        self._detectors = {}     # condition_name -> detector_fn(name, tensor, ctx) -> list[Finding]
        self._prescribers = {}   # condition_name -> (risk, description, prescriber_fn(finding) -> Prescription)

    def register(self, condition, detector, prescriber=None, risk="medium", description=""):
        """Register a condition detector and optional prescriber."""
        self._detectors[condition] = detector
        if prescriber:
            self._prescribers[condition] = {
                "risk": risk,
                "description": description,
                "make_rx": prescriber,
            }

    def detect_all(self, state_dict, context=None, verbose=False):
        """Run all detectors on a state dict."""
        context = context or {}
        findings = []
        # Pre-populate all tensor names so detectors can inspect the full key set
        context.setdefault("_all_tensor_names", set(state_dict.keys()))
        # Deduplicate detectors (same function may be registered under multiple names)
        seen_detectors = set()
        for condition, detector in self._detectors.items():
            detector_id = id(detector)
            if detector_id in seen_detectors:
                continue
            seen_detectors.add(detector_id)
            if verbose:
                print(f"  Checking {condition}...")
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if _is_metadata_tensor(name, tensor):
                    continue
                results = detector(name, tensor, context)
                if results:
                    findings.extend(results)
        return findings

    def prescribe(self, findings, conservative=False):
        """Map findings to prescriptions."""
        prescriptions = []
        for f in findings:
            if f.condition not in self._prescribers:
                continue
            rule = self._prescribers[f.condition]
            if conservative and rule["risk"] != "low":
                continue
            rx = rule["make_rx"](f)
            if rx:
                prescriptions.append(rx)
        return prescriptions


# ── Global registry ────────────────────────────────────────────────────────

REGISTRY = ConditionRegistry()


# ── Detectors ──────────────────────────────────────────────────────────────

def detect_dead_neurons(name, tensor, ctx):
    """Dead rows/columns in weight matrices."""
    if tensor.dim() < 2:
        return []
    if tensor.numel() > MAX_TENSOR_ELEMENTS:
        return []
    findings = []
    t = tensor.float()

    for dim_name, dim_idx, total_dim in [("rows", 1, 0), ("cols", 0, 1)]:
        norms = t.norm(dim=dim_idx)
        dead = (norms < DEAD_NEURON_THRESHOLD).nonzero(as_tuple=True)[0]
        if len(dead) > 0:
            pct = len(dead) / t.shape[total_dim]
            sev = "ERROR" if pct > 0.1 else "WARN" if pct > 0.01 else "INFO"
            findings.append(Finding(
                condition="dead_neurons",
                severity=sev,
                param_name=name,
                details={
                    "dead_indices": dead.tolist()[:50],
                    "dead_count": len(dead),
                    "total": t.shape[total_dim],
                    "pct": pct,
                    "dim": dim_name,
                },
            ))
    return findings


def detect_stuck_gates(name, tensor, ctx):
    """Scalar gate params stuck near 0 or 1."""
    if tensor.dim() != 0 or "gate" not in name.lower():
        return []
    val = tensor.float().item()
    sig = torch.sigmoid(tensor.float()).item()
    if sig < GATE_CLOSED_THRESHOLD:
        return [Finding("stuck_gate_closed", "WARN", name,
                        {"raw": val, "sigmoid": sig})]
    if sig > GATE_OPEN_THRESHOLD:
        return [Finding("stuck_gate_open", "WARN", name,
                        {"raw": val, "sigmoid": sig})]
    return []


def _per_elem_norm(tensor):
    """Compute per-element norm without allocating a copy."""
    if not tensor.is_floating_point() and not tensor.is_complex():
        return tensor.float().norm().item() / (tensor.numel() ** 0.5)
    return tensor.norm().item() / (tensor.numel() ** 0.5)


def detect_exploding_norm(name, tensor, ctx):
    """Per-element norm too high."""
    if tensor.numel() <= 10:
        return []
    per_elem = _per_elem_norm(tensor)
    if per_elem > EXPLODING_NORM_THRESHOLD:
        return [Finding("exploding_norm", "WARN", name,
                        {"per_elem_norm": per_elem, "shape": list(tensor.shape)})]
    return []


def detect_vanishing_norm(name, tensor, ctx):
    """Near-zero parameters (not bias/gate)."""
    if tensor.numel() <= 100 or "bias" in name or "gate" in name:
        return []
    per_elem = _per_elem_norm(tensor)
    if per_elem < VANISHING_NORM_THRESHOLD:
        return [Finding("vanishing_norm", "WARN", name,
                        {"per_elem_norm": per_elem, "shape": list(tensor.shape)})]
    return []


def detect_heavy_tails(name, tensor, ctx):
    """Extreme kurtosis in weight distributions."""
    if tensor.numel() < 100:
        return []
    # For large tensors, sample to avoid OOM
    if tensor.numel() > KURTOSIS_SAMPLE_SIZE:
        indices = torch.randint(0, tensor.numel(), (KURTOSIS_SAMPLE_SIZE,))
        t = tensor.flatten()[indices].float()
    else:
        t = tensor.flatten().float()
    std = t.std().item()
    if std < 1e-8:
        del t
        return []
    z = (t - t.mean()) / std
    kurtosis = z.pow(4).mean().item()
    del t, z
    if kurtosis > KURTOSIS_THRESHOLD:
        return [Finding("heavy_tails", "WARN", name,
                        {"kurtosis": kurtosis, "std": std})]
    return []


def detect_norm_drift(name, tensor, ctx):
    """LayerNorm/RMSNorm weights drifted from 1.0."""
    if tensor.dim() != 1:
        return []
    if not any(kw in name.lower() for kw in ["norm.weight", "layernorm", "rmsnorm"]):
        return []
    mean_val = tensor.float().mean().item()
    if abs(mean_val - 1.0) > NORM_DRIFT_THRESHOLD:
        return [Finding("norm_drift", "WARN", name,
                        {"mean": mean_val, "expected": 1.0})]
    return []


def detect_saturated_weights(name, tensor, ctx):
    """Too many values pinned at extremes."""
    if tensor.dim() < 2 or tensor.numel() < 100 or tensor.numel() > MAX_TENSOR_ELEMENTS:
        return []
    t = tensor.float()
    abs_max = t.abs().max().item()
    if abs_max <= 0.1:
        return []
    near_max = (t.abs() > 0.95 * abs_max).float().mean().item()
    if near_max > SATURATION_THRESHOLD:
        return [Finding("saturated_weights", "WARN", name,
                        {"near_max_pct": near_max, "abs_max": abs_max})]
    return []


def detect_nan_inf(name, tensor, ctx):
    """NaN or Inf in any parameter."""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        return [Finding("nan_inf", "ERROR", name,
                        {"nan_count": nan_count, "inf_count": inf_count,
                         "total": tensor.numel()})]
    return []


def detect_dtype_mismatch(name, tensor, ctx):
    """Track dtypes — finding generated at end from context."""
    dtypes = ctx.setdefault("_dtypes", defaultdict(list))
    dtypes[str(tensor.dtype)].append(name)
    return []


def detect_identical_rows(name, tensor, ctx):
    """Duplicate rows in weight matrices (broken symmetry)."""
    if tensor.dim() != 2 or tensor.shape[0] < 4:
        return []
    # Skip very large tensors (>10M elements) to avoid OOM
    if tensor.numel() > MAX_TENSOR_PAIRWISE:
        return []
    # Sample rows, cast only the sample to float
    n = min(tensor.shape[0], ROW_SAMPLE_SIZE)
    sample = tensor[:n].float()
    norms = sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = sample / norms
    sims = normed @ normed.T
    # Zero the diagonal and lower triangle, count upper-triangle pairs > threshold
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    dup_pairs = int((sims[triu_mask] > SIMILARITY_THRESHOLD).sum().item())
    max_sim = sims[triu_mask].max().item() if triu_mask.any() else 0
    del sample, normed, sims, triu_mask
    if dup_pairs > 0:
        return [Finding("identical_rows", "WARN", name,
                        {"max_similarity": max_sim, "duplicate_pairs": dup_pairs})]
    return []


def detect_attention_imbalance(name, tensor, ctx):
    """Q/K/V projection norm imbalance within a layer."""
    attn_norms = ctx.setdefault("_attn_norms", defaultdict(dict))
    for proj in ["q_proj", "k_proj", "v_proj"]:
        if proj in name and tensor.dim() >= 2:
            # Extract layer identifier
            layer_key = name.replace(f".{proj}.weight", "").replace(f".{proj}.bias", "")
            attn_norms[layer_key][proj] = tensor.float().norm().item()
    return []


def detect_weight_corruption(name, tensor, ctx):
    """Detect truncated or corrupted tensors."""
    if tensor.numel() <= 100:
        return []
    # Skip bias and norm tensors — they can legitimately be uniform
    name_lower = name.lower()
    if any(kw in name_lower for kw in ["bias", "norm.weight", "layernorm", "rmsnorm"]):
        return []
    findings = []
    t = tensor.float()

    # All-zero weight matrices
    if tensor.dim() >= 2 and t.abs().max().item() == 0:
        findings.append(Finding(
            "weight_corruption", "ERROR", name,
            {"reason": "all_zeros", "shape": list(tensor.shape)},
        ))
        return findings

    # All-same-value tensors (constant)
    if t.min().item() == t.max().item():
        findings.append(Finding(
            "weight_corruption", "WARN", name,
            {"reason": "constant_value", "value": t.min().item(),
             "shape": list(tensor.shape)},
        ))
        return findings

    # >50% identical values (quantization artifact or corruption)
    if tensor.numel() <= MAX_TENSOR_ELEMENTS:
        flat = t.flatten()
        # Use mode to find the most common value
        mode_val = flat.mode().values.item()
        same_count = (flat == mode_val).sum().item()
        frac = same_count / flat.numel()
        if frac > 0.5:
            findings.append(Finding(
                "weight_corruption", "WARN", name,
                {"reason": "majority_same_value", "value": mode_val,
                 "fraction": frac, "shape": list(tensor.shape)},
            ))

    return findings


def detect_head_redundancy(name, tensor, ctx):
    """Collect Q projection weights for post-scan head redundancy check."""
    if "q_proj" not in name or tensor.dim() < 2:
        return []
    q_weights = ctx.setdefault("_q_proj_weights", {})
    # Extract layer key
    layer_key = name.replace(".q_proj.weight", "").replace(".q_proj.bias", "")
    q_weights[layer_key] = tensor.detach().float()
    return []


def detect_positional_issues(name, tensor, ctx):
    """Detect broken positional encodings."""
    name_lower = name.lower()
    if not any(kw in name_lower for kw in ["position", "rotary", "rope"]):
        return []
    findings = []
    t = tensor.float()

    # Check for NaN/Inf
    if torch.isnan(t).any() or torch.isinf(t).any():
        findings.append(Finding(
            "positional_encoding_issues", "ERROR", name,
            {"reason": "nan_or_inf",
             "nan_count": int(torch.isnan(t).sum().item()),
             "inf_count": int(torch.isinf(t).sum().item())},
        ))
        return findings

    # All zeros (broken)
    if t.abs().max().item() == 0:
        findings.append(Finding(
            "positional_encoding_issues", "ERROR", name,
            {"reason": "all_zeros", "shape": list(tensor.shape)},
        ))
        return findings

    # All same value
    if t.min().item() == t.max().item():
        findings.append(Finding(
            "positional_encoding_issues", "WARN", name,
            {"reason": "constant_value", "value": t.min().item()},
        ))
        return findings

    # If 2D, rows should be distinct (each position should be different)
    if tensor.dim() == 2 and tensor.shape[0] >= 4:
        n = min(tensor.shape[0], ROW_SAMPLE_SIZE)
        sample = t[:n]
        norms = sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = sample / norms
        sims = normed @ normed.T
        triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        high_sim = int((sims[triu_mask] > SIMILARITY_THRESHOLD).sum().item())
        if high_sim > 0:
            findings.append(Finding(
                "positional_encoding_issues", "WARN", name,
                {"reason": "duplicate_positions", "duplicate_pairs": high_sim,
                 "shape": list(tensor.shape)},
            ))

    return findings


def detect_token_collapse(name, tensor, ctx):
    """Check if lm_head/output projection has near-identical rows (tokens that always get same score)."""
    if tensor.dim() != 2:
        return []
    name_lower = name.lower()
    if "lm_head" not in name_lower and "output" not in name_lower:
        return []
    n_rows = tensor.shape[0]
    if n_rows < 4:
        return []
    # Sample up to 500 rows
    sample_n = min(n_rows, 500)
    if sample_n < n_rows:
        indices = torch.randperm(n_rows)[:sample_n]
        sample = tensor[indices].float()
    else:
        sample = tensor.float()
    # Compute pairwise cosine similarity
    norms = sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normed = sample / norms
    sims = normed @ normed.T
    triu_mask = torch.triu(torch.ones(sample_n, sample_n, dtype=torch.bool), diagonal=1)
    high_sim_count = int((sims[triu_mask] > 0.99).sum().item())
    total_pairs = int(triu_mask.sum().item())
    del sample, normed, sims, triu_mask
    if total_pairs > 0 and high_sim_count / total_pairs > 0.10:
        return [Finding(
            "token_collapse", "WARN", name,
            {"collapsed_pair_fraction": high_sim_count / total_pairs,
             "collapsed_pairs": high_sim_count,
             "total_pairs": total_pairs,
             "sampled_rows": sample_n},
        )]
    return []


def detect_gradient_noise(name, tensor, ctx):
    """Estimate gradient noise from weight distribution characteristics.

    High condition number (max/min singular value ratio) predicts gradient
    instability during further training.
    """
    if tensor.dim() != 2 or tensor.numel() <= 1000:
        return []
    # Embedding/token tables have inherently high condition numbers (rare tokens
    # have near-zero vectors) — this is expected, not a defect.
    name_lower = name.lower()
    if any(kw in name_lower for kw in ["embed_tokens", "token_embedding", "wte", "word_embedding"]):
        return []
    t = tensor.float()
    m, n = t.shape
    # Sample a submatrix if large
    max_dim = 256
    if m > max_dim:
        row_idx = torch.randperm(m)[:max_dim]
        t = t[row_idx]
    if t.shape[1] > max_dim:
        col_idx = torch.randperm(n)[:max_dim]
        t = t[:, col_idx]
    try:
        sv = torch.linalg.svdvals(t)
        sv_pos = sv[sv > 1e-10]
        if len(sv_pos) < 2:
            return []
        condition_number = (sv_pos[0] / sv_pos[-1]).item()
        # Condition numbers up to ~50K are common in healthy pretrained models.
        # Only flag at 50K+ (WARN) or 1M+ (ERROR) for real instability.
        if condition_number > 50_000:
            severity = "WARN" if condition_number < 1_000_000 else "ERROR"
            return [Finding(
                "gradient_noise", severity, name,
                {"condition_number": condition_number,
                 "max_sv": sv_pos[0].item(),
                 "min_sv": sv_pos[-1].item(),
                 "shape": list(tensor.shape)},
            )]
    except Exception:
        return []
    return []


def _collect_layer_norms(name, tensor, ctx):
    """Collector: record per-element norm for each weight tensor, keyed by layer."""
    import re
    if tensor.dim() < 2:
        return []
    # Extract a layer index from the name (e.g. "layers.3.mlp.weight" -> 3)
    match = re.search(r'layers?[._](\d+)', name)
    if not match:
        return []
    layer_idx = int(match.group(1))
    layer_norms = ctx.setdefault("_layer_weight_norms", {})
    norms_list = layer_norms.setdefault(layer_idx, [])
    per_elem = tensor.float().norm().item() / (tensor.numel() ** 0.5)
    norms_list.append(per_elem)
    return []


def detect_moe_router_collapse(name, tensor, ctx):
    """Detect router/gate collapse in MoE models."""
    name_lower = name.lower()
    if tensor.dim() != 2:
        return []
    # Must contain "router" or "gate" — but exclude common non-MoE gates
    # like gate_proj (Qwen/LLaMA MLP gating), gate_up_proj, etc.
    is_router = "router" in name_lower
    is_gate = "gate" in name_lower and not any(
        x in name_lower for x in ["gate_proj", "gate_up", "gate_down"]
    )
    if not is_router and not is_gate:
        return []
    # Must be a routing matrix (2D), not a scalar gate
    if tensor.shape[0] < 2 or tensor.shape[1] < 2:
        return []
    t = tensor.float()
    # Compute softmax per row to get routing probabilities
    probs = F.softmax(t, dim=-1)
    # Compute per-row entropy
    log_probs = torch.log(probs.clamp(min=1e-10))
    entropy = -(probs * log_probs).sum(dim=-1)
    avg_entropy = entropy.mean().item()
    # Maximum possible entropy for uniform distribution
    max_entropy = torch.log(torch.tensor(float(t.shape[-1]))).item()
    # Normalized entropy (0 = collapsed, 1 = uniform)
    norm_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0
    findings = []
    if norm_entropy < 0.3:
        findings.append(Finding(
            "moe_router_collapse", "WARN", name,
            {"avg_entropy": avg_entropy, "max_entropy": max_entropy,
             "normalized_entropy": norm_entropy, "reason": "collapsed",
             "shape": list(tensor.shape)},
        ))
    elif norm_entropy > 0.95:
        findings.append(Finding(
            "moe_router_collapse", "INFO", name,
            {"avg_entropy": avg_entropy, "max_entropy": max_entropy,
             "normalized_entropy": norm_entropy, "reason": "near_uniform",
             "shape": list(tensor.shape)},
        ))
    return findings


def detect_lora_merge_artifacts(name, tensor, ctx):
    """Detect artifacts from LoRA weight merging.

    After LoRA merge, effective rank may be much lower than matrix dimensions,
    indicating the merge dominated the base weights.

    Only runs if the checkpoint shows evidence of LoRA (adapter keys present),
    since pretrained attention matrices are inherently low-rank.
    """
    if tensor.dim() != 2:
        return []
    # Skip check if model has no LoRA evidence — pretrained base models have
    # naturally low-rank attention matrices; that is NOT a defect.
    has_lora = ctx.get("_has_lora_keys")
    if has_lora is None:
        # First time: scan all keys collected so far
        all_keys = ctx.get("_all_tensor_names", set())
        has_lora = any(
            any(kw in k.lower() for kw in ["lora_a", "lora_b", "adapter", "lora_up", "lora_down"])
            for k in all_keys
        )
        ctx["_has_lora_keys"] = has_lora
    if not has_lora:
        return []
    name_lower = name.lower()
    if not any(kw in name_lower for kw in ["q_proj", "k_proj", "v_proj", "o_proj"]):
        return []
    m, n = tensor.shape
    if min(m, n) < 4:
        return []
    t = tensor.float()
    # Sample a 256x256 submatrix for SVD
    max_dim = 256
    if m > max_dim:
        row_idx = torch.randperm(m)[:max_dim]
        t = t[row_idx]
    if t.shape[1] > max_dim:
        col_idx = torch.randperm(t.shape[1])[:max_dim]
        t = t[:, col_idx]
    try:
        sv = torch.linalg.svdvals(t)
        sv_pos = sv[sv > 1e-10]
        if len(sv_pos) < 2:
            return []
        # Effective rank: nuclear norm / spectral norm
        nuclear_norm = sv_pos.sum().item()
        spectral_norm = sv_pos[0].item()
        effective_rank = nuclear_norm / spectral_norm if spectral_norm > 0 else 0
        min_dim = min(t.shape[0], t.shape[1])
        rank_ratio = effective_rank / min_dim if min_dim > 0 else 0
        if rank_ratio < 0.1:
            return [Finding(
                "lora_merge_artifacts", "WARN", name,
                {"effective_rank": effective_rank,
                 "matrix_min_dim": min_dim,
                 "rank_ratio": rank_ratio,
                 "top_sv": sv_pos[:5].tolist(),
                 "shape": list(tensor.shape)},
            )]
    except Exception:
        return []
    return []


def detect_quantization_degradation(name, tensor, ctx):
    """Detect quality loss from quantization (GPTQ, AWQ, INT8, FP8).

    Signs of quantization degradation include very low unique-value counts
    relative to tensor size, uniform spacing between values (grid pattern),
    and large blocks of identical values.

    Note: bf16 has ~65K representable values, so large bf16 tensors naturally
    have low unique ratios. We only flag when unique count is extremely low
    AND the spacing is grid-like (uniform), which indicates actual quantization.
    """
    if tensor.dim() < 2 or tensor.numel() < 1000:
        return []
    # Skip bf16/fp16 tensors for ratio-only checks — they have inherently
    # limited precision. Only flag if grid-quantized pattern detected.
    is_low_precision = tensor.dtype in (torch.bfloat16, torch.float16)
    t = tensor.float().flatten()
    # Sample up to 100K elements
    sample_size = min(t.numel(), 100_000)
    if sample_size < t.numel():
        indices = torch.randperm(t.numel())[:sample_size]
        sample = t[indices]
    else:
        sample = t
    unique_vals = torch.unique(sample)
    n_unique = len(unique_vals)
    unique_ratio = n_unique / sample_size
    findings = []

    # For bf16/fp16: only flag if unique count is extremely low (< 256 unique
    # values = likely INT8 or aggressive quantization) AND grid pattern detected
    # For fp32: use the original thresholds
    if is_low_precision:
        threshold = 256  # absolute count, not ratio
        if n_unique >= threshold:
            return []
    elif unique_ratio >= 0.05:
        return []

    if not is_low_precision and unique_ratio < 0.05:
        severity = "WARN" if unique_ratio < 0.01 else "INFO"
    elif is_low_precision:
        severity = "WARN" if n_unique < 64 else "INFO"
    else:
        severity = "INFO"

    details = {
        "unique_values": n_unique,
        "sample_size": sample_size,
        "unique_ratio": unique_ratio,
        "shape": list(tensor.shape),
    }
    # Check for grid-quantized values (uniform spacing)
    if n_unique >= 2:
        sorted_unique = unique_vals.sort().values
        spacings = sorted_unique[1:] - sorted_unique[:-1]
        if spacings.numel() > 1:
            mean_spacing = spacings.mean().item()
            std_spacing = spacings.std().item()
            if mean_spacing > 0:
                cv = std_spacing / mean_spacing
                details["spacing_cv"] = cv
                details["grid_quantized"] = cv < 0.1
    findings.append(Finding(
        "quantization_degradation", severity, name, details,
    ))
    return findings


def _collect_model_aging(name, tensor, ctx):
    """Collector: gather embedding and output weights for model aging detection."""
    if tensor.dim() != 2:
        return []
    name_lower = name.lower()
    if "embed" in name_lower:
        embeds = ctx.setdefault("_aging_embed_weights", {})
        embeds[name] = tensor.detach().float()
    elif "lm_head" in name_lower or "output" in name_lower:
        outputs = ctx.setdefault("_aging_output_weights", {})
        outputs[name] = tensor.detach().float()
    return []


def post_detect_model_aging(ctx):
    """Detect signs of catastrophic forgetting or model aging.

    Checks for:
    - Collapsed embedding representations (low effective rank)
    - Inverted layer norm gradient (early layers >> later layers)
    - Near-identical rows in output projection (token merging from forgetting)
    """
    findings = []

    # Check embedding effective rank
    embed_weights = ctx.get("_aging_embed_weights", {})
    for ename, etensor in embed_weights.items():
        m, n = etensor.shape
        if min(m, n) < 4:
            continue
        try:
            # Sample submatrix for SVD efficiency
            max_dim = 256
            t = etensor
            if m > max_dim:
                t = t[torch.randperm(m)[:max_dim]]
            if t.shape[1] > max_dim:
                t = t[:, torch.randperm(t.shape[1])[:max_dim]]
            sv = torch.linalg.svdvals(t)
            sv_pos = sv[sv > 1e-10]
            if len(sv_pos) < 2:
                continue
            nuclear_norm = sv_pos.sum().item()
            spectral_norm = sv_pos[0].item()
            effective_rank = nuclear_norm / spectral_norm if spectral_norm > 0 else 0
            min_dim = min(t.shape[0], t.shape[1])
            rank_ratio = effective_rank / min_dim if min_dim > 0 else 0
            if rank_ratio < 0.05:
                findings.append(Finding(
                    "model_aging", "WARN", ename,
                    {"reason": "collapsed_embeddings",
                     "effective_rank": effective_rank,
                     "matrix_min_dim": min_dim,
                     "rank_ratio": rank_ratio},
                ))
        except Exception:
            continue

    # Check inverted layer norm gradient (early >> later)
    layer_norms = ctx.get("_layer_weight_norms", {})
    if len(layer_norms) >= 4:
        sorted_layers = sorted(layer_norms.keys())
        n_layers = len(sorted_layers)
        quarter = max(1, n_layers // 4)
        early_layers = sorted_layers[:quarter]
        late_layers = sorted_layers[-quarter:]
        early_mean = sum(
            sum(layer_norms[l]) / len(layer_norms[l])
            for l in early_layers
        ) / len(early_layers)
        late_mean = sum(
            sum(layer_norms[l]) / len(layer_norms[l])
            for l in late_layers
        ) / len(late_layers)
        if late_mean > 1e-10:
            norm_ratio = early_mean / late_mean
            if norm_ratio > 5.0:
                findings.append(Finding(
                    "model_aging", "WARN", f"layers.{early_layers[0]}->{late_layers[-1]}",
                    {"reason": "inverted_norm_gradient",
                     "early_mean_norm": early_mean,
                     "late_mean_norm": late_mean,
                     "ratio": norm_ratio},
                ))

    # Check output projection for near-identical rows (token merging)
    output_weights = ctx.get("_aging_output_weights", {})
    for oname, otensor in output_weights.items():
        n_rows = otensor.shape[0]
        if n_rows < 4:
            continue
        sample_n = min(n_rows, 500)
        if sample_n < n_rows:
            indices = torch.randperm(n_rows)[:sample_n]
            sample = otensor[indices]
        else:
            sample = otensor
        norms = sample.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = sample / norms
        sims = normed @ normed.T
        triu_mask = torch.triu(
            torch.ones(sample_n, sample_n, dtype=torch.bool), diagonal=1
        )
        high_sim_count = int((sims[triu_mask] > 0.99).sum().item())
        total_pairs = int(triu_mask.sum().item())
        del sample, normed, sims, triu_mask
        if total_pairs > 0:
            merged_fraction = high_sim_count / total_pairs
            if merged_fraction > 0.05:
                findings.append(Finding(
                    "model_aging", "WARN", oname,
                    {"reason": "token_merging",
                     "merged_pair_fraction": merged_fraction,
                     "merged_pairs": high_sim_count,
                     "sampled_rows": sample_n},
                ))

    return findings


# ── Post-scan detectors (run after all params scanned) ─────────────────────

def post_detect_dtype_mismatch(ctx):
    """Check for mixed dtypes across parameters."""
    dtypes = ctx.get("_dtypes", {})
    if len(dtypes) <= 1:
        return []
    # Find minority dtype
    sorted_dtypes = sorted(dtypes.items(), key=lambda x: -len(x[1]))
    majority = sorted_dtypes[0][0]
    findings = []
    for dtype, params in sorted_dtypes[1:]:
        if len(params) > 0:
            findings.append(Finding(
                "dtype_mismatch", "WARN", f"mixed({dtype})",
                {"minority_dtype": dtype, "majority_dtype": majority,
                 "minority_count": len(params), "examples": params[:5]}
            ))
    return findings


def post_detect_attention_imbalance(ctx):
    """Check Q/K/V norm ratios per layer."""
    attn_norms = ctx.get("_attn_norms", {})
    findings = []
    for layer_key, norms in attn_norms.items():
        if len(norms) < 3:
            continue
        vals = list(norms.values())
        ratio = max(vals) / max(min(vals), 1e-10)
        if ratio > 10:
            findings.append(Finding(
                "attention_imbalance", "WARN", layer_key,
                {"norms": norms, "ratio": ratio}
            ))
    return findings


def post_detect_head_redundancy(ctx):
    """Compare Q projection weights across heads within each layer."""
    q_weights = ctx.get("_q_proj_weights", {})
    findings = []
    for layer_key, q_tensor in q_weights.items():
        if q_tensor.dim() != 2:
            continue
        # Infer number of heads from hidden_size in meta
        meta = ctx.get("meta", {})
        hidden_size = 0
        if isinstance(meta, dict):
            hidden_size = meta.get("hidden_size", 0)
        else:
            hidden_size = getattr(meta, "hidden_size", 0)
        if hidden_size <= 0:
            hidden_size = q_tensor.shape[1]
        # Q proj shape is (num_heads * head_dim, hidden_size)
        total_out = q_tensor.shape[0]
        if hidden_size > 0 and total_out > hidden_size:
            head_dim = hidden_size  # rough estimate
            n_heads = total_out // head_dim
        else:
            # Try common head dims
            for hd in [64, 128, 96, 80, 48, 32]:
                if total_out % hd == 0:
                    n_heads = total_out // hd
                    head_dim = hd
                    break
            else:
                continue
        if n_heads < 2:
            continue
        # Extract per-head weight blocks and compare
        head_dim = total_out // n_heads
        head_vecs = []
        for h in range(n_heads):
            block = q_tensor[h * head_dim:(h + 1) * head_dim].flatten()
            norm = block.norm().clamp(min=1e-8)
            head_vecs.append(block / norm)
        redundant_pairs = []
        for i in range(n_heads):
            for j in range(i + 1, n_heads):
                sim = (head_vecs[i] * head_vecs[j]).sum().item()
                if sim > 0.99:
                    redundant_pairs.append((i, j, sim))
        if redundant_pairs:
            findings.append(Finding(
                "head_redundancy", "WARN", layer_key,
                {"redundant_pairs": [(i, j, round(s, 4)) for i, j, s in redundant_pairs[:10]],
                 "num_heads": n_heads, "head_dim": head_dim},
            ))
    return findings


def post_detect_representation_drift(ctx):
    """Compare per-element weight norms of consecutive layers.

    If adjacent layers have dramatically different norm profiles (ratio > 10x),
    this indicates representation drift — the model's internal representations
    shift abruptly between layers.
    """
    layer_norms = ctx.get("_layer_weight_norms", {})
    if len(layer_norms) < 2:
        return []
    # Compute mean per-element norm for each layer
    layer_means = {}
    for layer_idx, norms_list in sorted(layer_norms.items()):
        if norms_list:
            layer_means[layer_idx] = sum(norms_list) / len(norms_list)
    sorted_layers = sorted(layer_means.keys())
    findings = []
    for i in range(len(sorted_layers) - 1):
        l_curr = sorted_layers[i]
        l_next = sorted_layers[i + 1]
        norm_curr = layer_means[l_curr]
        norm_next = layer_means[l_next]
        if norm_curr < 1e-10 and norm_next < 1e-10:
            continue
        ratio = max(norm_curr, norm_next) / max(min(norm_curr, norm_next), 1e-10)
        if ratio > 10:
            findings.append(Finding(
                "representation_drift", "WARN",
                f"layers.{l_curr}->layers.{l_next}",
                {"layer_a": l_curr, "layer_b": l_next,
                 "norm_a": norm_curr, "norm_b": norm_next,
                 "ratio": ratio},
            ))
    return findings


# ── Causal tracing detectors ───────────────────────────────────────────────

def _collect_causal_norms(name, tensor, ctx):
    """Collector: accumulate per-element norms keyed by layer type for causal outlier detection."""
    import re
    if tensor.dim() < 2:
        return []
    # Extract a layer type suffix (e.g. "attention.q_proj.weight", "mlp.down_proj.weight")
    match = re.search(r'layers?[._]\d+[._](.*)', name)
    if not match:
        return []
    layer_type = match.group(1)
    per_elem = tensor.float().norm().item() / (tensor.numel() ** 0.5)
    causal_norms = ctx.setdefault("_causal_type_norms", {})
    entries = causal_norms.setdefault(layer_type, [])
    entries.append({"name": name, "per_elem_norm": per_elem})
    return []


def post_detect_causal_outlier(ctx):
    """Identify layers whose per-element norm is a causal outlier among same-type layers.

    A layer is flagged if its norm is >2x (WARN) or >3x (ERROR) the mean norm
    of all layers of the same type, OR if its condition number exceeds 1M.
    These layers are statistically most likely to cause generation collapse.
    """
    causal_norms = ctx.get("_causal_type_norms", {})
    findings = []
    for layer_type, entries in causal_norms.items():
        if len(entries) < 2:
            continue
        norms = [e["per_elem_norm"] for e in entries]
        mean_norm = sum(norms) / len(norms)
        if mean_norm < 1e-10:
            continue
        for entry in entries:
            ratio = entry["per_elem_norm"] / mean_norm
            if ratio > 3.0:
                severity = "ERROR"
            elif ratio > 2.0:
                severity = "WARN"
            else:
                continue
            findings.append(Finding(
                condition="causal_outlier",
                severity=severity,
                param_name=entry["name"],
                details={
                    "per_elem_norm": entry["per_elem_norm"],
                    "type_mean_norm": mean_norm,
                    "ratio": ratio,
                    "layer_type": layer_type,
                },
            ))
    return findings


def _collect_layer_isolation(name, tensor, ctx):
    """Collector: gather flattened weight statistics per layer index for isolation detection."""
    import re
    if tensor.dim() < 2:
        return []
    match = re.search(r'layers?[._](\d+)', name)
    if not match:
        return []
    layer_idx = int(match.group(1))
    # Extract sub-component type (e.g. "attention.q_proj.weight")
    type_match = re.search(r'layers?[._]\d+[._](.*)', name)
    if not type_match:
        return []
    component = type_match.group(1)
    isolation_data = ctx.setdefault("_layer_isolation_data", {})
    by_component = isolation_data.setdefault(component, {})
    # Store mean and std of the weight tensor for this layer
    t = tensor.float()
    by_component[layer_idx] = {
        "name": name,
        "mean": t.mean().item(),
        "std": t.std().item(),
        "per_elem_norm": t.norm().item() / (t.numel() ** 0.5),
    }
    return []


def post_detect_layer_isolation(ctx):
    """Detect when a layer's weight space is isolated from its neighbors.

    For each component type (e.g. attention.q_proj.weight), compares
    consecutive layer statistics. If the norm ratio between adjacent layers
    of the same type exceeds 5x, flags as isolated.
    """
    isolation_data = ctx.get("_layer_isolation_data", {})
    findings = []
    for component, layer_data in isolation_data.items():
        if len(layer_data) < 2:
            continue
        sorted_indices = sorted(layer_data.keys())
        for i in range(len(sorted_indices) - 1):
            idx_a = sorted_indices[i]
            idx_b = sorted_indices[i + 1]
            data_a = layer_data[idx_a]
            data_b = layer_data[idx_b]
            norm_a = data_a["per_elem_norm"]
            norm_b = data_b["per_elem_norm"]
            if norm_a < 1e-10 and norm_b < 1e-10:
                continue
            ratio = max(norm_a, norm_b) / max(min(norm_a, norm_b), 1e-10)
            if ratio > 5.0:
                # Flag the layer with the higher norm as isolated
                if norm_a > norm_b:
                    flagged_name = data_a["name"]
                else:
                    flagged_name = data_b["name"]
                findings.append(Finding(
                    condition="layer_isolation",
                    severity="WARN",
                    param_name=flagged_name,
                    details={
                        "layer_a": idx_a,
                        "layer_b": idx_b,
                        "norm_a": norm_a,
                        "norm_b": norm_b,
                        "ratio": ratio,
                        "component": component,
                    },
                ))
    return findings


def causal_rank(findings, state_dict):
    """Rank tensors by causal responsibility for downstream failures.

    Given a list of findings and the state dict, produces a sorted ranking
    of which tensors are most likely causing generation collapse or other
    failures. Higher score = more likely culprit.

    Args:
        findings: list of Finding objects from diagnose()
        state_dict: the model state dict

    Returns:
        list of dicts: [{tensor_name, causal_score, reason}, ...] sorted by score descending
    """
    # Severity weights for scoring
    severity_score = {"ERROR": 10, "WARN": 5, "INFO": 1}
    # Condition weights — conditions more indicative of causal responsibility
    condition_weight = {
        "causal_outlier": 3.0,
        "layer_isolation": 2.5,
        "exploding_norm": 2.0,
        "nan_inf": 5.0,
        "dead_neurons": 1.5,
        "representation_drift": 2.0,
        "weight_corruption": 3.0,
        "heavy_tails": 1.0,
        "norm_drift": 1.0,
        "gradient_noise": 1.5,
        "vanishing_norm": 1.0,
        "saturated_weights": 0.5,
    }

    # Accumulate scores per tensor
    tensor_scores = {}
    tensor_reasons = {}
    for f in findings:
        name = f.param_name
        base = severity_score.get(f.severity, 1)
        weight = condition_weight.get(f.condition, 1.0)
        score = base * weight
        tensor_scores[name] = tensor_scores.get(name, 0) + score
        reasons = tensor_reasons.setdefault(name, [])
        reasons.append(f.condition)

    # Build result list
    results = []
    for name, score in tensor_scores.items():
        reasons = tensor_reasons[name]
        # Deduplicate reasons while preserving order
        seen = set()
        unique_reasons = []
        for r in reasons:
            if r not in seen:
                seen.add(r)
                unique_reasons.append(r)
        results.append({
            "tensor_name": name,
            "causal_score": score,
            "reason": ", ".join(unique_reasons),
        })

    # Sort by score descending
    results.sort(key=lambda x: x["causal_score"], reverse=True)
    return results


# ── Register all detectors ─────────────────────────────────────────────────

def _rx_dead_neurons(f):
    return Prescription(
        name="reinit_dead_neurons",
        description=f"Reinit {f.details['dead_count']} dead {f.details['dim']} in {f.param_name}",
        risk="low", finding=f, action="reinit_dead",
        params={"indices": f.details["dead_indices"], "dim": f.details["dim"]},
        explanation=(
            "Dead neurons contribute nothing to the model's computation. "
            "Reinitializing them with small random values gives them a chance "
            "to learn useful features during further training. Risk is low "
            "because the neurons were already inactive."
        ),
    )

def _rx_nudge_gate(f):
    return Prescription(
        name="nudge_gate_open",
        description=f"Nudge {f.param_name} from {f.details['sigmoid']:.4f} toward 4.7%",
        risk="medium", finding=f, action="set_gate",
        params={"value": -3.0},
        explanation=(
            "This gate is stuck closed (sigmoid near 0), blocking the signal "
            "path entirely. Nudging it to sigmoid ~4.7% allows a small signal "
            "through without overwhelming the network. Risk is medium because "
            "it changes the model's behavior."
        ),
    )

def _rx_pull_gate(f):
    return Prescription(
        name="pull_gate_back",
        description=f"Pull {f.param_name} from {f.details['sigmoid']:.4f} back to 95%",
        risk="medium", finding=f, action="set_gate",
        params={"value": 3.0},
        explanation=(
            "This gate is stuck fully open (sigmoid near 1), passing the "
            "full signal unconditionally. Pulling it back to ~95% restores "
            "gradient flow so the gate can learn to modulate. Risk is medium "
            "because it changes the model's behavior."
        ),
    )

def _rx_scale_norm(f):
    return Prescription(
        name="scale_down_norm",
        description=f"Scale {f.param_name} norm from {f.details['per_elem_norm']:.2f} toward 1.0",
        risk="medium", finding=f, action="scale_norm",
        params={"target_per_elem": 1.0},
        explanation=(
            "Exploding weight norms amplify activations through the network, "
            "leading to numerical instability and poor generation quality. "
            "Scaling to healthy range reduces this amplification."
        ),
    )

def _rx_reinit_vanishing(f):
    return Prescription(
        name="reinit_vanishing",
        description=f"Reinit {f.param_name} (norm {f.details['per_elem_norm']:.2e})",
        risk="low", finding=f, action="reinit_full",
        explanation=(
            "Near-zero parameters have effectively no contribution and produce "
            "near-zero gradients, preventing learning. Reinitializing gives "
            "them a fresh start."
        ),
    )

def _rx_clamp_tails(f):
    return Prescription(
        name="clamp_tails",
        description=f"Clamp {f.param_name} outliers (kurtosis={f.details['kurtosis']:.0f})",
        risk="medium", finding=f, action="clamp_outliers",
        params={"sigma": CLAMP_SIGMA},
        explanation=(
            "Heavy-tailed weight distributions (high kurtosis) contain extreme "
            "outliers that dominate computation for certain inputs. Clamping "
            "beyond 4 standard deviations removes these outliers while "
            "preserving the bulk of the distribution."
        ),
    )

def _rx_reset_norm(f):
    return Prescription(
        name="reset_norm",
        description=f"Reset {f.param_name} mean from {f.details['mean']:.3f} toward 1.0",
        risk="low", finding=f, action="reset_norm_weights",
        explanation=(
            "LayerNorm weights should be close to 1.0. Drift indicates the "
            "normalization layer is scaling features unevenly, which can cause "
            "training instability. Resetting to 1.0 restores uniform scaling."
        ),
    )

def _rx_spectral_denoise(f):
    """Prescribe spectral denoising for gradient_noise findings with high condition numbers."""
    cond = f.details.get("condition_number", 0)
    if cond > 10_000:
        return Prescription(
            name="spectral_denoise",
            description=f"Spectral denoising for {f.param_name} (condition={cond:.0f})",
            risk="medium",
            finding=f,
            action="spectral_denoise",
            params={
                "energy_threshold": 0.99,
                "max_condition": 1000,
                "min_rank_ratio": 0.1,
            },
            explanation=(
                f"This parameter has a condition number of {cond:.0f}, indicating "
                f"near-singular directions that amplify noise during training. "
                f"Spectral denoising removes the smallest singular values to cap "
                f"the condition number at 1000, retaining 99% of spectral energy."
            ),
        )
    # Fall through to advisory for moderate condition numbers
    return Prescription(
        name=f"advisory_{f.condition}",
        description=f"Advisory: {f.condition} detected in {f.param_name}",
        risk="low", finding=f, action="advisory",
        explanation=(
            "This is an advisory finding — no automatic fix is available. "
            "Manual investigation is recommended."
        ),
    )

def _rx_desaturate(f):
    return Prescription(
        name="desaturate",
        description=f"Desaturate {f.param_name} ({f.details['near_max_pct']:.0%} at boundary)",
        risk="medium", finding=f, action="desaturate",
        params={"factor": 0.8},
        explanation=(
            "Saturated weights are pinned near their maximum values, reducing "
            "the effective dynamic range of the layer. Scaling down increases "
            "gradient flow and allows finer-grained adjustments during training."
        ),
    )

def _rx_fix_nan(f):
    return Prescription(
        name="fix_nan_inf",
        description=f"Zero out {f.details['nan_count']} NaN + {f.details['inf_count']} Inf in {f.param_name}",
        risk="high", finding=f, action="fix_nan_inf",
        explanation=(
            "NaN/Inf values cause cascading failures in any computation they "
            "touch. Zeroing them out prevents propagation but the underlying "
            "cause (overflow, division by zero) may need investigation. Risk "
            "is high because zeroing may change model behavior."
        ),
    )

def _rx_perturb_identical(f):
    return Prescription(
        name="perturb_identical_rows",
        description=f"Add noise to {f.details['duplicate_pairs']} duplicate row pairs in {f.param_name}",
        risk="low", finding=f, action="perturb_identical",
        explanation=(
            "Identical rows mean redundant computation — the model wastes "
            "capacity on duplicate features. Adding small noise breaks the "
            "symmetry so each row can specialize during further training."
        ),
    )

def _rx_advisory(f):
    return Prescription(
        name=f"advisory_{f.condition}",
        description=f"Advisory: {f.condition} detected in {f.param_name}",
        risk="low", finding=f, action="advisory",
        explanation=(
            "This is an advisory finding — no automatic fix is available. "
            "Manual investigation is recommended."
        ),
    )

# Register everything
REGISTRY.register("dead_neurons", detect_dead_neurons, _rx_dead_neurons, "low",
                   "Reinitialize dead neurons with small Kaiming values")
REGISTRY.register("stuck_gate_closed", detect_stuck_gates, _rx_nudge_gate, "medium",
                   "Nudge closed gate toward trainable range")
REGISTRY.register("stuck_gate_open", detect_stuck_gates, _rx_pull_gate, "medium",
                   "Pull back saturated-open gate")
REGISTRY.register("exploding_norm", detect_exploding_norm, _rx_scale_norm, "medium",
                   "Scale down weights to healthy norm range")
REGISTRY.register("vanishing_norm", detect_vanishing_norm, _rx_reinit_vanishing, "low",
                   "Reinitialize near-zero parameter")
REGISTRY.register("heavy_tails", detect_heavy_tails, _rx_clamp_tails, "medium",
                   "Clamp outlier weights to reduce heavy tails")
REGISTRY.register("norm_drift", detect_norm_drift, _rx_reset_norm, "low",
                   "Reset LayerNorm weights toward 1.0")
REGISTRY.register("saturated_weights", detect_saturated_weights, _rx_desaturate, "medium",
                   "Scale down saturated weights")
REGISTRY.register("nan_inf", detect_nan_inf, _rx_fix_nan, "high",
                   "Fix NaN/Inf values in parameters")
REGISTRY.register("identical_rows", detect_identical_rows, _rx_perturb_identical, "low",
                   "Break symmetry in duplicate rows")
REGISTRY.register("dtype_mismatch", detect_dtype_mismatch, risk="low",
                   description="Mixed dtypes across parameters")
REGISTRY.register("attention_imbalance", detect_attention_imbalance,
                   risk="medium", description="Q/K/V projection norm imbalance")
REGISTRY.register("weight_corruption", detect_weight_corruption, risk="info",
                   description="Truncated or corrupted weight tensors")
REGISTRY.register("head_redundancy", detect_head_redundancy, risk="info",
                   description="Redundant attention heads")
REGISTRY.register("positional_encoding_issues", detect_positional_issues, risk="info",
                   description="Broken or degenerate positional encodings")
REGISTRY.register("token_collapse", detect_token_collapse, _rx_advisory, "low",
                   "Near-identical output token embeddings")
REGISTRY.register("gradient_noise", detect_gradient_noise, _rx_spectral_denoise, "medium",
                   "Spectral denoising for high condition number (gradient instability)")
REGISTRY.register("_collect_layer_norms", _collect_layer_norms, risk="low",
                   description="Collector for representation drift detection")
REGISTRY.register("moe_router_collapse", detect_moe_router_collapse, _rx_advisory, "low",
                   "MoE router/gate entropy collapse or uniformity")
REGISTRY.register("lora_merge_artifacts", detect_lora_merge_artifacts, _rx_advisory, "low",
                   "Low effective rank suggesting LoRA merge artifacts")
REGISTRY.register("quantization_degradation", detect_quantization_degradation, _rx_advisory, "low",
                   "Quality loss from quantization (low unique-value count, grid patterns)")
REGISTRY.register("_collect_model_aging", _collect_model_aging, risk="low",
                   description="Collector for model aging detection")
REGISTRY.register("_collect_causal_norms", _collect_causal_norms, risk="low",
                   description="Collector for causal outlier detection")
REGISTRY.register("_collect_layer_isolation", _collect_layer_isolation, risk="low",
                   description="Collector for layer isolation detection")


# ── Diagnosis engine ───────────────────────────────────────────────────────

def diagnose(state_dict, meta=None, verbose=False, plugins=True):
    """Run all diagnostic checks. Returns list of Findings.

    Parameters
    ----------
    state_dict : dict
        Model state dict (parameter name -> tensor).
    meta : dict, optional
        Model metadata.
    verbose : bool
        Print progress messages.
    plugins : bool
        If True (default), auto-load installed plugins before diagnosis.
        Set to False to skip plugin loading.
    """
    if plugins:
        from model_clinic._plugins import load_plugins, plugins_loaded
        if not plugins_loaded():
            loaded = load_plugins(REGISTRY)
            if loaded and verbose:
                print(f"  Loaded plugins: {', '.join(loaded)}")

    context = {"meta": meta or {}}
    findings = REGISTRY.detect_all(state_dict, context, verbose=verbose)

    # Post-scan detectors
    if verbose:
        print("  Checking dtype_mismatch...")
    findings.extend(post_detect_dtype_mismatch(context))
    if verbose:
        print("  Checking attention_imbalance...")
    findings.extend(post_detect_attention_imbalance(context))
    if verbose:
        print("  Checking head_redundancy...")
    findings.extend(post_detect_head_redundancy(context))
    if verbose:
        print("  Checking representation_drift...")
    findings.extend(post_detect_representation_drift(context))
    if verbose:
        print("  Checking model_aging...")
    findings.extend(post_detect_model_aging(context))
    if verbose:
        print("  Checking causal_outlier...")
    findings.extend(post_detect_causal_outlier(context))
    if verbose:
        print("  Checking layer_isolation...")
    findings.extend(post_detect_layer_isolation(context))

    return findings


def prescribe(findings, conservative=False):
    """Map findings to prescriptions."""
    return REGISTRY.prescribe(findings, conservative=conservative)


# ── Batch processing ──────────────────────────────────────────────────────

def _examine_one(path, hf=False):
    """Examine a single model, returning an ExamResult (never raises)."""
    try:
        sd, raw_meta = load_state_dict(path, hf=hf)
        meta = build_meta(sd, source=raw_meta.get("source", "checkpoint"))
        findings = diagnose(sd, meta)
        prescriptions = prescribe(findings)
        health = compute_health_score(findings)
        return ExamResult(
            path=path,
            findings=findings,
            prescriptions=prescriptions,
            health_score=health,
            meta=meta,
        )
    except Exception as e:
        return ExamResult(path=path, error=str(e))


def examine_batch(paths, hf=False, parallel=False):
    """Examine multiple models at once.

    Args:
        paths: list of checkpoint paths or HF model names
        hf: treat all as HuggingFace models
        parallel: if True, use ThreadPoolExecutor (I/O bound loading)

    Returns:
        list of ExamResult, one per model
    """
    if parallel:
        from concurrent.futures import ThreadPoolExecutor
        max_workers = min(len(paths), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            results = list(pool.map(lambda p: _examine_one(p, hf=hf), paths))
        return results
    return [_examine_one(p, hf=hf) for p in paths]


# ── Treatment pipelines ─────────────────────────────────────────────────────

class TreatmentPipeline:
    """A reusable sequence of treatment steps."""

    def __init__(self, steps):
        """
        Args:
            steps: list of (condition_name, params_override) tuples
        """
        self.steps = steps

    def run(self, state_dict, conservative=False, dry_run=False):
        """Run the pipeline on a state dict.

        Returns:
            PipelineResult with findings, prescriptions, treatments,
            and health scores before and after.
        """
        # Diagnose
        all_findings = diagnose(state_dict)
        health_before = compute_health_score(all_findings)

        # Filter to requested conditions
        step_conditions = {name for name, _ in self.steps}
        relevant = [f for f in all_findings if f.condition in step_conditions]

        # Build prescriptions
        all_rx = prescribe(relevant, conservative=conservative)

        # Build override map: condition -> params
        overrides = {name: params for name, params in self.steps}

        # Apply overrides to prescription params
        for rx in all_rx:
            extra = overrides.get(rx.finding.condition, {})
            if extra:
                rx.params.update(extra)

        # Apply treatments in order
        treatments = []
        for rx in all_rx:
            result = apply_treatment(state_dict, rx, dry_run=dry_run)
            treatments.append(result)

        # Re-diagnose for after score
        after_findings = diagnose(state_dict)
        health_after = compute_health_score(after_findings)

        return PipelineResult(
            findings=relevant,
            prescriptions=all_rx,
            treatments=treatments,
            health_before=health_before,
            health_after=health_after,
        )

    def describe(self):
        """Print what the pipeline will do."""
        print("Treatment Pipeline:")
        for i, (condition, params) in enumerate(self.steps, 1):
            if params:
                print(f"  {i}. {condition} (overrides: {params})")
            else:
                print(f"  {i}. {condition}")


def create_pipeline(steps):
    """Create a treatment pipeline.

    Args:
        steps: list of (condition_name, params_override) tuples
               e.g. [("dead_neurons", {"scale": 0.01}), ("norm_drift", {})]

    Returns:
        TreatmentPipeline object
    """
    return TreatmentPipeline(steps)


# ── Treatment engine ───────────────────────────────────────────────────────

def apply_treatment(state_dict, rx, dry_run=False):
    """Apply a single prescription. Returns TreatmentResult."""
    name = rx.finding.param_name
    if name not in state_dict:
        return TreatmentResult(rx, False, f"Parameter {name} not found")

    tensor = state_dict[name]

    # Skip non-float tensors (counters, indices, flags)
    if not tensor.is_floating_point():
        return TreatmentResult(rx, False, f"Skipped non-float tensor ({tensor.dtype})")

    backup = tensor.clone()

    if dry_run:
        return TreatmentResult(rx, True, f"[DRY RUN] Would apply {rx.action} to {name}", backup)

    success, desc = _do_treatment(tensor, rx)
    return TreatmentResult(rx, success, desc, backup)


def _do_treatment(tensor, rx):
    """Execute a treatment operation on a tensor."""
    action = rx.action
    params = rx.params

    if action == "reinit_dead":
        indices = params["indices"]
        dim = params["dim"]
        with torch.no_grad():
            if dim == "rows" and tensor.dim() >= 2:
                for idx in indices:
                    if idx < tensor.shape[0]:
                        fan_in = tensor.shape[1]
                        std = (2.0 / fan_in) ** 0.5
                        tensor[idx] = torch.randn_like(tensor[idx]) * std * 0.1
                return True, f"Reinit {len(indices)} dead rows (0.1x Kaiming)"
            elif dim == "cols" and tensor.dim() >= 2:
                for idx in indices:
                    if idx < tensor.shape[1]:
                        fan_in = tensor.shape[0]
                        std = (2.0 / fan_in) ** 0.5
                        tensor[:, idx] = torch.randn(tensor.shape[0], device=tensor.device, dtype=tensor.dtype) * std * 0.1
                return True, f"Reinit {len(indices)} dead cols (0.1x Kaiming)"
        return False, "Unsupported dim"

    elif action == "set_gate":
        with torch.no_grad():
            old = tensor.item()
            old_sig = torch.sigmoid(tensor.float()).item()
            tensor.fill_(params["value"])
            new_sig = torch.sigmoid(tensor.float()).item()
            return True, f"Gate: {old:.2f} (sig={old_sig:.4f}) -> {params['value']:.2f} (sig={new_sig:.4f})"

    elif action == "scale_norm":
        with torch.no_grad():
            t = tensor.float()
            current = t.norm().item() / (t.numel() ** 0.5)
            target = params["target_per_elem"]
            if current > 0:
                factor = target / current
                tensor.mul_(factor)
                new_norm = tensor.float().norm().item() / (tensor.numel() ** 0.5)
                return True, f"Norm: {current:.4f} -> {new_norm:.4f}"
        return False, "Zero norm"

    elif action == "reinit_full":
        with torch.no_grad():
            if tensor.dim() >= 2:
                torch.nn.init.kaiming_uniform_(tensor)
                tensor.mul_(0.1)
                return True, f"Reinit (0.1x Kaiming), norm={tensor.float().norm().item():.4f}"
            else:
                tensor.zero_()
                return True, "Reset to zero"

    elif action == "clamp_outliers":
        with torch.no_grad():
            t = tensor.float()
            mean, std = t.mean(), t.std()
            sigma = params["sigma"]
            lo, hi = (mean - sigma * std).item(), (mean + sigma * std).item()
            old_range = (t.min().item(), t.max().item())
            tensor.clamp_(lo, hi)
            new_range = (tensor.min().item(), tensor.max().item())
            return True, f"Clamped [{old_range[0]:.4f}, {old_range[1]:.4f}] -> [{new_range[0]:.4f}, {new_range[1]:.4f}]"

    elif action == "reset_norm_weights":
        with torch.no_grad():
            old_mean = tensor.float().mean().item()
            tensor.fill_(1.0)
            return True, f"Norm weights: {old_mean:.4f} -> 1.0"

    elif action == "desaturate":
        with torch.no_grad():
            old_max = tensor.abs().max().item()
            tensor.mul_(params["factor"])
            return True, f"Desaturated: max {old_max:.4f} -> {tensor.abs().max().item():.4f}"

    elif action == "fix_nan_inf":
        with torch.no_grad():
            mask = torch.isnan(tensor) | torch.isinf(tensor)
            count = mask.sum().item()
            tensor[mask] = 0
            return True, f"Zeroed {count} NaN/Inf values"

    elif action == "perturb_identical":
        with torch.no_grad():
            if tensor.dim() >= 2:
                noise_scale = tensor.std().item() * 0.01
                t = tensor.float()
                norms = t.norm(dim=1, keepdim=True).clamp(min=1e-8)
                normed = t / norms
                n = min(tensor.shape[0], 200)
                sims = normed[:n] @ normed[:n].T
                mask = ~torch.eye(n, dtype=torch.bool, device=sims.device)
                dups = (sims[mask] > SIMILARITY_THRESHOLD).nonzero(as_tuple=True)[0]
                # Add small noise to break symmetry
                for idx in range(min(len(dups), 100)):
                    row = dups[idx].item() // (n - 1)
                    tensor[row] += torch.randn_like(tensor[row]) * noise_scale
                return True, f"Perturbed {min(len(dups), 100)} near-duplicate rows"
        return False, "Not a 2D tensor"

    elif action == "spectral_denoise":
        from model_clinic._repair.spectral import spectral_denoise_with_report
        with torch.no_grad():
            energy_threshold = params.get("energy_threshold", 0.99)
            max_condition = params.get("max_condition", 1000)
            min_rank_ratio = params.get("min_rank_ratio", 0.1)
            denoised, report = spectral_denoise_with_report(
                tensor, rx.finding.param_name,
                energy_threshold=energy_threshold,
                max_condition=max_condition,
                min_rank_ratio=min_rank_ratio,
            )
            if report.effective_rank < report.original_rank:
                tensor.copy_(denoised)
                return True, (
                    f"Spectral denoise: rank {report.original_rank} -> {report.effective_rank}, "
                    f"cond {report.condition_before:.0f} -> {report.condition_after:.0f}, "
                    f"energy retained {report.energy_retained:.4f}"
                )
            return True, "Already well-conditioned — no change needed"

    elif action == "advisory":
        return True, "Advisory only — no automatic fix"

    return False, f"Unknown action: {action}"


def rollback_treatment(state_dict, result):
    """Roll back a single treatment using saved backup."""
    if result.backup is not None and result.prescription.finding.param_name in state_dict:
        state_dict[result.prescription.finding.param_name] = result.backup
        return True
    return False


# ── Runtime diagnostics ────────────────────────────────────────────────────

def diagnose_runtime(model, tokenizer, device, prompts=None):
    """Run runtime diagnostics on a live model. Returns list of Findings."""
    prompts = prompts or DEFAULT_PROMPTS
    findings = []

    # 1. Logit entropy / collapse detection
    entropy_data = eval_logit_entropy(model, tokenizer, device, prompts)
    if entropy_data["collapsed"]:
        findings.append(Finding(
            "generation_collapse", "ERROR", "model",
            {"avg_entropy": entropy_data["avg_entropy"],
             "avg_top1_prob": entropy_data["avg_top1_prob"],
             "vocab_used": entropy_data["vocab_used"]},
        ))
    elif entropy_data["avg_top1_prob"] > 0.8:
        findings.append(Finding(
            "low_entropy", "WARN", "model",
            {"avg_entropy": entropy_data["avg_entropy"],
             "avg_top1_prob": entropy_data["avg_top1_prob"]},
        ))

    # 2. Generation coherence
    coherent, total, details = eval_coherence(model, tokenizer, device, prompts)
    if coherent < total * 0.5:
        findings.append(Finding(
            "low_coherence", "ERROR" if coherent == 0 else "WARN", "model",
            {"coherent": coherent, "total": total,
             "details": [{"prompt": d["prompt"], "coherent": d["coherent"],
                          "word_count": d["word_count"], "repetition": d["repetition"]}
                         for d in details]},
        ))

    # 3. Diversity
    diversity = eval_diversity(model, tokenizer, device, prompts)
    if diversity["all_same"]:
        findings.append(Finding(
            "response_uniformity", "WARN", "model",
            {"distinct_1": diversity["distinct_1"],
             "unique_responses": diversity["unique_responses"]},
        ))

    # 4. Activation health (hook-based)
    findings.extend(_check_activation_health(model, tokenizer, device, prompts))

    # 5. Residual stream growth
    findings.extend(_check_residual_growth(model, tokenizer, device, prompts))

    return findings


def _check_activation_health(model, tokenizer, device, prompts):
    """Hook into forward pass and check activation stats per layer."""
    findings = []
    layer_stats = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            t = out.detach().float()
            layer_stats[name] = {
                "mean": t.mean().item(),
                "std": t.std().item(),
                "max": t.abs().max().item(),
                "has_nan": torch.isnan(t).any().item(),
                "has_inf": torch.isinf(t).any().item(),
                "zero_frac": (t.abs() < 1e-8).float().mean().item(),
            }
        return hook_fn

    # Register hooks on all layers
    hooks = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        return findings

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(f"layer.{i}"))
        hooks.append(h)

    # Run one prompt
    model.eval()
    prompt = prompts[0]
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    # Analyze
    for name, stats in layer_stats.items():
        if stats["has_nan"]:
            findings.append(Finding("activation_nan", "ERROR", name,
                                    {"stats": stats}))
        if stats["has_inf"]:
            findings.append(Finding("activation_inf", "ERROR", name,
                                    {"stats": stats}))
        if stats["max"] > 1e4:
            findings.append(Finding("activation_explosion", "WARN", name,
                                    {"max": stats["max"]}))
        if stats["std"] < 1e-8:
            findings.append(Finding("activation_collapse", "WARN", name,
                                    {"std": stats["std"], "zero_frac": stats["zero_frac"]}))

    return findings


def _check_residual_growth(model, tokenizer, device, prompts):
    """Track residual stream norm growth across layers."""
    findings = []
    norms = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            norms.append((idx, out.detach().float().norm().item()))
        return hook_fn

    hooks = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        return findings

    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(i))
        hooks.append(h)

    model.eval()
    prompt = prompts[0]
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = prompt
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        try:
            model(input_ids=input_ids)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    if len(norms) >= 3:
        norm_vals = [n for _, n in sorted(norms)]
        # Check for exponential growth
        if norm_vals[-1] > norm_vals[0] * 100:
            findings.append(Finding(
                "residual_explosion", "WARN", "residual_stream",
                {"first_layer_norm": norm_vals[0],
                 "last_layer_norm": norm_vals[-1],
                 "ratio": norm_vals[-1] / max(norm_vals[0], 1e-10)},
            ))
        # Check for shrinkage
        if norm_vals[-1] < norm_vals[0] * 0.01 and norm_vals[0] > 1e-6:
            findings.append(Finding(
                "residual_collapse", "WARN", "residual_stream",
                {"first_layer_norm": norm_vals[0],
                 "last_layer_norm": norm_vals[-1],
                 "ratio": norm_vals[-1] / max(norm_vals[0], 1e-10)},
            ))

    return findings


# ── Printing ───────────────────────────────────────────────────────────────

RISK_SYMBOL = {"low": "[LOW]", "medium": "[MED]", "high": "[HI!]"}


def print_findings(findings):
    """Print diagnosed findings grouped by condition."""
    if not findings:
        print("  No issues found. Model looks healthy.")
        return

    by_condition = defaultdict(list)
    for f in findings:
        by_condition[f.condition].append(f)

    for condition, fs in sorted(by_condition.items()):
        severities = [f.severity for f in fs]
        worst = "ERROR" if "ERROR" in severities else "WARN" if "WARN" in severities else "INFO"
        print(f"\n  [{worst}] {condition} ({len(fs)} instance(s))")
        for f in fs[:10]:
            _print_finding_detail(f)
        if len(fs) > 10:
            print(f"    ... and {len(fs) - 10} more")


def _print_finding_detail(f):
    """Print details for a single finding."""
    d = f.details
    formatters = {
        "dead_neurons": lambda: f"    {f.param_name}: {d['dead_count']}/{d['total']} dead {d['dim']} ({d['pct']:.1%})",
        "stuck_gate_closed": lambda: f"    {f.param_name}: raw={d['raw']:+.2f}, sigmoid={d['sigmoid']:.6f}",
        "stuck_gate_open": lambda: f"    {f.param_name}: raw={d['raw']:+.2f}, sigmoid={d['sigmoid']:.6f}",
        "exploding_norm": lambda: f"    {f.param_name}: per-elem norm={d['per_elem_norm']:.4f}",
        "vanishing_norm": lambda: f"    {f.param_name}: per-elem norm={d['per_elem_norm']:.2e}",
        "heavy_tails": lambda: f"    {f.param_name}: kurtosis={d['kurtosis']:.0f} (normal=3)",
        "norm_drift": lambda: f"    {f.param_name}: mean={d['mean']:.4f} (should be ~1.0)",
        "saturated_weights": lambda: f"    {f.param_name}: {d['near_max_pct']:.0%} of weights at boundary",
        "nan_inf": lambda: f"    {f.param_name}: {d['nan_count']} NaN, {d['inf_count']} Inf / {d['total']} total",
        "identical_rows": lambda: f"    {f.param_name}: {d['duplicate_pairs']} near-duplicate pairs (max sim={d['max_similarity']:.4f})",
        "attention_imbalance": lambda: f"    {f.param_name}: Q/K/V norm ratio={d['ratio']:.1f}x",
        "dtype_mismatch": lambda: f"    {d['minority_count']} params in {d['minority_dtype']} (majority: {d['majority_dtype']})",
        "generation_collapse": lambda: f"    avg top-1 prob={d['avg_top1_prob']:.2%}, entropy={d['avg_entropy']:.2f}, vocab used={d['vocab_used']}",
        "low_entropy": lambda: f"    avg top-1 prob={d['avg_top1_prob']:.2%}, entropy={d['avg_entropy']:.2f}",
        "low_coherence": lambda: f"    {d['coherent']}/{d['total']} coherent responses",
        "response_uniformity": lambda: f"    {d['unique_responses']} unique response(s), distinct-1={d['distinct_1']:.3f}",
        "activation_nan": lambda: f"    {f.param_name}: NaN in activations",
        "activation_inf": lambda: f"    {f.param_name}: Inf in activations",
        "activation_explosion": lambda: f"    {f.param_name}: max activation={d['max']:.0f}",
        "activation_collapse": lambda: f"    {f.param_name}: std={d['std']:.2e}, {d['zero_frac']:.0%} zeros",
        "residual_explosion": lambda: f"    norm ratio first->last: {d['ratio']:.0f}x",
        "residual_collapse": lambda: f"    norm ratio first->last: {d['ratio']:.4f}x (shrinking)",
        "weight_corruption": lambda: f"    {f.param_name}: {d['reason']} (shape={d.get('shape', '?')})",
        "head_redundancy": lambda: f"    {f.param_name}: {len(d['redundant_pairs'])} redundant head pair(s) across {d['num_heads']} heads",
        "positional_encoding_issues": lambda: f"    {f.param_name}: {d['reason']}",
        "token_collapse": lambda: f"    {f.param_name}: {d['collapsed_pair_fraction']:.1%} of sampled row pairs have cosine sim > 0.99",
        "gradient_noise": lambda: f"    {f.param_name}: condition number={d['condition_number']:.0f} (max_sv={d['max_sv']:.4f}, min_sv={d['min_sv']:.2e})",
        "representation_drift": lambda: f"    {f.param_name}: norm ratio={d['ratio']:.1f}x ({d['norm_a']:.4f} vs {d['norm_b']:.4f})",
        "moe_router_collapse": lambda: f"    {f.param_name}: {d['reason']} (norm_entropy={d['normalized_entropy']:.3f})",
        "lora_merge_artifacts": lambda: f"    {f.param_name}: effective_rank={d['effective_rank']:.1f}/{d['matrix_min_dim']} ({d['rank_ratio']:.3f})",
        "quantization_degradation": lambda: f"    {f.param_name}: {d['unique_values']} unique values in {d['sample_size']} samples ({d['unique_ratio']:.3%}){' [grid]' if d.get('grid_quantized') else ''}",
        "model_aging": lambda: f"    {f.param_name}: {d['reason']}" + (f" (rank_ratio={d['rank_ratio']:.3f})" if 'rank_ratio' in d else f" (ratio={d.get('ratio', d.get('merged_pair_fraction', '?'))})"),
    }
    fmt = formatters.get(f.condition, lambda: f"    {f.param_name}: {d}")
    print(fmt())


def print_exam(findings, prescriptions, explain=False):
    """Print full exam report."""
    errors = sum(1 for f in findings if f.severity == "ERROR")
    warns = sum(1 for f in findings if f.severity == "WARN")
    infos = sum(1 for f in findings if f.severity == "INFO")

    print(f"\n{'='*80}")
    print(f"DIAGNOSIS -- {len(findings)} finding(s) ({errors} errors, {warns} warnings, {infos} info)")
    print(f"{'='*80}")
    print_findings(findings)

    print(f"\n{'='*80}")
    print(f"TREATMENT PLAN -- {len(prescriptions)} prescription(s)")
    print(f"{'='*80}")

    if not prescriptions:
        print("  No treatments available.")
        return

    for i, rx in enumerate(prescriptions):
        risk = RISK_SYMBOL.get(rx.risk, rx.risk)
        print(f"\n  Rx #{i+1}: {rx.name} {risk}")
        print(f"    {rx.description}")
        for k, v in rx.params.items():
            if k == "indices":
                print(f"    {k}: [{len(v)} indices]")
            else:
                print(f"    {k}: {v}")
        if explain and rx.explanation:
            print(f"    WHY: {rx.explanation}")
            refs = format_references(rx.finding.condition)
            if refs:
                print(refs)


# ── Main ───────────────────────────────────────────────────────────────────

def build_parser(subparsers=None):
    """Build argparse parser (standalone or as subcommand)."""
    if subparsers:
        parser = subparsers.add_parser("exam", help="Diagnose model health")
        parser_treat = subparsers.add_parser("treat", help="Diagnose and treat model")
        for p in [parser, parser_treat]:
            _add_common_args(p)
        parser_treat.add_argument("--save", type=str, help="Save treated checkpoint")
        parser_treat.add_argument("--test", action="store_true", help="Run generation tests")
        parser_treat.add_argument("--dry-run", action="store_true", help="Show what would change")
        parser_treat.add_argument("--conservative", action="store_true", help="Only low-risk fixes")
        parser_treat.add_argument("--no-rollback", action="store_true", help="Don't auto-rollback on regression")
        parser_treat.add_argument("--manifest", type=str, default=None,
                                  help="Save treatment manifest (auto-saved with --save)")
        parser.set_defaults(func=run_exam)
        parser_treat.set_defaults(func=run_treat)
        return parser
    else:
        parser = argparse.ArgumentParser(description="Model clinic -- diagnose, prescribe, treat")
        _add_common_args(parser)
        parser.add_argument("--exam", action="store_true", help="Diagnose only")
        parser.add_argument("--treat", action="store_true", help="Apply treatments")
        parser.add_argument("--save", type=str, help="Save treated checkpoint")
        parser.add_argument("--test", action="store_true", help="Run generation tests")
        parser.add_argument("--dry-run", action="store_true", help="Show what would change")
        parser.add_argument("--conservative", action="store_true", help="Only low-risk fixes")
        parser.add_argument("--no-rollback", action="store_true", help="Don't auto-rollback")
        parser.add_argument("--manifest", type=str, default=None,
                            help="Save treatment manifest (auto-saved with --save)")
        return parser


def _add_common_args(parser):
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--runtime", action="store_true", help="Include runtime diagnostics")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated conditions")
    parser.add_argument("--export", type=str, default=None, help="Export report to JSON")
    parser.add_argument("--json", action="store_true", help="Output as JSON only")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed progress (which detector/treatment is running)")
    parser.add_argument("--explain", action="store_true", help="Show explanations for prescriptions")
    parser.add_argument("--example-prompts", action="store_true",
                        help="Use diverse example prompts for runtime testing instead of defaults")
    parser.add_argument("--profile", type=str, default=None,
                        choices=["llm", "vit", "diffusion", "auto"],
                        help="Architecture profile: run profile-specific detectors with healthy baselines")


def run_exam(args):
    """Run examination (diagnosis only)."""
    if getattr(args, "json", False):
        args.quiet = True
    verbose = getattr(args, "verbose", False) and not args.quiet
    if not args.quiet:
        print(f"Loading: {args.model}")
    state_dict, meta_dict = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict, source=meta_dict.get("source", "unknown"), extra=meta_dict)
    if not args.quiet:
        print(f"Loaded {meta.num_tensors} tensors, {meta.num_params:,} parameters")

    # Profile-based diagnosis
    profile_name = getattr(args, "profile", None)
    if profile_name:
        from model_clinic._profiles import get_profile, auto_detect_profile
        if profile_name == "auto":
            profile = auto_detect_profile(state_dict)
            if profile is None:
                if not args.quiet:
                    print("Could not auto-detect architecture profile, running full diagnosis")
            else:
                if not args.quiet:
                    print(f"Auto-detected profile: {profile.name}")
        else:
            profile = get_profile(profile_name)
        if profile_name != "auto" or (profile_name == "auto" and profile is not None):
            if not args.quiet:
                print(f"Using {profile.name} profile ({len(profile.detectors)} detectors)")
            findings = profile.diagnose(state_dict, meta_dict, verbose=verbose)
            # Show baseline comparison
            if not args.quiet and not getattr(args, "json", False):
                baselines = profile.healthy_baselines()
                if baselines:
                    print(f"\n  Healthy baselines for {profile.name}:")
                    for metric, rng in baselines.items():
                        parts = []
                        if "min" in rng:
                            parts.append(f"min={rng['min']}")
                        if "max" in rng:
                            parts.append(f"max={rng['max']}")
                        print(f"    {metric}: {', '.join(parts)}")
                    print()
        else:
            findings = diagnose(state_dict, meta_dict, verbose=verbose)
    else:
        findings = diagnose(state_dict, meta_dict, verbose=verbose)

    # Runtime diagnostics
    runtime_prompts = EXAMPLE_RUNTIME_PROMPTS if getattr(args, "example_prompts", False) else None
    if args.runtime:
        if not args.quiet:
            print("\nRunning runtime diagnostics...")
        try:
            model, tokenizer, device = load_model(args.model, hf=args.hf)
            runtime_findings = diagnose_runtime(model, tokenizer, device, prompts=runtime_prompts)
            findings.extend(runtime_findings)
            del model, tokenizer
        except Exception as e:
            if not args.quiet:
                print(f"  Runtime diagnostics skipped: {e}")

    # Filter
    if args.only:
        allowed = set(args.only.split(","))
        findings = [f for f in findings if f.condition in allowed]

    prescriptions = prescribe(findings)
    report = ExamReport(args.model, meta, findings, prescriptions)

    health = compute_health_score(findings)

    if args.json:
        report_dict = report.to_dict()
        report_dict["health_score"] = {
            "overall": health.overall,
            "grade": health.grade,
            "categories": health.categories,
        }
        print(json.dumps(report_dict, indent=2, default=str))
    else:
        print_exam(findings, prescriptions, explain=getattr(args, "explain", False))
        print_health_score(health)

        # Verdict
        errors = sum(1 for f in findings if f.severity == "ERROR")
        warns = sum(1 for f in findings if f.severity == "WARN")
        print(f"{'='*80}")
        if errors:
            print(f"VERDICT: UNHEALTHY ({errors} errors, {warns} warnings)")
        elif warns:
            print(f"VERDICT: OK with {warns} warning(s)")
        else:
            print(f"VERDICT: HEALTHY")
        print(f"{'='*80}")

    if args.export:
        with open(args.export, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        if not args.quiet:
            print(f"\nExported to {args.export}")


def run_treat(args):
    """Run diagnosis and apply treatments."""
    verbose = getattr(args, "verbose", False) and not args.quiet
    if not args.quiet:
        print(f"Loading: {args.model}")
    state_dict, meta_dict = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict, source=meta_dict.get("source", "unknown"), extra=meta_dict)
    if not args.quiet:
        print(f"Loaded {meta.num_tensors} tensors, {meta.num_params:,} parameters")

    # Diagnose
    findings = diagnose(state_dict, meta_dict, verbose=verbose)

    runtime_prompts = EXAMPLE_RUNTIME_PROMPTS if getattr(args, "example_prompts", False) else None
    if args.runtime:
        try:
            model, tokenizer, device = load_model(args.model, hf=args.hf)
            findings.extend(diagnose_runtime(model, tokenizer, device, prompts=runtime_prompts))
            del model, tokenizer
        except Exception as e:
            if not args.quiet:
                print(f"  Runtime diagnostics skipped: {e}")

    if args.only:
        allowed = set(args.only.split(","))
        findings = [f for f in findings if f.condition in allowed]

    prescriptions = prescribe(findings, conservative=getattr(args, "conservative", False))

    if not args.quiet:
        print_exam(findings, prescriptions, explain=getattr(args, "explain", False))

    if not prescriptions:
        if not args.quiet:
            print("\nNothing to treat.")
        return

    # Before test
    before_score = before_ppl = None
    if getattr(args, "test", False):
        if not args.quiet:
            print(f"\n{'='*80}")
            print("BEFORE TREATMENT")
            print(f"{'='*80}")
        try:
            model, tokenizer, device = load_model(args.model, hf=args.hf)
            before_score, total, results = eval_coherence(model, tokenizer, device)
            before_ppl = eval_perplexity(model, tokenizer, device)
            if not args.quiet:
                print(f"  Coherent: {before_score}/{total}, PPL: {before_ppl}")
                for r in results:
                    tag = "OK" if r["coherent"] else "BAD"
                    print(f"    [{tag}] {r['prompt']}")
                    print(f"         {safe_str(r['response'][:120])}")
            del model, tokenizer
        except Exception as e:
            if not args.quiet:
                print(f"  Testing skipped: {e}")

    # Apply treatments one at a time
    if not args.quiet:
        print(f"\n{'='*80}")
        print("APPLYING TREATMENTS")
        print(f"{'='*80}")

    applied = []
    dry_run = getattr(args, "dry_run", False)

    # Treatment manifest for audit logging
    from model_clinic._manifest import TreatmentManifest
    manifest = TreatmentManifest()

    for i, rx in enumerate(prescriptions):
        if verbose:
            print(f"  [{i+1}/{len(prescriptions)}] Applying {rx.name}...")
        result = apply_treatment(state_dict, rx, dry_run=dry_run)
        if not args.quiet:
            status = "OK" if result.success else "FAIL"
            risk = RISK_SYMBOL.get(rx.risk, rx.risk)
            print(f"  [{status}] Rx #{i+1} {rx.name} {risk}")
            print(f"    {result.description}")
        if not dry_run:
            manifest.record(result, state_dict)
        if result.success:
            applied.append(result)

    if not args.quiet:
        print(f"\n  Applied: {len(applied)}/{len(prescriptions)}")

    # After test + auto-rollback
    if getattr(args, "test", False) and not dry_run and applied:
        if not args.quiet:
            print(f"\n{'='*80}")
            print("AFTER TREATMENT")
            print(f"{'='*80}")
        try:
            model, tokenizer, device = load_model(args.model, hf=args.hf)
            if args.hf:
                model.load_state_dict(state_dict)
            after_score, total, results = eval_coherence(model, tokenizer, device)
            after_ppl = eval_perplexity(model, tokenizer, device)
            if not args.quiet:
                print(f"  Coherent: {after_score}/{total} (was {before_score}/{total})")
                print(f"  PPL: {after_ppl} (was {before_ppl})")
                for r in results:
                    tag = "OK" if r["coherent"] else "BAD"
                    print(f"    [{tag}] {r['prompt']}")
                    print(f"         {safe_str(r['response'][:120])}")

            # Auto-rollback
            if not getattr(args, "no_rollback", False):
                if before_score is not None and after_score < before_score:
                    if not args.quiet:
                        print(f"\n  ROLLING BACK: generation regressed ({before_score} -> {after_score})")
                    for result in reversed(applied):
                        rollback_treatment(state_dict, result)
                    applied = []
                elif before_ppl is not None and after_ppl > before_ppl * 1.2:
                    if not args.quiet:
                        print(f"\n  ROLLING BACK: PPL regressed ({before_ppl} -> {after_ppl})")
                    for result in reversed(applied):
                        rollback_treatment(state_dict, result)
                    applied = []

            del model, tokenizer
        except Exception as e:
            if not args.quiet:
                print(f"  Post-treatment testing skipped: {e}")

    # Save
    if getattr(args, "save", None) and not dry_run and applied:
        patched = save_state_dict(state_dict, args.model, args.save)
        if not args.quiet:
            print(f"\nSaved treated model to {args.save} ({patched} params patched)")

        # Auto-save manifest alongside checkpoint
        manifest_path = getattr(args, "manifest", None) or f"{args.save}.manifest.json"
        manifest.save(manifest_path)
        if not args.quiet:
            manifest.print_summary()
            print(f"Manifest saved to {manifest_path}")

    # Explicit --manifest without --save (save manifest only)
    elif getattr(args, "manifest", None) and not dry_run and applied:
        manifest.save(args.manifest)
        if not args.quiet:
            manifest.print_summary()
            print(f"Manifest saved to {args.manifest}")

    # Export
    if args.export:
        report = ExamReport(args.model, meta, findings, prescriptions,
                            [r for r in applied],
                            before_score, before_ppl)
        with open(args.export, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)
        if not args.quiet:
            print(f"Exported to {args.export}")


def cli_main():
    """Entry point when called via 'model-clinic exam/treat' (subcommand already in sys.argv)."""
    import sys
    command = sys.argv[1] if len(sys.argv) > 1 else "exam"

    # Build parser for this specific subcommand
    parser = argparse.ArgumentParser(
        prog=f"model-clinic {command}",
        description="Model clinic — diagnose, prescribe, treat",
    )
    _add_common_args(parser)

    if command == "treat":
        parser.add_argument("--save", type=str, help="Save treated checkpoint")
        parser.add_argument("--test", action="store_true", help="Run generation tests")
        parser.add_argument("--dry-run", action="store_true", help="Show what would change")
        parser.add_argument("--conservative", action="store_true", help="Only low-risk fixes")
        parser.add_argument("--no-rollback", action="store_true", help="Don't auto-rollback")
        parser.add_argument("--manifest", type=str, default=None,
                            help="Save treatment manifest to path (auto-saved with --save)")

    # Parse everything after the subcommand
    args = parser.parse_args(sys.argv[2:])

    if command == "treat":
        run_treat(args)
    else:
        run_exam(args)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.exam and not args.treat:
        args.exam = True

    if args.exam:
        run_exam(args)
    elif args.treat:
        run_treat(args)


if __name__ == "__main__":
    main()
