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

from model_clinic._types import Finding, Prescription, TreatmentResult, ExamReport
from model_clinic._loader import load_state_dict, load_model, build_meta, save_state_dict
from model_clinic._eval import (
    eval_coherence, eval_perplexity, eval_logit_entropy, eval_diversity,
    DEFAULT_PROMPTS,
)
from model_clinic._utils import safe_str, to_float


# ── Thresholds & constants ────────────────────────────────────────────────

DEAD_NEURON_THRESHOLD = 1e-7        # Norm below this = dead
GATE_CLOSED_THRESHOLD = 0.01       # sigmoid below this = stuck closed
GATE_OPEN_THRESHOLD = 0.99         # sigmoid above this = stuck open
EXPLODING_NORM_THRESHOLD = 10.0    # Per-element norm above this = exploding
VANISHING_NORM_THRESHOLD = 1e-6    # Per-element norm below this = vanishing
KURTOSIS_THRESHOLD = 50            # Kurtosis above this = heavy tails
NORM_DRIFT_THRESHOLD = 0.5         # |mean - 1.0| above this = drifted
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

    def detect_all(self, state_dict, context=None):
        """Run all detectors on a state dict."""
        context = context or {}
        findings = []
        for condition, detector in self._detectors.items():
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


# ── Register all detectors ─────────────────────────────────────────────────

def _rx_dead_neurons(f):
    return Prescription(
        name="reinit_dead_neurons",
        description=f"Reinit {f.details['dead_count']} dead {f.details['dim']} in {f.param_name}",
        risk="low", finding=f, action="reinit_dead",
        params={"indices": f.details["dead_indices"], "dim": f.details["dim"]},
    )

def _rx_nudge_gate(f):
    return Prescription(
        name="nudge_gate_open",
        description=f"Nudge {f.param_name} from {f.details['sigmoid']:.4f} toward 4.7%",
        risk="medium", finding=f, action="set_gate",
        params={"value": -3.0},
    )

def _rx_pull_gate(f):
    return Prescription(
        name="pull_gate_back",
        description=f"Pull {f.param_name} from {f.details['sigmoid']:.4f} back to 95%",
        risk="medium", finding=f, action="set_gate",
        params={"value": 3.0},
    )

def _rx_scale_norm(f):
    return Prescription(
        name="scale_down_norm",
        description=f"Scale {f.param_name} norm from {f.details['per_elem_norm']:.2f} toward 1.0",
        risk="medium", finding=f, action="scale_norm",
        params={"target_per_elem": 1.0},
    )

def _rx_reinit_vanishing(f):
    return Prescription(
        name="reinit_vanishing",
        description=f"Reinit {f.param_name} (norm {f.details['per_elem_norm']:.2e})",
        risk="low", finding=f, action="reinit_full",
    )

def _rx_clamp_tails(f):
    return Prescription(
        name="clamp_tails",
        description=f"Clamp {f.param_name} outliers (kurtosis={f.details['kurtosis']:.0f})",
        risk="medium", finding=f, action="clamp_outliers",
        params={"sigma": CLAMP_SIGMA},
    )

def _rx_reset_norm(f):
    return Prescription(
        name="reset_norm",
        description=f"Reset {f.param_name} mean from {f.details['mean']:.3f} toward 1.0",
        risk="low", finding=f, action="reset_norm_weights",
    )

def _rx_desaturate(f):
    return Prescription(
        name="desaturate",
        description=f"Desaturate {f.param_name} ({f.details['near_max_pct']:.0%} at boundary)",
        risk="medium", finding=f, action="desaturate",
        params={"factor": 0.8},
    )

def _rx_fix_nan(f):
    return Prescription(
        name="fix_nan_inf",
        description=f"Zero out {f.details['nan_count']} NaN + {f.details['inf_count']} Inf in {f.param_name}",
        risk="high", finding=f, action="fix_nan_inf",
    )

def _rx_perturb_identical(f):
    return Prescription(
        name="perturb_identical_rows",
        description=f"Add noise to {f.details['duplicate_pairs']} duplicate row pairs in {f.param_name}",
        risk="low", finding=f, action="perturb_identical",
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


# ── Diagnosis engine ───────────────────────────────────────────────────────

def diagnose(state_dict, meta=None):
    """Run all diagnostic checks. Returns list of Findings."""
    context = {"meta": meta or {}}
    findings = REGISTRY.detect_all(state_dict, context)

    # Post-scan detectors
    findings.extend(post_detect_dtype_mismatch(context))
    findings.extend(post_detect_attention_imbalance(context))

    return findings


def prescribe(findings, conservative=False):
    """Map findings to prescriptions."""
    return REGISTRY.prescribe(findings, conservative=conservative)


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
    }
    fmt = formatters.get(f.condition, lambda: f"    {f.param_name}: {d}")
    print(fmt())


def print_exam(findings, prescriptions):
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
        return parser


def _add_common_args(parser):
    parser.add_argument("model", help="Path to .pt checkpoint or HF model name")
    parser.add_argument("--hf", action="store_true", help="Load as HuggingFace model")
    parser.add_argument("--runtime", action="store_true", help="Include runtime diagnostics")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated conditions")
    parser.add_argument("--export", type=str, default=None, help="Export report to JSON")
    parser.add_argument("--json", action="store_true", help="Output as JSON only")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")


def run_exam(args):
    """Run examination (diagnosis only)."""
    if getattr(args, "json", False):
        args.quiet = True
    if not args.quiet:
        print(f"Loading: {args.model}")
    state_dict, meta_dict = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict, source=meta_dict.get("source", "unknown"), extra=meta_dict)
    if not args.quiet:
        print(f"Loaded {meta.num_tensors} tensors, {meta.num_params:,} parameters")

    findings = diagnose(state_dict, meta_dict)

    # Runtime diagnostics
    if args.runtime:
        if not args.quiet:
            print("\nRunning runtime diagnostics...")
        try:
            model, tokenizer, device = load_model(args.model, hf=args.hf)
            runtime_findings = diagnose_runtime(model, tokenizer, device)
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

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        print_exam(findings, prescriptions)

        # Verdict
        errors = sum(1 for f in findings if f.severity == "ERROR")
        warns = sum(1 for f in findings if f.severity == "WARN")
        print(f"\n{'='*80}")
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
    if not args.quiet:
        print(f"Loading: {args.model}")
    state_dict, meta_dict = load_state_dict(args.model, hf=args.hf)
    meta = build_meta(state_dict, source=meta_dict.get("source", "unknown"), extra=meta_dict)
    if not args.quiet:
        print(f"Loaded {meta.num_tensors} tensors, {meta.num_params:,} parameters")

    # Diagnose
    findings = diagnose(state_dict, meta_dict)

    if args.runtime:
        try:
            model, tokenizer, device = load_model(args.model, hf=args.hf)
            findings.extend(diagnose_runtime(model, tokenizer, device))
            del model, tokenizer
        except Exception as e:
            if not args.quiet:
                print(f"  Runtime diagnostics skipped: {e}")

    if args.only:
        allowed = set(args.only.split(","))
        findings = [f for f in findings if f.condition in allowed]

    prescriptions = prescribe(findings, conservative=getattr(args, "conservative", False))

    if not args.quiet:
        print_exam(findings, prescriptions)

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

    for i, rx in enumerate(prescriptions):
        result = apply_treatment(state_dict, rx, dry_run=dry_run)
        if not args.quiet:
            status = "OK" if result.success else "FAIL"
            risk = RISK_SYMBOL.get(rx.risk, rx.risk)
            print(f"  [{status}] Rx #{i+1} {rx.name} {risk}")
            print(f"    {result.description}")
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
