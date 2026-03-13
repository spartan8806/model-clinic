"""Architecture profiles for model-clinic.

Curated diagnostic bundles per model family with known-healthy baselines.
Each profile selects the relevant detectors, defines expected metric ranges,
and lists false-positive suppressions so that findings are architecture-aware.

Usage:
    from model_clinic._profiles import get_profile, auto_detect_profile

    profile = get_profile("llm")
    findings = profile.diagnose(state_dict)
    print(profile.describe())
    baselines = profile.healthy_baselines()

    # Or auto-detect from tensor names:
    profile = auto_detect_profile(state_dict)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

from model_clinic._types import Finding


@dataclass
class ArchProfile:
    """Architecture-specific diagnostic profile."""
    name: str
    description: str
    key_layers: list = field(default_factory=list)       # layer name patterns to focus on
    detectors: list = field(default_factory=list)         # which conditions to run
    baselines: dict = field(default_factory=dict)         # expected healthy ranges per metric
    warnings: list = field(default_factory=list)          # things normal for this arch (suppress)
    notes: list = field(default_factory=list)             # additional context / reference info

    def diagnose(self, state_dict, meta=None, verbose=False):
        """Run profile-specific diagnosis on a state dict.

        Only runs the detectors listed in self.detectors, then filters out
        any findings whose conditions appear in self.warnings (false positives).
        """
        from model_clinic.clinic import REGISTRY, post_detect_dtype_mismatch, \
            post_detect_attention_imbalance, post_detect_head_redundancy, \
            post_detect_representation_drift, post_detect_model_aging, \
            _is_metadata_tensor

        context = {"meta": meta or {}}
        context.setdefault("_all_tensor_names", set(state_dict.keys()))

        allowed = set(self.detectors)
        findings = []

        # Run per-tensor detectors that match our allowed set
        seen_detectors = set()
        for condition, detector in REGISTRY._detectors.items():
            if condition.startswith("_"):
                # Always run collector detectors (e.g. _collect_layer_norms)
                pass
            elif condition not in allowed:
                continue
            detector_id = id(detector)
            if detector_id in seen_detectors:
                continue
            seen_detectors.add(detector_id)
            if verbose:
                print(f"  [{self.name}] Checking {condition}...")
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if _is_metadata_tensor(name, tensor):
                    continue
                results = detector(name, tensor, context)
                if results:
                    findings.extend(results)

        # Post-scan detectors (only if relevant conditions are in our set)
        _post_detectors = {
            "dtype_mismatch": post_detect_dtype_mismatch,
            "attention_imbalance": post_detect_attention_imbalance,
            "head_redundancy": post_detect_head_redundancy,
            "representation_drift": post_detect_representation_drift,
            "model_aging": post_detect_model_aging,
        }
        for cond, post_fn in _post_detectors.items():
            if cond in allowed:
                if verbose:
                    print(f"  [{self.name}] Post-check {cond}...")
                findings.extend(post_fn(context))

        # Filter out suppressed conditions (false positives for this arch)
        suppressed = set(self.warnings)
        findings = [f for f in findings if f.condition not in suppressed]

        return findings

    def describe(self) -> str:
        """Return a human-readable description of this profile."""
        lines = [
            f"Architecture Profile: {self.name}",
            f"  {self.description}",
            "",
            f"  Key layers: {', '.join(self.key_layers)}",
            f"  Detectors ({len(self.detectors)}): {', '.join(self.detectors)}",
        ]
        if self.warnings:
            lines.append(f"  Suppressed (normal for {self.name}): {', '.join(self.warnings)}")
        if self.notes:
            lines.append("")
            for note in self.notes:
                lines.append(f"  Note: {note}")
        return "\n".join(lines)

    def healthy_baselines(self) -> dict:
        """Return expected healthy metric ranges for this architecture."""
        return dict(self.baselines)


# ---- Profile Definitions -----------------------------------------------

_LLM_PROFILE = ArchProfile(
    name="llm",
    description="Large Language Model profile (GPT, LLaMA, Qwen, Mistral, etc.)",
    key_layers=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "layernorm", "rmsnorm", "input_layernorm", "post_attention_layernorm",
    ],
    detectors=[
        "dead_neurons",
        "exploding_norm",
        "vanishing_norm",
        "norm_drift",
        "head_redundancy",
        "identical_rows",
        "heavy_tails",
        "gradient_noise",
        "nan_inf",
        "stuck_gate_closed",
        "stuck_gate_open",
    ],
    baselines={
        "per_element_norm": {"min": 0.01, "max": 5.0},
        "norm_weight_mean": {"min": 0.8, "max": 1.5},
        "kurtosis": {"max": 50},
        "dead_neuron_fraction": {"max": 0.05},
        "head_cosine_similarity": {"max": 0.95},
    },
    warnings=[
        "quantization_degradation",  # bf16 is standard in LLMs, not a defect
    ],
    notes=[
        "Qwen2.5-0.5B-Instruct baseline: 61/D with 15 findings",
        "bf16 is the standard dtype for modern LLMs -- quantization_degradation is suppressed",
    ],
)

_VIT_PROFILE = ArchProfile(
    name="vit",
    description="Vision Transformer profile (ViT, DeiT, BEiT, etc.)",
    key_layers=[
        "patch_embed", "cls_token", "pos_embed",
        "attn", "qkv", "proj",
        "mlp", "fc1", "fc2",
        "norm", "layernorm",
    ],
    detectors=[
        "dead_neurons",
        "exploding_norm",
        "attention_imbalance",
        "head_redundancy",
        "saturated_weights",
        "nan_inf",
        "identical_rows",
    ],
    baselines={
        "per_element_norm": {"min": 0.01, "max": 8.0},  # ViT norms run larger
        "norm_weight_mean": {"min": 0.5, "max": 2.0},
        "patch_embed_norm_cv": {"max": 0.5},  # patch embedding norms should be tight
        "head_cosine_similarity": {"max": 0.90},  # heads should be diverse
    },
    warnings=[],
    notes=[
        "ViT norms are typically larger than LLM norms",
        "Patch embedding norms should be tightly distributed",
        "Attention heads should show diverse patterns across spatial positions",
    ],
)

_DIFFUSION_PROFILE = ArchProfile(
    name="diffusion",
    description="Diffusion model profile (Stable Diffusion, SDXL, etc.)",
    key_layers=[
        "conv", "conv_in", "conv_out",
        "time_embed", "time_emb", "timestep_embedding",
        "attn", "attention", "to_q", "to_k", "to_v",
        "norm", "group_norm", "groupnorm",
    ],
    detectors=[
        "dead_neurons",
        "exploding_norm",
        "vanishing_norm",
        "nan_inf",
        "norm_drift",
        "identical_rows",
    ],
    baselines={
        "per_element_norm": {"min": 0.001, "max": 10.0},  # conv weights vary more
        "norm_weight_mean": {"min": 0.5, "max": 2.0},
        "time_embed_diversity": {"min": 0.1},  # time embeddings should be diverse
    },
    warnings=[
        "quantization_degradation",  # many diffusion models are fp16
    ],
    notes=[
        "Conv weight norms vary more than linear layers",
        "Time embedding should have diverse representations across timesteps",
        "fp16 is standard for diffusion inference -- quantization findings suppressed",
    ],
)

# ---- Registry -----------------------------------------------------------

_PROFILES = {
    "llm": _LLM_PROFILE,
    "vit": _VIT_PROFILE,
    "diffusion": _DIFFUSION_PROFILE,
}


def get_profile(name: str) -> ArchProfile:
    """Get a named architecture profile.

    Args:
        name: One of "llm", "vit", "diffusion".

    Returns:
        ArchProfile for the requested architecture.

    Raises:
        ValueError: If the profile name is not recognized.
    """
    if name not in _PROFILES:
        available = ", ".join(sorted(_PROFILES.keys()))
        raise ValueError(f"Unknown profile: {name!r}. Available: {available}")
    return _PROFILES[name]


def auto_detect_profile(state_dict) -> Optional[ArchProfile]:
    """Heuristically detect the architecture profile from tensor names.

    Checks tensor names for architecture-specific patterns:
    - LLM: self_attn, q_proj, k_proj, gate_proj, lm_head
    - ViT: patch_embed, cls_token, pos_embed
    - Diffusion: time_embed, unet, conv_in, timestep

    Returns:
        Matching ArchProfile, or None if no match.
    """
    names = set(state_dict.keys())
    names_lower = {n.lower() for n in names}
    joined = " ".join(names_lower)

    # Score each profile by keyword matches
    _KEYWORDS = {
        "vit": ["patch_embed", "cls_token", "pos_embed"],
        "diffusion": ["time_embed", "time_emb", "unet", "conv_in", "timestep"],
        "llm": ["self_attn", "q_proj", "k_proj", "gate_proj", "lm_head",
                 "o_proj", "up_proj", "down_proj"],
    }

    scores = {}
    for profile_name, keywords in _KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in joined)
        scores[profile_name] = score

    # Return the highest-scoring profile if it matched at least 2 keywords
    best = max(scores, key=scores.get)
    if scores[best] >= 2:
        return _PROFILES[best]

    return None


def list_profiles() -> list:
    """Return list of available profile names."""
    return sorted(_PROFILES.keys())
