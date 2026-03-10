"""Unified model/state_dict loader for model-clinic.

Handles:
- HuggingFace models (by name or local path)
- PyTorch .pt/.pth checkpoints (various formats)
- Safetensors files
- Composite checkpoints (multiple nested state dicts)
"""

import sys
from pathlib import Path

import torch

from model_clinic._types import ModelMeta
from model_clinic._utils import device_auto, infer_model_shape


def _is_hf_model(path):
    """Auto-detect if path is a HuggingFace model (directory or hub name)."""
    p = Path(path)
    if p.is_dir():
        # HF local dir has config.json
        return (p / "config.json").exists()
    # Hub name: contains "/" but isn't a file path
    if "/" in str(path) and not p.exists():
        return True
    return False


def _is_safetensors(path):
    """Check if path is a safetensors file."""
    return str(path).endswith(".safetensors")


def _load_safetensors(path):
    """Load safetensors file."""
    try:
        from safetensors.torch import load_file
        return load_file(path), {}
    except ImportError:
        raise ImportError("Install safetensors: pip install safetensors")


def _extract_composite_params(checkpoint):
    """Extract params from composite checkpoint with multiple state dicts."""
    params = {}
    for section in ["memory_state_dict", "bridge_state_dict", "act_state_dict",
                    "drives_state_dict", "state_injection_state_dict",
                    "qwen_lora_state_dict"]:
        if section in checkpoint:
            for k, v in checkpoint[section].items():
                if isinstance(v, torch.Tensor):
                    params[f"{section}/{k}"] = v

    if "wrapper_state_dict" in checkpoint:
        ws = checkpoint["wrapper_state_dict"]
        for sub in ["layers", "norm"]:
            if sub in ws and isinstance(ws[sub], dict):
                for k, v in ws[sub].items():
                    if isinstance(v, torch.Tensor):
                        params[f"wrapper/{sub}/{k}"] = v
        for k in ["gate", "pre_memory_gate", "state_gate"]:
            if k in ws and isinstance(ws[k], torch.Tensor):
                params[f"wrapper/{k}"] = ws[k]

    if "persistent_state" in checkpoint:
        params["persistent_state"] = checkpoint["persistent_state"]

    return params, checkpoint.get("extra", {})


def _extract_generic_params(checkpoint):
    """Extract params from generic checkpoint dict."""
    # Try known section keys
    for key in ["model_state_dict", "state_dict"]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key], {}

    # Flatten: top-level tensors + one level of nested dicts
    sd = {}
    for k, v in checkpoint.items():
        if isinstance(v, torch.Tensor):
            sd[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    sd[f"{k}/{k2}"] = v2
    return sd, {}


def load_state_dict(path, hf=False):
    """Load parameters from any supported format.

    Returns:
        (state_dict, meta_dict): Flat dict of name->Tensor, plus metadata.
    """
    # Auto-detect HF
    if hf or _is_hf_model(path):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32
        )
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        del model
        return sd, {"source": "huggingface"}

    # Safetensors
    if _is_safetensors(path):
        sd, meta = _load_safetensors(path)
        return sd, {"source": "safetensors", **meta}

    # Check file exists before trying to load
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # PyTorch checkpoint — try weights_only=True first (safer, avoids segfaults on Python 3.14+)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Composite format (multiple named state dicts)
    if isinstance(checkpoint, dict) and checkpoint.get("model_type") in ("ATLESQwen", "composite"):
        params, extra = _extract_composite_params(checkpoint)
        return params, {"source": "composite", **extra}

    # Generic checkpoint
    if isinstance(checkpoint, dict):
        params, extra = _extract_generic_params(checkpoint)
        return params, {"source": "checkpoint", **extra}

    raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")


def load_model(path, hf=False, device=None):
    """Load a full model + tokenizer for generation testing.

    Returns:
        (model, tokenizer, device_str)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or device_auto()

    if hf or _is_hf_model(path):
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float32
        ).to(device)
        model.eval()
        return model, tokenizer, device

    raise RuntimeError(
        f"Cannot load {path} as a runnable model. "
        f"Use --hf for HuggingFace models, or ensure the model architecture is available."
    )


def build_meta(state_dict, source="unknown", extra=None):
    """Build ModelMeta from a state dict."""
    extra = extra or {}
    hidden, layers, vocab = infer_model_shape(state_dict)
    num_params = sum(t.numel() for t in state_dict.values() if isinstance(t, torch.Tensor))
    num_tensors = sum(1 for t in state_dict.values() if isinstance(t, torch.Tensor))

    return ModelMeta(
        source=source,
        num_params=num_params,
        num_tensors=num_tensors,
        hidden_size=hidden,
        num_layers=layers,
        vocab_size=vocab,
        extra=extra,
    )


def save_state_dict(state_dict, original_path, output_path):
    """Save modified state dict back into the original checkpoint format.

    Reloads the original checkpoint and patches in modified tensors.
    """
    if _is_hf_model(original_path):
        torch.save(state_dict, output_path)
        return

    try:
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(original_path, map_location="cpu", weights_only=False)
    patched = 0

    for name, tensor in state_dict.items():
        # Direct top-level tensor
        if name in checkpoint and isinstance(checkpoint[name], torch.Tensor):
            checkpoint[name] = tensor
            patched += 1
            continue

        # Inside known dict sections
        for section_key in ["model_state_dict", "state_dict"]:
            if section_key in checkpoint and isinstance(checkpoint[section_key], dict):
                if name in checkpoint[section_key]:
                    checkpoint[section_key][name] = tensor
                    patched += 1
                    break

        # Composite "section/key" format
        parts = name.split("/", 1)
        if len(parts) == 2 and parts[0] in checkpoint:
            section = checkpoint[parts[0]]
            if isinstance(section, dict):
                subparts = parts[1].split("/", 1)
                if len(subparts) == 2 and subparts[0] in section:
                    if isinstance(section[subparts[0]], dict):
                        section[subparts[0]][subparts[1]] = tensor
                        patched += 1
                    else:
                        section[parts[1]] = tensor
                        patched += 1
                else:
                    section[parts[1]] = tensor
                    patched += 1

    torch.save(checkpoint, output_path)
    return patched
