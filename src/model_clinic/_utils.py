"""Shared utilities for model-clinic."""

import torch


def device_auto():
    """Pick best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def to_float(tensor):
    """Safe cast to float32 for analysis."""
    return tensor.float()


def safe_str(s):
    """Make string safe for terminal output (ASCII only)."""
    return s.encode("ascii", errors="replace").decode("ascii")


def find_param(model_or_sd, pattern):
    """Find parameters matching a name pattern.

    Works on both nn.Module (named_parameters) and state_dict (dict).
    """
    matches = []
    if isinstance(model_or_sd, dict):
        for name, tensor in model_or_sd.items():
            if isinstance(tensor, torch.Tensor) and pattern in name:
                matches.append((name, tensor))
    else:
        for name, param in model_or_sd.named_parameters():
            if pattern in name:
                matches.append((name, param))
        for name, buf in model_or_sd.named_buffers():
            if pattern in name:
                matches.append((name, buf))
    return matches


def infer_model_shape(state_dict):
    """Infer hidden_size, num_layers, vocab_size from state dict keys/shapes."""
    hidden_size = 0
    num_layers = 0
    vocab_size = 0

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        # Hidden size from embedding or projection
        if "embed" in name and tensor.dim() == 2:
            vocab_size = tensor.shape[0]
            hidden_size = tensor.shape[1]

        # Layer count from layer indices
        if "layers." in name or "layer." in name:
            parts = name.split(".")
            for i, p in enumerate(parts):
                if p in ("layers", "layer") and i + 1 < len(parts):
                    try:
                        idx = int(parts[i + 1])
                        num_layers = max(num_layers, idx + 1)
                    except ValueError:
                        pass

        # Hidden size fallback from q_proj or similar
        if hidden_size == 0 and "q_proj" in name and tensor.dim() == 2:
            hidden_size = tensor.shape[1]

    return hidden_size, num_layers, vocab_size
