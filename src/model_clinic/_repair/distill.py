"""Targeted re-initialization with knowledge distillation (Level 3 repair).

When a module subtree is truly dead — identical rows, gradient noise at ERROR
level, model aging — patching individual values won't help. This module resets
dead subtrees and uses the model's own working layers as a teacher to bring
them back to life.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from model_clinic._types import Finding


# Conditions that indicate a dead module (need 2+ to trigger)
_DEAD_CONDITIONS = frozenset({"identical_rows", "gradient_noise", "model_aging"})


@dataclass
class DistillReport:
    """Summary of a distillation repair run."""

    dead_modules: List[str]
    steps_run: int
    loss_start: float
    loss_end: float
    params_reset: int
    params_total: int


def identify_dead_modules(findings: Sequence[Finding]) -> List[str]:
    """Identify module subtrees that are dead based on diagnostic findings.

    A module subtree is considered dead when it has 2 or more of:
    - identical_rows
    - gradient_noise (ERROR level only)
    - model_aging

    Findings are grouped by module prefix. For example,
    ``internal_memory.tiers.0.keys`` and ``internal_memory.tiers.0.values``
    both belong to the prefix ``internal_memory.tiers.0``.

    Args:
        findings: List of Finding objects from ``diagnose()``.

    Returns:
        Sorted list of unique module prefixes that are dead.
    """
    # Collect relevant findings grouped by module prefix
    prefix_conditions: Dict[str, set] = defaultdict(set)

    for f in findings:
        if f.condition not in _DEAD_CONDITIONS:
            continue
        # gradient_noise only counts at ERROR level
        if f.condition == "gradient_noise" and f.severity != "ERROR":
            continue

        prefix = _module_prefix(f.param_name)
        prefix_conditions[prefix].add(f.condition)

    # A prefix is dead when it has 2+ distinct dead conditions
    dead = [
        prefix
        for prefix, conds in prefix_conditions.items()
        if len(conds) >= 2
    ]

    # Remove redundant prefixes (if both "a" and "a.b" are dead, keep "a")
    dead.sort(key=len)
    filtered: List[str] = []
    for d in dead:
        if not any(d.startswith(parent + ".") for parent in filtered):
            filtered.append(d)

    return sorted(filtered)


def _module_prefix(param_name: str) -> str:
    """Extract the module prefix from a parameter name.

    ``"internal_memory.tiers.0.keys.weight"`` -> ``"internal_memory.tiers.0"``

    Heuristic: drop the last two dotted segments (param type + leaf attribute).
    If that leaves nothing, fall back to dropping just the last segment.
    """
    parts = param_name.split(".")
    if len(parts) >= 3:
        return ".".join(parts[:-2])
    elif len(parts) >= 2:
        return parts[0]
    return param_name


def reset_module_params(
    state_dict: Dict[str, torch.Tensor], module_prefix: str
) -> Dict[str, torch.Tensor]:
    """Re-initialize all parameters under a module prefix.

    - 2-D+ float tensors: Xavier uniform
    - 1-D float tensors (biases): zeros
    - Non-float tensors (int counters, pointers): left unchanged

    Args:
        state_dict: Model state dict (modified in-place and returned).
        module_prefix: Dotted module prefix, e.g. ``"memory.tiers.0"``.

    Returns:
        The modified state_dict.
    """
    prefix_dot = module_prefix + "."
    for name, tensor in state_dict.items():
        if not (name == module_prefix or name.startswith(prefix_dot)):
            continue
        if not tensor.is_floating_point():
            continue
        if tensor.dim() >= 2:
            nn.init.xavier_uniform_(tensor)
        else:
            tensor.zero_()
    return state_dict


def distill_repair(
    model: nn.Module,
    dead_modules: List[str],
    calibration_loader: List[torch.Tensor],
    num_steps: int = 200,
    lr: float = 1e-4,
    device: str = "cpu",
) -> nn.Module:
    """Knowledge-distillation repair: reset dead modules and retrain them.

    Steps:
    1. Capture teacher activations from non-dead (working) modules.
    2. Reset dead modules with Xavier/zeros init.
    3. Freeze all parameters except those in dead modules.
    4. Train dead modules to minimize MSE between student and teacher
       activations at the model's output.
    5. Return the repaired model.

    Args:
        model: A ``torch.nn.Module`` instance. Modified in-place.
        dead_modules: List of dead module name prefixes (from
            ``identify_dead_modules``). If empty, the model is returned as-is.
        calibration_loader: List of input tensors. Each tensor is fed to the
            model's forward method directly.
        num_steps: Number of optimization steps.
        lr: Learning rate for AdamW.
        device: Device to use (``'cpu'`` or ``'cuda'``).

    Returns:
        The repaired model (same object, modified in-place).
    """
    if not dead_modules or not calibration_loader:
        return model

    model = model.to(device)
    model.eval()

    # --- 1. Capture teacher activations --------------------------------
    teacher_outputs = _capture_outputs(model, calibration_loader, device)

    # --- 2. Reset dead module parameters -------------------------------
    sd = model.state_dict()
    params_reset = 0
    params_total = sum(p.numel() for p in model.parameters())

    for dm in dead_modules:
        prefix_dot = dm + "."
        for name, tensor in sd.items():
            if not (name == dm or name.startswith(prefix_dot)):
                continue
            if not tensor.is_floating_point():
                continue
            params_reset += tensor.numel()

    for dm in dead_modules:
        reset_module_params(sd, dm)

    model.load_state_dict(sd)
    model = model.to(device)

    # --- 3. Freeze everything except dead modules ----------------------
    for name, param in model.named_parameters():
        param.requires_grad = _is_under_dead(name, dead_modules)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        # Nothing to train — all dead params might be non-float / no params
        return model

    # --- 4. Distillation loop ------------------------------------------
    model.train()
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    loss_fn = nn.MSELoss()

    loss_start = None
    loss_end = None

    for step in range(num_steps):
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for i, inp in enumerate(calibration_loader):
            inp = inp.to(device)
            student_out = model(inp)

            # Handle tuple/dict outputs — take the first tensor
            student_tensor = _extract_tensor(student_out)
            teacher_tensor = teacher_outputs[i].to(device)

            # Align shapes if needed (teacher may differ after reset)
            if student_tensor.shape != teacher_tensor.shape:
                min_len = min(student_tensor.shape[0], teacher_tensor.shape[0])
                student_tensor = student_tensor[:min_len]
                teacher_tensor = teacher_tensor[:min_len]

            loss = loss_fn(student_tensor, teacher_tensor)
            total_loss = total_loss + loss
            count += 1

        if count == 0:
            break

        avg_loss = total_loss / count

        if loss_start is None:
            loss_start = avg_loss.item()

        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()

        loss_end = avg_loss.item()

    # --- 5. Clean up ---------------------------------------------------
    # Unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    return model


def _is_under_dead(param_name: str, dead_modules: List[str]) -> bool:
    """Check if a parameter belongs to any dead module prefix."""
    for dm in dead_modules:
        if param_name == dm or param_name.startswith(dm + "."):
            return True
    return False


def _capture_outputs(
    model: nn.Module,
    calibration_loader: List[torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    """Run calibration data through the model and capture outputs."""
    outputs = []
    with torch.no_grad():
        for inp in calibration_loader:
            inp = inp.to(device)
            out = model(inp)
            tensor = _extract_tensor(out)
            outputs.append(tensor.detach().cpu())
    return outputs


def _extract_tensor(output) -> torch.Tensor:
    """Extract a tensor from a model output (handles tuples, dicts, etc.)."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
        raise ValueError("No tensor found in model output tuple/list")
    if isinstance(output, dict):
        for key in ("logits", "last_hidden_state", "output"):
            if key in output and isinstance(output[key], torch.Tensor):
                return output[key]
        # Fall back to first tensor value
        for v in output.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError("No tensor found in model output dict")
    raise TypeError(f"Unsupported model output type: {type(output)}")
