# model-clinic Plugin Guide

This guide explains how to create plugins that extend model-clinic with custom
detectors, prescribers, and fixes.

## Overview

model-clinic uses Python's setuptools entry points for plugin discovery. When a
plugin package is installed, model-clinic automatically finds and loads it. No
configuration or manual registration is needed on the user's side.

## Quick Start

Generate a starter plugin with the built-in scaffold command:

```bash
model-clinic new-plugin my_detector_pack
cd my_detector_pack
pip install -e .
model-clinic plugins   # verify it's detected
```

This creates a complete package with entry points, an example detector, and tests.

## How Plugins Work

1. Your plugin declares an entry point in `pyproject.toml` under the group
   `model_clinic.plugins`.
2. The entry point references a `register(registry)` function.
3. When `diagnose()` runs, model-clinic calls `load_plugins()`, which discovers
   all installed entry points, calls `register(registry)` for each, and your
   detectors become part of the exam.

### Entry Point Format

In your plugin's `pyproject.toml`:

```toml
[project.entry-points."model_clinic.plugins"]
my_plugin = "my_package.clinic_plugin:register"
```

- `my_plugin` is the plugin name (shown in `model-clinic plugins` output).
- `"my_package.clinic_plugin:register"` is the dotted path to the register function.

## Writing a Detector

A detector is a function that inspects a single parameter tensor and returns
a list of `Finding` objects (or an empty list if healthy).

```python
from model_clinic._types import Finding

def detect_spike(name, tensor, ctx):
    """Flag tensors where any value exceeds 1000."""
    if tensor.dim() < 2:
        return []
    max_val = tensor.float().abs().max().item()
    if max_val > 1000:
        return [Finding(
            condition="value_spike",
            severity="ERROR",
            param_name=name,
            details={"max_value": max_val},
        )]
    return []
```

### Function Signature

```python
def detector(name: str, tensor: torch.Tensor, ctx: dict) -> list[Finding]:
```

- `name` -- parameter name, e.g. `"model.layers.0.self_attn.q_proj.weight"`.
- `tensor` -- the parameter tensor (on CPU).
- `ctx` -- shared context dict. Contains `"meta"` (model metadata) and
  `"_all_tensor_names"` (set of all parameter names in the checkpoint).

### Severity Levels

- `"ERROR"` -- something is definitely broken and will affect outputs.
- `"WARN"` -- potential issue, may degrade quality.
- `"INFO"` -- notable but not necessarily harmful.

## Writing a Prescriber

A prescriber creates a `Prescription` for a `Finding`. It tells model-clinic
what fix to apply.

```python
from model_clinic._types import Prescription

def prescribe_spike(finding):
    """Clamp the spiking tensor to a safe range."""
    return Prescription(
        name="clamp_spike",
        description="Clamp extreme values to [-100, 100]",
        risk="low",
        finding=finding,
        action="clamp",
        params={"min": -100, "max": 100},
        explanation="Extreme weight values cause numerical instability during inference.",
    )
```

### Risk Levels

- `"low"` -- safe to apply automatically. Applied even in `--conservative` mode.
- `"medium"` -- likely helpful but may change model behavior.
- `"high"` -- significant change, only applied when explicitly requested.

## Registering with the Registry

The `register()` function is the entry point that model-clinic calls. It
receives a `ConditionRegistry` instance.

```python
def register(registry):
    """Register custom detectors and prescribers."""
    registry.register(
        condition="value_spike",
        detector=detect_spike,
        prescriber=prescribe_spike,  # optional
        risk="low",
        description="Detect extreme weight values (>1000)",
    )

    # You can register multiple conditions
    registry.register(
        condition="dead_bias",
        detector=detect_dead_bias,
        # No prescriber -- detection only
    )
```

### `registry.register()` Parameters

| Parameter     | Required | Description                                     |
|---------------|----------|-------------------------------------------------|
| `condition`   | Yes      | Unique condition name (e.g. `"value_spike"`)    |
| `detector`    | Yes      | Detector function                               |
| `prescriber`  | No       | Prescriber function (omit for detection-only)   |
| `risk`        | No       | `"low"`, `"medium"`, or `"high"` (default: `"medium"`) |
| `description` | No       | Human-readable description of the condition     |

## Complete Example Plugin

```
my_safety_checks/
  pyproject.toml
  src/my_safety_checks/
    __init__.py
    clinic_plugin.py
  tests/
    test_plugin.py
```

### pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my_safety_checks"
version = "1.0.0"
dependencies = ["model-clinic>=0.3.0"]

[project.entry-points."model_clinic.plugins"]
my_safety_checks = "my_safety_checks.clinic_plugin:register"

[tool.hatch.build.targets.wheel]
packages = ["src/my_safety_checks"]
```

### clinic_plugin.py

```python
"""Safety-focused detectors for model-clinic."""

import torch
from model_clinic._types import Finding, Prescription


def register(registry):
    registry.register(
        condition="extreme_weights",
        detector=detect_extreme_weights,
        prescriber=prescribe_extreme_weights,
        risk="low",
        description="Detect weights with values beyond safe range",
    )
    registry.register(
        condition="suspicious_bias",
        detector=detect_suspicious_bias,
        risk="medium",
        description="Detect bias tensors with abnormal distributions",
    )


def detect_extreme_weights(name, tensor, ctx):
    if tensor.dim() < 2:
        return []
    max_abs = tensor.float().abs().max().item()
    if max_abs > 500:
        return [Finding(
            condition="extreme_weights",
            severity="ERROR" if max_abs > 5000 else "WARN",
            param_name=name,
            details={"max_abs": max_abs},
        )]
    return []


def prescribe_extreme_weights(finding):
    max_abs = finding.details.get("max_abs", 500)
    clamp_to = min(max_abs * 0.1, 100)
    return Prescription(
        name="clamp_extreme_weights",
        description=f"Clamp values to [-{clamp_to}, {clamp_to}]",
        risk="low",
        finding=finding,
        action="clamp",
        params={"min": -clamp_to, "max": clamp_to},
        explanation="Extreme weights cause overflow during matrix multiplication.",
    )


def detect_suspicious_bias(name, tensor, ctx):
    if "bias" not in name.lower() or tensor.dim() != 1:
        return []
    mean = tensor.float().mean().item()
    if abs(mean) > 10:
        return [Finding(
            condition="suspicious_bias",
            severity="WARN",
            param_name=name,
            details={"mean": mean},
        )]
    return []
```

### test_plugin.py

```python
import torch
from my_safety_checks.clinic_plugin import register, detect_extreme_weights
from model_clinic.clinic import ConditionRegistry


def test_register():
    reg = ConditionRegistry()
    register(reg)
    assert "extreme_weights" in reg._detectors
    assert "suspicious_bias" in reg._detectors


def test_extreme_weights_fires():
    t = torch.randn(16, 16) * 1000
    findings = detect_extreme_weights("layer.weight", t, {})
    assert len(findings) == 1


def test_normal_weights_pass():
    t = torch.randn(16, 16)
    findings = detect_extreme_weights("layer.weight", t, {})
    assert len(findings) == 0
```

## Controlling Plugin Loading

### In the Python API

```python
from model_clinic import diagnose

# Load plugins (default)
findings = diagnose(state_dict)

# Skip plugins
findings = diagnose(state_dict, plugins=False)

# Manual plugin loading
from model_clinic import load_plugins
loaded = load_plugins()
print(f"Loaded: {loaded}")
```

### In the CLI

Plugins are always loaded for `model-clinic exam` and `model-clinic treat`.
There is currently no CLI flag to disable them -- use the Python API if needed.

## Listing Installed Plugins

```bash
$ model-clinic plugins
Installed plugins:
  my_safety_checks         v1.0.0     (my_safety_checks.clinic_plugin:register)
```

Or from Python:

```python
from model_clinic import list_plugins
for p in list_plugins():
    print(f"{p['name']} ({p['version']})")
```

## Publishing a Plugin

1. Choose a name like `model-clinic-<yourname>` or `<yourname>-model-clinic`.
2. Build: `pip install build && python -m build`
3. Upload: `pip install twine && twine upload dist/*`

Users install with `pip install model-clinic-<yourname>` and the plugin is
automatically available.

## Tips

- Keep detectors fast. They run once per tensor in the checkpoint.
- Use `tensor.float()` before computing statistics to avoid dtype issues.
- Check `tensor.dim()` and `tensor.numel()` early to skip irrelevant tensors.
- Use `ctx["_all_tensor_names"]` to look for related parameters (e.g., find
  matching bias for a weight tensor).
- Test with `model_clinic.make_everything_broken()` for a synthetic model
  with many known issues.
- Use `ConditionRegistry()` in tests to avoid polluting the global registry.
