"""Scaffold a new model-clinic plugin package.

Usage:
    model-clinic new-plugin my_detector_pack
"""

import argparse
import os
import sys


_PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
description = "A model-clinic plugin with custom detectors"
requires-python = ">=3.10"
dependencies = [
    "model-clinic>=0.3.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0"]

[project.entry-points."model_clinic.plugins"]
{name} = "{name}.clinic_plugin:register"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]

[tool.pytest.ini_options]
testpaths = ["tests"]
"""

_INIT_TEMPLATE = '"""model-clinic plugin: {name}."""\n'

_PLUGIN_TEMPLATE = '''\
"""model-clinic plugin registration for {name}.

This module is loaded automatically by model-clinic when the package is installed.
The register() function receives the ConditionRegistry and adds custom detectors.
"""

from model_clinic._types import Finding, Prescription


def register(registry):
    """Register all custom detectors and prescribers with model-clinic."""
    registry.register(
        condition="example_custom_check",
        detector=detect_example,
        prescriber=prescribe_example,
        risk="low",
        description="Example custom detector (replace with your own)",
    )


def detect_example(name, tensor, ctx):
    """Example detector: flags tensors with exactly zero std deviation.

    Parameters
    ----------
    name : str
        Parameter name (e.g. "model.layers.0.weight").
    tensor : torch.Tensor
        The parameter tensor.
    ctx : dict
        Shared context dict (contains metadata, all tensor names, etc.).

    Returns
    -------
    list[Finding]
        Findings for this parameter, or empty list if healthy.
    """
    if tensor.dim() < 2:
        return []
    if tensor.float().std().item() == 0.0:
        return [Finding(
            condition="example_custom_check",
            severity="WARN",
            param_name=name,
            details={{"message": "Tensor has zero variance (all identical values)"}},
        )]
    return []


def prescribe_example(finding):
    """Create a prescription for the example condition."""
    return Prescription(
        name="reinit_constant_tensor",
        description="Re-initialize tensor with small random values",
        risk="low",
        finding=finding,
        action="reinit_normal",
        params={{"std": 0.02}},
        explanation="A tensor with zero variance carries no information.",
    )
'''

_TEST_TEMPLATE = '''\
"""Tests for {name} plugin."""

import torch
from {name}.clinic_plugin import register, detect_example
from model_clinic.clinic import ConditionRegistry
from model_clinic._types import Finding


class TestPlugin:
    def test_register_adds_detector(self):
        registry = ConditionRegistry()
        register(registry)
        assert "example_custom_check" in registry._detectors

    def test_detect_example_flags_constant_tensor(self):
        t = torch.ones(16, 16)
        findings = detect_example("test.weight", t, {{}})
        assert len(findings) == 1
        assert findings[0].condition == "example_custom_check"

    def test_detect_example_passes_normal_tensor(self):
        t = torch.randn(16, 16)
        findings = detect_example("test.weight", t, {{}})
        assert len(findings) == 0
'''

_README_TEMPLATE = """\
# {name}

A [model-clinic](https://github.com/your-org/model-clinic) plugin.

## Installation

```bash
pip install -e .
```

Once installed, model-clinic will automatically discover and load this plugin.

## Usage

```bash
# Verify the plugin is detected
model-clinic plugins

# Run an exam (plugin detectors are included automatically)
model-clinic exam your_model.pt
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Adding Detectors

Edit `src/{name}/clinic_plugin.py` and add new detectors in the `register()` function:

```python
def register(registry):
    registry.register(
        condition="my_condition",
        detector=my_detector_fn,
        prescriber=my_prescriber_fn,  # optional
        risk="low",                   # low, medium, high
        description="What this checks for",
    )
```

A detector function signature:

```python
def my_detector(name: str, tensor: torch.Tensor, ctx: dict) -> list[Finding]:
    ...
```
"""


def main():
    parser = argparse.ArgumentParser(
        prog="model-clinic new-plugin",
        description="Generate a starter model-clinic plugin package",
    )
    parser.add_argument("name", help="Plugin package name (e.g. my_detector_pack)")
    parser.add_argument(
        "--output-dir", "-o", default=".",
        help="Directory to create the plugin in (default: current directory)",
    )
    args = parser.parse_args()

    name = args.name.replace("-", "_")
    plugin_dir = os.path.join(args.output_dir, name)

    if os.path.exists(plugin_dir):
        print(f"Error: directory '{plugin_dir}' already exists.", file=sys.stderr)
        sys.exit(1)

    # Create directory structure
    src_dir = os.path.join(plugin_dir, "src", name)
    tests_dir = os.path.join(plugin_dir, "tests")
    os.makedirs(src_dir, exist_ok=True)
    os.makedirs(tests_dir, exist_ok=True)

    _write(os.path.join(plugin_dir, "pyproject.toml"),
           _PYPROJECT_TEMPLATE.format(name=name))
    _write(os.path.join(src_dir, "__init__.py"),
           _INIT_TEMPLATE.format(name=name))
    _write(os.path.join(src_dir, "clinic_plugin.py"),
           _PLUGIN_TEMPLATE.format(name=name))
    _write(os.path.join(tests_dir, "test_plugin.py"),
           _TEST_TEMPLATE.format(name=name))
    _write(os.path.join(plugin_dir, "README.md"),
           _README_TEMPLATE.format(name=name))

    print(f"Created plugin package: {plugin_dir}/")
    print(f"  src/{name}/clinic_plugin.py  -- add your detectors here")
    print(f"  tests/test_plugin.py         -- plugin tests")
    print(f"  pyproject.toml               -- entry point configured")
    print()
    print("Next steps:")
    print(f"  cd {plugin_dir}")
    print(f"  pip install -e .")
    print(f"  model-clinic plugins          # verify detection")


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    main()
