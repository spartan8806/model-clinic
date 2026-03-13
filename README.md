# model-clinic

[![PyPI version](https://img.shields.io/pypi/v/model-clinic.svg)](https://pypi.org/project/model-clinic/)
[![Tests](https://img.shields.io/github/actions/workflow/status/spartan8806/model-clinic/publish.yml?label=tests)](https://github.com/spartan8806/model-clinic/actions)
[![Python versions](https://img.shields.io/pypi/pyversions/model-clinic.svg)](https://pypi.org/project/model-clinic/)

Diagnose, treat, and understand neural network models. Like a doctor for your PyTorch models.

```
pip install model-clinic
```

## What it does

Finds problems in model weights, prescribes fixes, applies them with before/after testing, and rolls back if things get worse.

**Static analysis** (no GPU needed):
- Dead neurons, stuck gates, NaN/Inf
- Exploding/vanishing norms, LayerNorm drift
- Heavy-tailed distributions, saturated weights
- Duplicate rows, attention Q/K/V imbalance
- Mixed dtypes, weight corruption
- Head redundancy, positional encoding issues
- Token collapse, gradient noise, representation drift
- MoE router collapse, LoRA merge artifacts
- Quantization degradation, model aging/forgetting

**Runtime analysis** (needs model + tokenizer):
- Generation collapse detection (entropy, top-1 probability)
- Coherence scoring across diverse prompts
- Activation health per layer (hooks)
- Residual stream growth tracking
- Response diversity metrics

## Quick start

```bash
# Examine a checkpoint (diagnose only)
model-clinic exam checkpoint.pt

# Examine a HuggingFace model
model-clinic exam Qwen/Qwen2.5-0.5B-Instruct --hf

# Include runtime diagnostics
model-clinic exam checkpoint.pt --hf --runtime

# Use diverse example prompts for runtime testing
model-clinic exam checkpoint.pt --hf --runtime --example-prompts

# Verbose output (show each detector as it runs)
model-clinic exam checkpoint.pt --verbose

# Treat and save
model-clinic treat checkpoint.pt --save treated.pt

# Treat with before/after generation testing
model-clinic treat checkpoint.pt --test --save treated.pt

# Only safe fixes
model-clinic treat checkpoint.pt --conservative --save treated.pt

# Dry run
model-clinic treat checkpoint.pt --dry-run

# JSON output (for CI pipelines)
model-clinic exam checkpoint.pt --json

# Show why each fix is recommended
model-clinic exam checkpoint.pt --explain

# Generate HTML diagnostic report
model-clinic report checkpoint.pt --output report.html

# Compare two checkpoints
model-clinic compare before.pt after.pt

# Try it with a synthetic broken model (no checkpoint needed)
model-clinic demo everything-broken
```

## Example output

### Exam

```
$ model-clinic exam my_model.pt

Loading: my_model.pt
Loaded 156 tensors, 494,032,896 parameters

================================================================================
DIAGNOSIS -- 7 finding(s) (1 errors, 4 warnings, 2 info)
================================================================================

  [ERROR] nan_inf (1 instance(s))
    layers.5.mlp.gate_proj.weight: 3 NaN, 0 Inf / 2,097,152 total

  [WARN] dead_neurons (2 instance(s))
    layers.3.mlp.down_proj.weight: 12/4096 dead rows (0.3%)
    layers.7.mlp.down_proj.weight: 8/4096 dead rows (0.2%)

  [WARN] norm_drift (1 instance(s))
    model.norm.weight: mean=1.7724 (should be ~1.0)

  [WARN] heavy_tails (1 instance(s))
    layers.2.attention.q_proj.weight: kurtosis=87 (normal=3)

Model Health Score
---------------------------------------------
  Overall: 72/100  C

  weights        ################....  80/100
  stability      ###########.........  55/100
  output         #################### 100/100
  activations    #################### 100/100

================================================================================
VERDICT: UNHEALTHY (1 errors, 4 warnings)
================================================================================
```

### Treat

```
$ model-clinic treat my_model.pt --conservative --save treated.pt

  [OK] Rx #1 reinit_dead_neurons [LOW]
    Reinit 12 dead rows (0.1x Kaiming)
  [OK] Rx #2 reset_norm [LOW]
    Norm weights: 1.7724 -> 1.0

  Applied: 2/4 (conservative mode: 2 skipped)
  Saved treated model to treated.pt
```

### Validate

```
$ model-clinic validate treated.pt

  [PASS] Load: 156 tensors, 494M parameters (1.87 GB)
  [PASS] Integrity: all tensors finite
  [PASS] Shapes: all valid
  [INFO] Dtypes: float32 (156 tensors)

RESULT: VALID
```

## All tools

| Command | What it does |
|---------|-------------|
| `model-clinic exam` | Diagnose model health, show treatment plan |
| `model-clinic treat` | Diagnose and apply fixes |
| `model-clinic validate` | Verify a checkpoint loads and infers correctly |
| `model-clinic report` | Generate an HTML diagnostic report |
| `model-clinic compare` | Compare health impact between two checkpoints |
| `model-xray` | Per-parameter weight stats (shape, norm, sparsity) |
| `model-diff` | Compare two checkpoints param-by-param |
| `model-health` | Quick health check (dead neurons, norms, gates) |
| `model-surgery` | Direct parameter modification (interactive REPL) |
| `model-ablate` | Disable parts systematically, measure impact |
| `model-neurons` | Profile neuron activations across prompts |
| `model-attention` | Attention patterns per head per layer |
| `model-logit-lens` | Watch predictions form layer by layer |
| `model-clinic demo` | Generate and examine a synthetic broken model |

## Python API

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment

# Load any checkpoint format
state_dict, meta = load_state_dict("checkpoint.pt")

# Diagnose
findings = diagnose(state_dict)
for f in findings:
    print(f"[{f.severity}] {f.condition}: {f.param_name}")

# Prescribe
prescriptions = prescribe(findings, conservative=True)

# Treat
for rx in prescriptions:
    result = apply_treatment(state_dict, rx)
    print(f"{'OK' if result.success else 'FAIL'}: {result.description}")

# Health score
from model_clinic import compute_health_score
health = compute_health_score(findings)
print(f"Score: {health.overall}/100 ({health.grade})")

# Training monitor (call during training loop)
from model_clinic import ClinicMonitor
monitor = ClinicMonitor(check_every=500, alert_on=["nan_inf", "dead_neurons"])
# Inside training loop:
#   alerts = monitor.check(model)
```

### Full API

```python
# Types
from model_clinic import (
    Finding, Prescription, TreatmentResult, ExamReport, ModelMeta,
    HealthScore, ExamResult, PipelineResult, MonitorAlert, MonitorSummary,
)

# Loader
from model_clinic import load_state_dict, load_model, build_meta, save_state_dict

# Clinic
from model_clinic import diagnose, prescribe, apply_treatment, rollback_treatment
from model_clinic import examine_batch, create_pipeline, TreatmentPipeline

# Health score
from model_clinic import compute_health_score, print_health_score

# Monitor
from model_clinic import ClinicMonitor, ClinicTrainerCallback

# Manifest
from model_clinic import TreatmentManifest

# Evaluation (requires transformers)
from model_clinic import generate, eval_coherence, eval_perplexity
from model_clinic import eval_logit_entropy, eval_diversity

# Synthetic models (for testing/CI)
from model_clinic import SYNTHETIC_MODELS, make_healthy_mlp, make_everything_broken
```

## Conditions detected

| Condition | Severity | Treatment |
|-----------|----------|-----------|
| `nan_inf` | ERROR | Zero out NaN/Inf values |
| `dead_neurons` | WARN/ERROR | Reinit with small Kaiming values |
| `stuck_gate_closed` | WARN | Nudge toward trainable range |
| `stuck_gate_open` | WARN | Pull back from saturation |
| `exploding_norm` | WARN | Scale to healthy range |
| `vanishing_norm` | WARN | Reinit near-zero params |
| `heavy_tails` | WARN | Clamp beyond 4σ |
| `norm_drift` | WARN | Reset LayerNorm to 1.0 |
| `saturated_weights` | WARN | Scale down |
| `identical_rows` | WARN | Perturb to break symmetry |
| `attention_imbalance` | WARN | Advisory |
| `dtype_mismatch` | WARN | Advisory |
| `weight_corruption` | WARN | Advisory |
| `head_redundancy` | WARN | Advisory |
| `positional_encoding_issues` | WARN | Advisory |
| `token_collapse` | WARN | Advisory |
| `gradient_noise` | WARN | Advisory |
| `representation_drift` | WARN | Advisory |
| `moe_router_collapse` | WARN/INFO | Advisory |
| `lora_merge_artifacts` | WARN | Advisory |
| `generation_collapse` | ERROR | (runtime) Advisory |
| `low_coherence` | WARN/ERROR | (runtime) Advisory |
| `activation_nan/inf` | ERROR | (runtime) Check weight surgery |
| `activation_explosion` | WARN | (runtime) Check norms |
| `residual_explosion` | WARN | (runtime) Layer investigation needed |
| `quantization_degradation` | WARN/INFO | Advisory |
| `model_aging` | WARN | Advisory |

## Custom conditions

```python
from model_clinic.clinic import REGISTRY
from model_clinic import Finding, Prescription

def my_detector(name, tensor, ctx):
    if "my_layer" in name and tensor.norm() > 100:
        return [Finding("my_issue", "WARN", name, {"norm": tensor.norm().item()})]
    return []

def my_prescriber(finding):
    return Prescription("fix_my_issue", "Scale it down", "low", finding, "scale_norm",
                       {"target_per_elem": 1.0})

REGISTRY.register("my_issue", my_detector, my_prescriber, "low", "My custom check")
```

## Synthetic models (for testing and demos)

```python
from model_clinic import make_everything_broken, SYNTHETIC_MODELS

# Generate a model with every type of issue
state_dict = make_everything_broken()

# Available presets
for name in sorted(SYNTHETIC_MODELS.keys()):
    print(name)
# healthy, dead-neurons, nan, exploding, norm-drift, collapsed,
# heavy-tails, duplicate-rows, stuck-gates, corrupted, everything-broken
```

```bash
# CLI demo (no checkpoint needed)
model-clinic demo everything-broken
model-clinic demo dead-neurons --treat
model-clinic demo --list
```

## CI integration (GitHub Actions)

```yaml
# In your workflow:
- uses: spartan8806/model-clinic@v0.3.0
  with:
    model-path: checkpoints/model.pt
    threshold: 60  # Fail if health score < 60
```

See `action.yml` and `.github/workflows/model-health.yml` for full examples.

## Supported formats

- HuggingFace models (local or hub)
- PyTorch `.pt`/`.pth` checkpoints
- Safetensors (`.safetensors`) --- requires `pip install model-clinic[safetensors]`
- Nested checkpoint dicts (`model_state_dict`, `state_dict`)
- Composite checkpoints (multiple named state dicts)

## Installation

```bash
# Core (static analysis only, no HuggingFace dependency)
pip install model-clinic

# With HuggingFace support (runtime analysis, generation testing)
pip install model-clinic[hf]

# With safetensors support
pip install model-clinic[safetensors]

# Everything
pip install model-clinic[all]

# Development
pip install model-clinic[dev]
```

## Development

```bash
git clone https://github.com/spartan8806/model-clinic.git
cd model-clinic
pip install -e ".[dev,all]"
pytest tests/ -v
```

## License

MIT
