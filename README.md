# model-clinic

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
- Mixed dtypes

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
```

## All tools

| Command | What it does |
|---------|-------------|
| `model-clinic exam` | Diagnose model health, show treatment plan |
| `model-clinic treat` | Diagnose and apply fixes |
| `model-xray` | Per-parameter weight stats (shape, norm, sparsity) |
| `model-diff` | Compare two checkpoints param-by-param |
| `model-health` | Quick health check (dead neurons, norms, gates) |
| `model-surgery` | Direct parameter modification (interactive REPL) |
| `model-ablate` | Disable parts systematically, measure impact |
| `model-neurons` | Profile neuron activations across prompts |
| `model-attention` | Attention patterns per head per layer |
| `model-logit-lens` | Watch predictions form layer by layer |

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
```

### Full API

```python
# Types
from model_clinic import Finding, Prescription, TreatmentResult, ExamReport, ModelMeta

# Loader
from model_clinic import load_state_dict, load_model, build_meta, save_state_dict

# Clinic
from model_clinic import diagnose, prescribe, apply_treatment, rollback_treatment

# Evaluation (requires transformers)
from model_clinic import generate, eval_coherence, eval_perplexity
from model_clinic import eval_logit_entropy, eval_diversity
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
| `generation_collapse` | ERROR | (runtime) Advisory |
| `low_coherence` | WARN/ERROR | (runtime) Advisory |
| `activation_nan/inf` | ERROR | (runtime) Check weight surgery |
| `activation_explosion` | WARN | (runtime) Check norms |
| `residual_explosion` | WARN | (runtime) Layer investigation needed |

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

## Supported formats

- HuggingFace models (local or hub)
- PyTorch `.pt`/`.pth` checkpoints
- Safetensors (`.safetensors`) — requires `pip install model-clinic[safetensors]`
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
