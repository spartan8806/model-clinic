# Troubleshooting Guide

Common problems and their solutions when using model-clinic.

---

## "My model generates garbage but exam shows 90/100"

**Diagnosis steps**:
1. Check whether runtime analysis was included: `model-clinic exam checkpoint.pt --hf --runtime`
2. Static analysis only looks at weight tensors — it cannot detect all generation failures
3. Run with `--explain` to see if any advisory (INFO) findings were suppressed

**Likely cause**:
- The model has runtime problems that static analysis cannot see: context window
  handling, tokenizer mismatch, sampling parameters, or prompt format dependency
- The 90/100 score reflects healthy weight distributions, not generation quality
- Some conditions (e.g. `low_coherence`, `generation_collapse`) are runtime-only
  detectors and require `--hf --runtime` to appear

**Solution**:
```bash
# Run full runtime analysis (requires transformers and a tokenizer)
model-clinic exam checkpoint.pt --hf --runtime --example-prompts

# Or check generation quality directly
python -c "
from model_clinic import load_model, eval_coherence, eval_logit_entropy
model, tokenizer = load_model('checkpoint.pt', trust_remote_code=True)
score = eval_coherence(model, tokenizer)
entropy = eval_logit_entropy(model, tokenizer)
print(f'Coherence: {score:.2f}/1.0')
print(f'Logit entropy: {entropy:.3f} (healthy: 3-6)')
"
```

If logit entropy is near 0, the model has generation collapse regardless of
what static analysis shows. See also: `generation_collapse` condition documentation.

---

## "Treatment made things worse"

**Diagnosis steps**:
1. Check what was applied: re-run `model-clinic exam original.pt` vs `treated.pt`
2. Check if any medium/high-risk treatments were applied

**Likely cause**:
- A medium or high-risk treatment (scale adjustments, heavy-tail clamping) modified
  weights that encode specialized knowledge
- The health score improved (fewer structural issues) but generation quality dropped
  because the model relied on those "outlier" weights

**Solution**:
```bash
# Roll back: always keep the original
cp checkpoint.pt checkpoint_backup.pt

# Next time, use conservative mode (low-risk only)
model-clinic treat checkpoint.pt --conservative --save fixed.pt

# Compare before/after treatment
model-clinic compare checkpoint.pt fixed.pt
```

```python
# Programmatic rollback
from model_clinic import rollback_treatment

# apply_treatment returns a TreatmentResult with a .backup field
for result in applied_results:
    if result.success and result.backup is not None:
        rollback_treatment(state_dict, result)
```

**Rule**: Always keep a copy of the original checkpoint. Never overwrite in-place.
If you need to test generation quality, use `--test` (requires `--hf`) to automatically
rollback if generation scores drop.

---

## "Score is 0/F but model works fine"

**Diagnosis steps**:
1. Check the full finding list: `model-clinic exam model.pt --verbose`
2. See if `nan_inf` ERROR is driving the score to 0
3. Verify the checkpoint loaded correctly: `model-clinic validate model.pt`

**Likely cause**:
- A single ERROR-level finding (especially `nan_inf`) can heavily penalize the score
- Stale metadata tensors (training-time counters, EMA accumulators) are triggering
  false positives — these are expected to have unusual distributions
- The state dict contains non-weight tensors (optimizer state, scheduler state)
  that should not be analyzed

**Solution**:
```bash
# Check if it's specific tensors causing issues
model-clinic exam model.pt --verbose

# If the checkpoint includes optimizer state, extract just the model weights
python -c "
import torch
checkpoint = torch.load('model.pt', map_location='cpu')
# For a checkpoint that wraps the state dict:
if 'model_state_dict' in checkpoint:
    torch.save(checkpoint['model_state_dict'], 'weights_only.pt')
elif 'state_dict' in checkpoint:
    torch.save(checkpoint['state_dict'], 'weights_only.pt')
print('Extracted weights_only.pt')
"
model-clinic exam weights_only.pt
```

If specific tensor names like `neuron_age`, `gradient_sq_ema`, `step_count` appear
in findings, they are metadata tensors (training tracking arrays) and should be
ignored — model-clinic already filters known metadata keywords by default, but
custom training frameworks may use different names.

---

## "Memory error during exam"

**Diagnosis steps**:
1. Check model size: `model-xray model.pt | head -5`
2. Identify which detector is OOM-ing (use `--verbose`)

**Likely cause**:
- Pairwise operations (identical_rows, head_redundancy) load large tensors fully
  into memory
- On Windows: memory-mapped file access can fail on very large models

**Solution**:

```bash
# Skip expensive pairwise checks
model-clinic exam model.pt --skip-conditions identical_rows,head_redundancy

# Reduce analysis scope to specific conditions
model-clinic exam model.pt --conditions nan_inf,dead_neurons,norm_drift

# On Windows — if you see 'paging file too small' or safetensors crash:
# See "Windows: paging file" section below
```

```python
# Programmatic: stream-friendly analysis
from model_clinic import diagnose

# Only run specific detectors
findings = diagnose(state_dict, conditions=["nan_inf", "dead_neurons", "norm_drift"])
```

For models larger than ~7B parameters on machines with <32GB RAM, consider running
model-clinic on a subset of layers or using the CLI with `--conditions` to limit
which detectors run.

---

## "Findings show norm_drift but model trains fine"

**Diagnosis steps**:
1. Check the drift magnitude: `model-clinic exam model.pt --verbose`
2. Note whether findings are WARN or INFO

**Likely cause**:
- Large pretrained models (GPT-2, Llama, Qwen, etc.) naturally develop non-1.0
  LayerNorm weights during pretraining. A mean of 1.2-2.0 is common and not
  necessarily harmful
- The `norm_drift` threshold (default: |mean - 1.0| > 1.5) is intentionally
  permissive to avoid false positives on pretrained models
- Training "fine" from a loss perspective is compatible with elevated norm weights

**Solution**:

This finding is advisory. You do not need to fix it unless:
- The mean is very high (>4.0)
- It appeared after fine-tuning (i.e., it was not present in the base model)
- You have generation quality issues alongside it

```bash
# Check if base model also had this
model-clinic compare base_model.pt finetuned.pt

# Treat only if the drift is post-SFT and score dropped
model-clinic treat finetuned.pt --conditions norm_drift --conservative --save fixed.pt
```

**Reference**: Norm drift during fine-tuning is documented in "Scaling Instruction-Finetuned Language Models" (Chung et al., 2022).

---

## "My model has 500+ findings"

**Diagnosis steps**:
1. Check what conditions dominate: group by condition type
2. Check if metadata tensors are being analyzed

**Likely cause**:
- Gradient noise, representation drift, and attention imbalance findings are
  generated per-tensor and can produce hundreds of INFO findings on large models
- The model is a frankenstein merge with systematic issues across all layers

**Solution**:

```bash
# Focus on actionable findings only (ERROR and WARN, not INFO)
model-clinic exam model.pt --min-severity WARN

# Get finding count by condition
model-clinic exam model.pt --json | python -c "
import json, sys
from collections import Counter
data = json.load(sys.stdin)
counts = Counter(f['condition'] for f in data['findings'])
for cond, n in counts.most_common():
    print(f'{n:4d}  {cond}')
"
```

```python
from model_clinic import diagnose, compute_health_score

findings = diagnose(state_dict)
# Filter to only actionable findings
actionable = [f for f in findings if f.severity in ("ERROR", "WARN")]
health = compute_health_score(actionable)
print(f"Actionable findings: {len(actionable)} (of {len(findings)} total)")
```

Start with ERROR findings, then work down through WARN by impact. INFO findings
are advisory and generally do not need to be fixed.

---

## "Windows: 'paging file too small' error"

**Symptom**: Python crashes or hangs when loading large `.safetensors` models on Windows.
Error messages like `[WinError 1455] The paging file is too small` or silent crashes.

**Cause**: The `safetensors` library uses memory-mapped files by default. On Windows,
memory mapping large files (>4GB) can exhaust the paging file, especially with
insufficient virtual memory configured.

**Solution**:

Option 1 — Use PyTorch `.pt` format instead of `.safetensors`:
```bash
# Convert safetensors to pt first
python -c "
import torch
from safetensors.torch import load_file
sd = load_file('model.safetensors')  # may still fail
torch.save(sd, 'model.pt')
"
```

Option 2 — Use sequential loading (model-clinic's built-in workaround):
```python
# If using the Python API, force sequential read:
import safetensors.torch as st
import torch

# Read tensor-by-tensor to avoid large mmap
state_dict = {}
with st.safe_open("model.safetensors", framework="pt", device="cpu") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key).clone()
```

Option 3 — Increase the Windows paging file:
- Control Panel > System > Advanced System Settings > Performance Settings
- Advanced > Virtual memory > Change
- Set to at least 1.5x your RAM

**Note**: This is a Windows OS limitation, not a model-clinic bug.
model-clinic's CLI already uses sequential loading internally for safetensors.

---

## "Treatment: 'condition not fixable' message"

**Diagnosis steps**:
1. Check the condition type — not all conditions have automated fixes
2. Read the explanation: `model-clinic exam model.pt --explain`

**Likely cause**:
- Several conditions are advisory-only with no automated treatment because the
  fix requires human judgment or model-specific knowledge
- Advisory conditions: `attention_imbalance`, `dtype_mismatch`, `weight_corruption`,
  `head_redundancy`, `positional_encoding_issues`, `token_collapse`,
  `gradient_noise`, `representation_drift`, `moe_router_collapse`,
  `generation_collapse`, `low_coherence`

**Solution**:

Check the condition's advisory message for guidance:

| Condition | Manual fix |
|-----------|------------|
| `attention_imbalance` | Check Q/K/V init; consider separate LR for attention params |
| `dtype_mismatch` | Cast to consistent dtype: `{k: v.float() for k,v in sd.items()}` |
| `weight_corruption` | Zero/constant layers: likely a training bug; revert checkpoint |
| `head_redundancy` | Prune redundant heads; or accept (common in over-parameterized models) |
| `generation_collapse` | Runtime issue: check sampling params, temperature, repetition penalty |
| `low_coherence` | Check tokenizer alignment, prompt format, and SFT data quality |
| `moe_router_collapse` | Expert load imbalance: may require auxiliary loss during training |

---

## "exam command hangs on a large model"

**Diagnosis steps**:
1. Check model size and whether `--verbose` shows which detector is running
2. Some detectors have `O(n²)` pairwise complexity

**Solution**:
```bash
# Skip slow pairwise detectors
model-clinic exam model.pt --skip-conditions identical_rows,head_redundancy,attention_imbalance

# Run only fast detectors
model-clinic exam model.pt --conditions nan_inf,dead_neurons,norm_drift,exploding_norm,vanishing_norm
```

The `identical_rows` detector samples up to 200 rows but is still slow on
very wide weight matrices (vocab × hidden). For models with vocab >100K, it's
safe to skip.

---

## Related Resources

- `docs/treatment_recipes.md` — cookbook of proven fix combinations
- `notebooks/01_quickstart.ipynb` — basic diagnosis walkthrough
- `notebooks/02_treatment_guide.ipynb` — treatment with before/after comparison
- `notebooks/03_training_monitor.ipynb` — catch problems during training
- CLI `--explain` flag — per-finding explanation and risk information
