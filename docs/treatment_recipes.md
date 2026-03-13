# Treatment Recipes

Proven fix combinations for common model health problems. Each recipe includes the
symptom, root cause, CLI fix, Python fix, and a verification step.

---

## Recipe 1: Fix a Fine-Tuned Model with Norm Drift

**Symptom**: Health score drops 10-20 points after SFT. `norm_drift` findings on
most or all LayerNorm layers.

**Cause**: Learning rate too high during supervised fine-tuning. LayerNorm weights
drift from 1.0 as the optimizer over-adjusts them. Common when fine-tuning without
per-layer LR scaling or norm-weight freezing.

**Example output**:
```
[WARN] norm_drift  layers.0.norm.weight  mean=3.71 (should be ~1.0)
[WARN] norm_drift  layers.1.norm.weight  mean=4.12 (should be ~1.0)
[WARN] norm_drift  layers.2.norm.weight  mean=2.98 (should be ~1.0)
```

**Fix**:
```bash
model-clinic treat checkpoint.pt --conservative --conditions norm_drift --save fixed.pt
```

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment, save_state_dict

state_dict, meta = load_state_dict("checkpoint.pt")
findings = diagnose(state_dict)
norm_findings = [f for f in findings if f.condition == "norm_drift"]
prescriptions = prescribe(norm_findings, conservative=True)
for rx in prescriptions:
    apply_treatment(state_dict, rx)
save_state_dict(state_dict, "fixed.pt")
```

**Verify**:
```bash
model-clinic exam fixed.pt
# norm_drift should be gone; expect +10-20 pts on score
```

**Prevention**: During SFT, use a lower LR for norm weights (`lr_norm = lr * 0.1`)
or freeze LayerNorm weights entirely during the first epoch.

---

## Recipe 2: Rescue a Merged Model with Dead Neurons

**Symptom**: Model merges (TIES, DARE, linear interpolation) often leave neurons
with near-zero weights — the cancellation artifact. Low coherence despite reasonable
perplexity.

**Example output**:
```
[WARN] dead_neurons  layers.3.mlp.down_proj.weight  45/4096 dead rows (1.1%)
[WARN] dead_neurons  layers.7.mlp.down_proj.weight  92/4096 dead rows (2.2%)
```

**Fix**:
```bash
# Conservative: reinit dead neurons at 10% of Kaiming scale (safe)
model-clinic treat merged.pt --conservative --save fixed.pt

# More aggressive: full Kaiming reinit (risks changing model behavior)
model-clinic treat merged.pt --conditions dead_neurons --save fixed.pt
```

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment, save_state_dict, compute_health_score

state_dict, _ = load_state_dict("merged.pt")
findings = diagnose(state_dict)
health_before = compute_health_score(findings)

dead = [f for f in findings if f.condition == "dead_neurons"]
rxs = prescribe(dead, conservative=True)  # low-risk reinit only
for rx in rxs:
    apply_treatment(state_dict, rx)

health_after = compute_health_score(diagnose(state_dict))
print(f"Score: {health_before.overall} -> {health_after.overall}")
save_state_dict(state_dict, "fixed.pt")
```

**Verify**:
```bash
model-clinic exam fixed.pt        # dead_neurons count should drop
model-clinic compare merged.pt fixed.pt  # side-by-side diff
```

**Note**: If >10% of neurons are dead (severe merge failure), consider re-doing
the merge with a different coefficient or merge method rather than patching.

---

## Recipe 3: Fix Exploding Norms After LoRA Merge

**Symptom**: After merging LoRA adapters into the base model (`merge_and_unload`),
some weight matrices have exploding norms. Generation outputs garbage or nonsense.

**Example output**:
```
[WARN] exploding_norm  model.layers.12.self_attn.q_proj.weight  per-elem norm=47.3
[WARN] exploding_norm  model.layers.12.self_attn.k_proj.weight  per-elem norm=31.1
[WARN] lora_merge_artifacts  ...  effective_rank=2 (full rank expected)
```

**Cause**: LoRA rank too high relative to the target module, or alpha/rank ratio
(`lora_alpha / r`) much greater than 1. The scaling factor blows up merged weights.

**Fix**:
```bash
# Scale exploding norms back to healthy range (non-destructive)
model-clinic treat merged_lora.pt --conditions exploding_norm --save fixed.pt

# Dry run first to see what would change
model-clinic treat merged_lora.pt --conditions exploding_norm --dry-run
```

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment, save_state_dict

state_dict, _ = load_state_dict("merged_lora.pt")
findings = diagnose(state_dict)

# Treat both exploding norms and LoRA artifacts
target_conditions = {"exploding_norm", "lora_merge_artifacts"}
relevant = [f for f in findings if f.condition in target_conditions]
rxs = prescribe(relevant, conservative=False)  # medium risk OK here
for rx in rxs:
    result = apply_treatment(state_dict, rx)
    print(f"{'OK' if result.success else 'FAIL'}: {result.description}")

save_state_dict(state_dict, "fixed.pt")
```

**Verify**:
```bash
model-clinic validate fixed.pt
model-clinic exam fixed.pt --explain  # check exploding_norm is resolved
```

**Prevention**: Keep `lora_alpha == lora_r` (scaling factor = 1.0) or use
`use_rslora=True` in PEFT to normalize the scaling automatically.

---

## Recipe 4: Handle NaN/Inf in Checkpoint

**Symptom**: Model outputs `nan` during inference. `nan_inf` error finding.
May happen after training with mixed precision or after a failed gradient step.

**Example output**:
```
[ERROR] nan_inf  layers.5.mlp.gate_proj.weight  3 NaN, 0 Inf / 2,097,152 total
```

**Fix**:
```bash
# Zero out NaN/Inf values (the only safe treatment — no data is recoverable)
model-clinic treat checkpoint.pt --conditions nan_inf --save fixed.pt
```

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment, save_state_dict

state_dict, _ = load_state_dict("checkpoint.pt")
findings = diagnose(state_dict)

nan_findings = [f for f in findings if f.condition == "nan_inf"]
if nan_findings:
    print(f"Found NaN/Inf in {len(nan_findings)} tensor(s)")
    rxs = prescribe(nan_findings)
    for rx in rxs:
        apply_treatment(state_dict, rx)
    save_state_dict(state_dict, "fixed.pt")
    print("NaN/Inf zeroed. Re-run exam to confirm.")
else:
    print("No NaN/Inf found.")
```

**Verify**:
```bash
model-clinic validate fixed.pt   # should show [PASS] Integrity: all tensors finite
```

**Important**: Zeroing NaN values may degrade model quality in that layer.
If NaN/Inf spans >0.1% of a tensor, that checkpoint is likely unrecoverable —
fall back to the previous checkpoint. Check training logs for the step where NaN
first appeared.

---

## Recipe 5: Reduce Heavy Tails After Aggressive Training

**Symptom**: Very high kurtosis in weight distributions. Model sometimes produces
extreme logits / incoherent outputs on certain prompts, but works fine on most.

**Example output**:
```
[WARN] heavy_tails  layers.2.attention.q_proj.weight  kurtosis=87 (normal=3)
[WARN] heavy_tails  layers.4.attention.v_proj.weight  kurtosis=134 (normal=3)
```

**Cause**: Outlier weight values (a few weights with extreme magnitude) from
aggressive training, large gradient updates, or insufficient weight decay.

**Fix**:
```bash
# Clamp weights beyond 4 sigma (conservative, affects ~0.003% of values)
model-clinic treat checkpoint.pt --conditions heavy_tails --conservative --save fixed.pt
```

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment, save_state_dict

state_dict, _ = load_state_dict("checkpoint.pt")
findings = diagnose(state_dict)

tail_findings = [f for f in findings if f.condition == "heavy_tails"]
# Conservative mode uses 4-sigma clamping (minimal impact on distribution)
rxs = prescribe(tail_findings, conservative=True)
for rx in rxs:
    apply_treatment(state_dict, rx)

save_state_dict(state_dict, "fixed.pt")
```

**Verify**:
```bash
model-clinic exam fixed.pt       # kurtosis values should drop significantly
model-clinic compare checkpoint.pt fixed.pt  # confirm no other changes
```

**Trade-off**: Clamping outliers rarely hurts average-case performance but can
affect rare/specialized knowledge stored in extreme weights. Run generation tests
on diverse prompts before and after if possible.

---

## Recipe 6: Break Duplicate Rows (Merge Symmetry Collapse)

**Symptom**: After model merging, many weight rows are identical. `identical_rows`
findings. Model has reduced diversity in its representations.

**Example output**:
```
[WARN] identical_rows  lm_head.weight  top pair cosine_sim=1.0000 (500 pairs sampled)
[WARN] token_collapse  lm_head.weight  95.2% rows identical
```

**Cause**: Linear model merging with identical coefficients collapses rows that
were already similar. Often seen in `lm_head` when merging models trained on the
same base.

**Fix**:
```bash
# Add small noise to break symmetry (low risk)
model-clinic treat checkpoint.pt --conditions identical_rows --conservative --save fixed.pt
```

```python
from model_clinic import load_state_dict, diagnose, prescribe, apply_treatment, save_state_dict

state_dict, _ = load_state_dict("checkpoint.pt")
findings = diagnose(state_dict)

dup_findings = [f for f in findings if f.condition in ("identical_rows", "token_collapse")]
rxs = prescribe(dup_findings, conservative=True)
for rx in rxs:
    result = apply_treatment(state_dict, rx)
    print(f"{'OK' if result.success else 'FAIL'}: {result.description}")

save_state_dict(state_dict, "fixed.pt")
```

**Verify**:
```bash
model-clinic exam fixed.pt --conditions identical_rows,token_collapse
```

**Note**: Small noise perturbation is reversible in spirit but cannot be undone
exactly. Always save the original before applying.

---

## Recipe 7: Full Post-Merge Triage (Comprehensive)

**Symptom**: General model degradation after any merge operation. Multiple finding
types present. Score dropped from base model.

**Fix**: Run the full treatment pipeline with conservative mode:

```bash
# Step 1: Examine and show full treatment plan
model-clinic exam merged.pt --explain

# Step 2: Dry run to preview all changes
model-clinic treat merged.pt --conservative --dry-run

# Step 3: Apply conservative fixes
model-clinic treat merged.pt --conservative --save fixed.pt

# Step 4: Compare before/after
model-clinic compare merged.pt fixed.pt
```

```python
from model_clinic import (
    load_state_dict, diagnose, prescribe, apply_treatment,
    compute_health_score, save_state_dict, TreatmentManifest,
)
import copy

state_dict, meta = load_state_dict("merged.pt")
original = copy.deepcopy(state_dict)

# Diagnose
findings = diagnose(state_dict)
health_before = compute_health_score(findings)
print(f"Before: {health_before.overall}/100 ({health_before.grade})")
print(f"Findings: {len(findings)}")

# Treat conservatively
rxs = prescribe(findings, conservative=True)
manifest = TreatmentManifest()
results = []
for rx in rxs:
    result = apply_treatment(state_dict, rx)
    results.append(result)
    if result.success:
        manifest.record(
            param_name=rx.finding.param_name,
            action=rx.action,
            description=result.description,
        )

# Re-score
health_after = compute_health_score(diagnose(state_dict))
print(f"After:  {health_after.overall}/100 ({health_after.grade})")

# Save only if improved (or unchanged)
if health_after.overall >= health_before.overall:
    save_state_dict(state_dict, "fixed.pt")
    print(f"Saved fixed.pt  (+{health_after.overall - health_before.overall} pts)")
else:
    print("Warning: score dropped after treatment — investigate before saving")
```

**Note**: If conservative treatment doesn't move the score enough, re-run without
`conservative=True` to include medium-risk fixes, but test generation quality
before and after.

---

## Quick Reference: Condition → Recipe

| Condition | Recipe |
|-----------|--------|
| `norm_drift` | Recipe 1 |
| `dead_neurons` | Recipe 2 |
| `exploding_norm`, `lora_merge_artifacts` | Recipe 3 |
| `nan_inf` | Recipe 4 |
| `heavy_tails` | Recipe 5 |
| `identical_rows`, `token_collapse` | Recipe 6 |
| Multiple conditions after merge | Recipe 7 |

## Related

- `docs/troubleshooting.md` — when treatment doesn't work as expected
- `notebooks/02_treatment_guide.ipynb` — interactive treatment walkthrough
- CLI `--explain` flag — shows per-finding explanations and risk information
