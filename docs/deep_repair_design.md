# Deep Repair: Beyond Cosmetic Treatment

## Status (2026-03-13)

**All 5 levels implemented and tested.** 699 tests passing. Validated on real model.
**Gate opening experiment: MEMORY WORKS.** 251x perplexity improvement.

### Full Pipeline Results on ATLES 300M RecurrentTransformer

| Stage | Loss | PPL | Score | Params Trained | Time |
|-------|------|-----|-------|----------------|------|
| Original (gates closed) | 9.37 | 11,726 | 62/D | — | — |
| L1+L2 (static repair) | — | — | 65/C | 0 | <1s |
| L3 (distillation) | — | — | 71/C | 5.2M reset | 9s |
| **Gates opened + trained** | **3.84** | **46.6** | — | **5.2M** | **7 min** |

**The breakthrough:** Memory was never architecturally broken. The gating bug from
original training kept gates at sigmoid(-5)=0.7%, starving memory of gradient signal.
After surgical repair (clean banks) + gate opening (sigmoid(0)=50%) + 1000 steps of
next-token prediction training (5.2M params, 7 min on RTX 3060), the memory system
**reduces perplexity 251x** compared to the no-memory baseline.

The read gate naturally settled at 52% — the model chose to keep using memory.
It didn't learn to shut it off.

### What each stage did

**L1 (cosmetic):** Clamped memory tier 2 outliers (-7.3 → -1.9), perturbed duplicate rows.

**L2 (spectral surgery):** SVD denoised attention layers (condition 10K-30K → 27-35),
memory tier 2 (condition 1.4B → 1,000). Pure linear algebra, no GPU needed.

**L3 (distillation):** Reset all 5 memory tiers with Xavier init. Froze transformer,
distilled from working layers. Eliminated identical_rows, heavy_tails, memory gradient_noise.

**Gate opening:** Set read_gate/write_gate from -5 to 0 (0.7% → 50%). Trained only
memory (5.2M params) with next-token prediction loss. Loss curve:
- Step 0: 9.44 (memory hurting — worse than no-memory)
- Step 100: 4.70 (memory adapting)
- Step 300: 4.01 (memory contributing)
- Step 700: 3.69 (best loss)
- Step 999: 3.84 (slight overfit)

### Key findings

1. **Gate initialization matters more than architecture.** The same memory banks that
   looked "dead" for 645 hours worked perfectly once the gates were opened. The problem
   was never the memory design — it was that sigmoid(-5) gates produce ~0 gradient.

2. **The gate learned its own operating point.** Starting at 50%, it drifted to 52%.
   It didn't collapse to 0 (shut off) or 1 (fully open). The model found equilibrium.

3. **5.2M params can dominate 334M frozen params.** The memory contribution was large
   enough to cut perplexity 251x despite being <2% of total parameters. This validates
   the wrapper architecture: small trainable additions on top of a frozen backbone.

4. **Loss was still dropping at step 700.** The 1000-step run may be undertrained.
   5000+ steps could push PPL below 20.

### Implications for ATLESQwen

The ATLESQwen model uses the same memory architecture with the same gating fix
(sigmoid(-5) initialization). Based on these results:
- **Change gate init from -5 to 0** for Phase 1 training
- Memory should contribute from step 1 instead of being starved for 20K steps
- The wrapper architecture (small trainable modules on frozen Qwen) is validated

### Remaining work

1. **More training steps** — loss still dropping at step 700, best=3.69. Try 5000 steps.
2. **Unfreeze embeddings** — still collapsed at rank 11/256. Could recover more capacity.
3. **Run model-clinic on gates-open checkpoint** — get updated health score.
4. **Generation quality test** — PPL improved but does the model produce coherent text?
5. **Apply to ATLESQwen v2** — restart Phase 1 with gate_init=0.

### Scripts

- `scripts/surgical_repair.py` — L1+L2+L3 static + distillation repair
- `scripts/open_gates.py` — Gate opening + memory training

### Checkpoints

- Original: `~/atles-prototype/checkpoints/merged_final.pt`
- After L1+L2+L3: `~/atles-prototype/checkpoints/merged_final_REPAIRED.pt`
- After gate opening: `~/atles-prototype/checkpoints/merged_final_GATES_OPEN.pt`

## Problem Statement

Current model-clinic treatments (v0.3) are surface-level: clamp outliers, perturb
duplicate rows, reset norms. They improve health scores but don't fix the underlying
damage. A model with dead memory banks, collapsed embeddings, or near-singular
attention layers needs structural repair — not cosmetics.

This document specifies 4 new levels of repair depth, each building on the last.

---

## Level 2: Spectral Surgery

**Principle:** A weight matrix W = UΣV^T. The singular values Σ tell you which
directions carry signal vs noise. Near-zero singular values are noise — they contribute
nothing useful but amplify gradients and hurt conditioning. Truncating them denoises
the matrix without ever running the model.

**What it fixes:**
- gradient_noise (high condition numbers): removing near-zero SVs directly fixes this
- Poorly conditioned attention layers (condition number 710K → ~1K after truncation)
- Memory banks with 16M condition numbers → clean, usable matrices

**Implementation:**

```python
def spectral_denoise(tensor, energy_threshold=0.99, max_condition=1000):
    """SVD-based denoising. Keep singular values that capture `energy_threshold`
    of total spectral energy, and cap condition number at `max_condition`."""
    U, S, Vt = torch.linalg.svd(tensor, full_matrices=False)

    # Method 1: Energy-based rank selection
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
    rank = torch.searchsorted(cumulative_energy, energy_threshold).item() + 1

    # Method 2: Condition number cap
    max_rank_by_cond = (S > S[0] / max_condition).sum().item()

    # Take the more conservative (lower) rank
    effective_rank = min(rank, max_rank_by_cond)

    # Reconstruct with truncated SVD
    return U[:, :effective_rank] @ torch.diag(S[:effective_rank]) @ Vt[:effective_rank, :]
```

**Parameters:**
- `energy_threshold` (float, default 0.99): fraction of spectral energy to preserve
- `max_condition` (float, default 1000): maximum allowed condition number after surgery
- `min_rank_ratio` (float, default 0.1): never truncate below 10% of original rank

**New prescription action:** `spectral_denoise`
**Triggered by:** `gradient_noise` findings with condition_number > 10K

**Validation:** After surgery, re-check condition number. Must be < max_condition.
Compare Frobenius norm of (W_original - W_denoised) / ||W_original|| — should be < 5%.

**Risk:** medium. Changing the effective rank changes the layer's capacity. The
energy_threshold parameter controls aggressiveness.

---

## Level 3: Targeted Re-initialization with Knowledge Distillation

**Principle:** When a module is truly dead (memory gates stuck at 5%, all rows identical,
drives with zero importance), patching individual values won't help. Better to reset it
and use the working parts of the model as a teacher.

**What it fixes:**
- Dead memory banks (all tiers with identical rows, stuck gates)
- Inert drive systems (zero importance biases)
- Modules that trained but learned nothing useful

**Implementation:**

```python
def distill_repair(state_dict, dead_modules, calibration_data,
                   model_class, num_steps=200, lr=1e-4):
    """
    1. Load model from state_dict
    2. Capture activations from working layers on calibration_data (teacher signal)
    3. Re-initialize dead modules with Xavier/Kaiming init
    4. Freeze all non-dead modules
    5. Train dead modules for `num_steps` to match teacher activations
    """
    model = model_class.from_state_dict(state_dict)

    # Capture teacher activations (frozen, no grad)
    teacher_acts = capture_activations(model, calibration_data, layers=working_layers)

    # Reset dead modules
    for name in dead_modules:
        module = get_module(model, name)
        reset_parameters(module)  # Xavier uniform for linear, etc.

    # Freeze everything except dead modules
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(dm) for dm in dead_modules)

    # Distillation: match teacher activations
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    for step in range(num_steps):
        student_acts = capture_activations(model, calibration_data, layers=working_layers)
        loss = sum(F.mse_loss(s, t) for s, t in zip(student_acts, teacher_acts))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model.state_dict()
```

**Requirements:**
- Small calibration dataset (100-1000 samples, ~1MB)
- Model class that can load from state_dict
- GPU recommended but CPU works for small models
- ~10 minutes on RTX 3060 for 300M model

**What makes this different from retraining:**
- Only dead modules get gradients (10% of params typically)
- Teacher signal comes from the model's own working layers, not ground truth labels
- 200 steps, not 20,000. Physical therapy, not boot camp.

**New prescription action:** `distill_repair`
**Triggered by:** Combination of identical_rows + gradient_noise + model_aging in
the same module subtree.

**Requires:** `model_class` parameter and `calibration_data` path. Cannot be fully
automatic — user must confirm.

---

## Level 4: Cross-Checkpoint Grafting

**Principle:** Different training runs produce different failure modes. Run A might
have great attention but dead memory. Run B might have working memory but degraded
attention. Score each layer independently, take the best version of each.

**What it fixes:**
- Situations where you have multiple checkpoints and none is perfect
- "Best of both worlds" merging

**Implementation:**

```python
def graft(checkpoints, reference_data=None):
    """
    1. Load all checkpoints
    2. For each parameter/module, compute health score independently
    3. For each parameter, pick the version with the best health score
    4. Optionally: validate with reference_data (perplexity check)
    5. Return merged state_dict + manifest showing which layer came from where
    """
    all_sds = [load_state_dict(cp) for cp in checkpoints]
    merged = {}
    manifest = {}

    for key in all_sds[0].keys():
        candidates = [(i, sd[key]) for i, sd in enumerate(all_sds) if key in sd]

        # Score each candidate
        scores = []
        for idx, tensor in candidates:
            findings = diagnose({key: tensor})
            score = compute_health_score(findings)
            scores.append((score.overall, idx, tensor))

        # Pick best
        scores.sort(reverse=True)
        best_score, best_idx, best_tensor = scores[0]
        merged[key] = best_tensor
        manifest[key] = {
            "source_checkpoint": checkpoints[best_idx],
            "score": best_score,
            "runner_up_score": scores[1][0] if len(scores) > 1 else None
        }

    return merged, manifest
```

**New CLI command:** `model-clinic graft checkpoint1.pt checkpoint2.pt -o merged.pt`

**Output:** Merged checkpoint + JSON manifest showing provenance of every layer.

**Validation:** Run diagnose on merged result. Score should be >= max(individual scores).
Optionally run perplexity check if model class and data available.

**Risk:** low for same-architecture checkpoints. The main risk is that independently
healthy layers might not compose well (distribution mismatch between layers from
different training stages). The perplexity check catches this.

---

## Level 5: Activation-Guided Repair

**Principle:** Weight-level analysis can only tell you about the matrix properties.
To know if a layer is actually *destructive* — making representations worse — you
need to run data through it and measure.

**What it fixes:**
- Layers that look fine statically but destroy information at runtime
- Representation collapse (all tokens map to similar vectors)
- Entropy pathologies (layer increases/decreases entropy too aggressively)
- Layers where removing them would *improve* the model (negative contribution)

**Implementation:**

```python
def activation_audit(model, calibration_data, num_samples=100):
    """Run calibration data through the model and measure per-layer impact."""
    hooks = {}
    stats = {}

    def make_hook(name):
        def hook(module, input, output):
            inp = input[0] if isinstance(input, tuple) else input
            out = output[0] if isinstance(output, tuple) else output
            stats[name] = {
                "input_rank": effective_rank(inp),
                "output_rank": effective_rank(out),
                "rank_change": effective_rank(out) - effective_rank(inp),
                "input_entropy": token_entropy(inp),
                "output_entropy": token_entropy(out),
                "entropy_change": token_entropy(out) - token_entropy(inp),
                "cosine_similarity": F.cosine_similarity(
                    inp.flatten(), out.flatten(), dim=0
                ).item(),
                "norm_ratio": out.norm() / (inp.norm() + 1e-8),
            }
        return hook

    # Register hooks on all layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            hooks[name] = module.register_forward_hook(make_hook(name))

    # Run calibration data
    model.eval()
    with torch.no_grad():
        for batch in calibration_data[:num_samples]:
            model(batch)

    # Remove hooks
    for h in hooks.values():
        h.remove()

    return stats


def activation_repair(model, stats, strategy="passthrough"):
    """
    For layers identified as destructive:
    - "passthrough": replace with identity (skip connection)
    - "interpolate": replace with average of neighboring layers
    - "shrink": scale the layer's contribution by 0.1 (soft bypass)
    """
    destructive = [
        name for name, s in stats.items()
        if s["rank_change"] < -0.3 * s["input_rank"]  # kills >30% of rank
        or s["norm_ratio"] > 10  # explodes norms
        or s["norm_ratio"] < 0.1  # collapses norms
    ]

    for name in destructive:
        module = get_module(model, name)
        if strategy == "passthrough":
            replace_with_identity(model, name)
        elif strategy == "shrink":
            with torch.no_grad():
                for p in module.parameters():
                    p.mul_(0.1)
        elif strategy == "interpolate":
            neighbors = get_neighbor_layers(model, name)
            interpolate_weights(module, neighbors)

    return model.state_dict(), destructive
```

**New CLI commands:**
- `model-clinic activation-audit model.pt --data calibration.jsonl`
- `model-clinic activation-repair model.pt --data calibration.jsonl --strategy shrink`

**Metrics tracked per layer:**
| Metric | Healthy Range | Destructive |
|--------|--------------|-------------|
| rank_change | -10% to +10% | < -30% |
| norm_ratio | 0.5 to 2.0 | < 0.1 or > 10 |
| entropy_change | -0.1 to +0.1 | < -0.5 |
| cosine_similarity | > 0.8 | < 0.3 |

**Requirements:**
- Model must be loadable (need model class, not just state_dict)
- Calibration data (100+ samples)
- GPU strongly recommended

**Risk:** high. Replacing or scaling layers can cascade. Always validate with
perplexity check after repair. Keep backup.

---

## Integration Plan

All levels integrate into the existing model-clinic architecture:

```
model_clinic/
  _repair/
    __init__.py           # Exports all repair functions
    spectral.py           # Level 2: SVD surgery
    distill.py            # Level 3: Knowledge distillation repair
    graft.py              # Level 4: Cross-checkpoint grafting
    activation.py         # Level 5: Activation-guided repair
  _tools/
    spectral_cmd.py       # CLI: model-clinic spectral
    graft_cmd.py          # CLI: model-clinic graft
    activation_cmd.py     # CLI: model-clinic activation-audit / activation-repair
```

**Prescription integration:** Levels 2 and 4 add new prescription actions to the
existing `prescribe()` function. Levels 3 and 5 are separate commands because they
require additional inputs (model class, calibration data).

**Dependencies:**
- Level 2: torch only (already required)
- Level 3: torch + model class + calibration data
- Level 4: torch only
- Level 5: torch + model class + calibration data

---

## Test Strategy

Each level gets its own test file using synthetic models from `_synthetic.py`:

- `test_spectral.py` — Create matrix with known condition number, verify surgery fixes it
- `test_distill.py` — Create model with dead module, verify distillation activates it
- `test_graft.py` — Create 2 checkpoints with complementary damage, verify merge is better
- `test_activation.py` — Create model with destructive layer, verify detection and repair

Plus integration test: run full pipeline (diagnose → spectral → distill → graft) on
the ATLES 300M model and verify score improvement at each step.
