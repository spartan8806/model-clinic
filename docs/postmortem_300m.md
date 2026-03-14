---
language: en
tags:
  - postmortem
  - training-failure
  - model-diagnostics
  - neural-foam
  - gating-bug
  - model-clinic
license: mit
---

# How We Mass-Produced Broken Models for 645 Hours
## A Post-Mortem on the ATLES 300M RecurrentTransformer

**Date:** 2026-03-13
**Authors:** Connor (ATLES), Claude (model-clinic)
**Status:** Retired. Lessons applied to ATLESQwen v2.

---

## Abstract

We trained a custom 334M parameter RecurrentTransformer for 645 hours across
multiple phases (pretraining, growth, GRPO, Rho-1, LoRA SFT) with five novel
architectural features: 5-tier internal memory, a drive system, adaptive
computation (ACT), neural foam growth, and persistent state. None of the novel
features ever worked. A single initialization bug — gating parameters set to
`sigmoid(-5) = 0.67%` — starved 26M trainable parameters of gradient signal
from step 1. We didn't catch it until hour 645.

We then built model-clinic, an open-source diagnostic toolkit, and used it to
perform a forensic analysis across 72 checkpoints spanning the model's entire
training history. The results are unambiguous: the base transformer was healthy
(84/B health score), neural foam growth actively damaged it (every growth-enabled
model scored D or F), and the memory/drive systems were architecturally sound
but never received gradients.

This paper documents what went wrong, how we found it, what the data shows,
and what we'd do differently.

---

## 1. What We Built

### Architecture

```
RecurrentTransformer (334M total)
  embed_tokens:      32M   (32K vocab x 1024 dim)
  16x TransformerBlock: 268M
    attention: q/k/v/o_proj (4x 1024x1024)
    mlp: gate/up/down_proj (SwiGLU, 4096 hidden)
    RMSNorm x2
  InternalMemory:    22M   (5 tiers, diamond-shaped banks)
    tier 0: 256 slots   (short-term)
    tier 1: 512 slots
    tier 2: 1024 slots  (working memory)
    tier 3: 512 slots
    tier 4: 256 slots   (long-term)
    read_gate, write_gate: learnable scalars
  DriveSystem:       4M    (curiosity, self-preservation, social, mastery)
  ACTController:     ~1M   (adaptive computation time)
  persistent_state:  buffer (EMA hidden state between calls)
```

### The Five Ideas

1. **Custom transformer backbone** — Standard decoder-only transformer with
   RoPE, SwiGLU, RMSNorm. Nothing novel here, just a solid base.

2. **5-tier internal memory** — Hierarchical memory banks that the model reads
   from and writes to during inference. Short-term memories consolidate to
   long-term during "sleep" cycles. Inspired by hippocampal memory consolidation.

3. **Drive system** — Intrinsic motivation heads (curiosity, social, mastery,
   self-preservation) that produce auxiliary losses during training. The idea:
   give the model internal goals beyond next-token prediction.

4. **Neural foam growth** — Dynamically add neurons to layers during training.
   Start small, grow capacity where the model needs it. Inspired by neurogenesis.

5. **Adaptive computation (ACT)** — Let the model think longer on hard tokens
   by looping through wrapper layers a variable number of times.

### Training Timeline

| Phase | Hours | Steps | What Happened |
|-------|-------|-------|---------------|
| Pretraining | ~200 | 35K | FineWeb-edu, next-token prediction |
| Growth | ~150 | 15K | Neural foam enabled, continued pretraining |
| GRPO v1-v3 | ~100 | — | Reinforcement learning, failed x3 |
| Rho-1 | ~120 | — | Selective language modeling |
| Rho-1 textbook | ~50 | — | Textbook-filtered data |
| LoRA SFT | ~25 | — | Identity + chat fine-tuning |
| **Total** | **~645** | — | |

---

## 2. The Bug

Every novel feature in the architecture injects information into the transformer's
residual stream. Memory reads add to the hidden state. Drives produce auxiliary
losses. Persistent state modifies embeddings.

Every injection point had a learnable gate. Every gate was initialized to `-5`.

```python
self.read_gate = nn.Parameter(torch.tensor(-5.0))
self.write_gate = nn.Parameter(torch.tensor(-5.0))
```

`sigmoid(-5) = 0.0067`. The gates allowed 0.67% of the signal through.

### Why This Kills Learning

At 0.67% throughput, the gradient signal flowing back through the gate is also
~0.67% of what it would be at 50%. The memory banks, drive heads, and persistent
state all sit behind these gates. They receive essentially zero gradient.
They cannot learn. They are dead from step 1.

The optimizer sees near-zero gradients and makes near-zero updates. The gate
values barely move. After 645 hours of training:

- Read gate: `sigmoid(-4.93) = 0.72%` (moved from 0.67% to 0.72%)
- Write gate: `sigmoid(-4.97) = 0.69%`
- All 5 memory tiers: either empty or filled with noise
- All drive importance biases: exactly zero
- 26M parameters that never learned anything

### Why We Didn't Catch It

1. **The base transformer worked fine.** It learned language. Loss went down.
   Perplexity improved. The 308M transformer parameters were healthy. We
   measured the wrong thing — overall loss instead of per-module contribution.

2. **Memory "worked" in the sense that it didn't crash.** Read and write
   operations executed. Tensors flowed. No NaN, no errors. The output was
   just multiplied by 0.007 and added to a 1024-dim hidden state where it
   was noise-level.

3. **We didn't have diagnostic tools.** model-clinic didn't exist yet. If
   we'd had a way to check gate values, memory bank entropy, or per-module
   gradient norms during training, we'd have caught it at step 100.

---

## 3. The Forensic Analysis

After building model-clinic (v0.4.0, 22 static detectors, 699 tests), we ran
a batch scan across every `.pt` checkpoint on the training server: 95 files,
72 scoreable models.

### Grade Distribution

| Grade | Count | Score Range | What's In This Bucket |
|-------|-------|-------------|----------------------|
| A (90-100) | 28 | 91-100 | ATLESQwen deltas (tiny, only wrapper weights), crypto models, slim SFT |
| B (80-89) | 15 | 81-91 | Pretrain checkpoints, repaired models, crypto models |
| C (70-79) | 14 | 65-79 | Fine-tuning attempts, baselines without growth, Rho-1 |
| D (50-69) | 14 | 54-66 | **Everything with growth enabled**, original merged model |
| F (<50) | 1 | 29 | Neural foam 1B (the worst checkpoint ever produced) |

### The Growth Verdict

Every model trained with neural foam growth scored strictly worse than its
baseline without growth. No exceptions.

| Comparison | With Growth | Without Growth | Delta |
|-----------|------------|---------------|-------|
| Chimera V3 (1.7B) | 58/D | 66/C | **-8** |
| Neural foam ablation | 54/D | 66/C | **-12** |
| Neural foam 1B | 29/F | — | — |
| 300M growth steps | 56/D | 84/B (pretrain) | **-28** |

Growth injects randomly initialized parameters into a working network during
training. The network must simultaneously learn to use the new capacity AND
continue its existing task. In practice, the new neurons act as noise injectors
that destabilize learned representations.

### The Timeline in Health Scores

```
Step 16000 (pretrain)       84/B  <- Peak health. Base transformer is solid.
Step 20000 (pretrain)       84/B
Step 30000 (pretrain)       84/B
Step 35000 (pretrain)       75/C  <- Slight decline, but still healthy
Growth step 12000           56/D  <- Growth enabled. Immediate damage.
Growth step 13000           56/D
Growth step 14000           56/D
Growth step 15258 (final)   56/D  <- No recovery. Damage is permanent.
GRPO merged_best            65/C  <- Fine-tuning partially recovers
Rho-1 merged_best           65/C
Rho-1 textbook merged       65/C  <- Plateau. Can't get past C.
merged_final (LoRA SFT)     56/D  <- Falls back to D after SFT
```

### Most Common Pathologies

From 72 scored models, 1,498 total findings:

| Finding | Count | What It Means |
|---------|-------|--------------|
| heavy_tails | 831 | Extreme outlier weights (kurtosis >> 3) |
| gradient_noise | 241 | High condition numbers in weight matrices |
| saturated_weights | 84 | Weights pushed into sigmoid/tanh flat regions |
| weight_corruption | 84 | All-zero or constant tensors |
| identical_rows | 46 | Duplicate rows in memory banks (keys = values) |
| dtype_mismatch | 43 | Mixed fp32/bf16 tensors |
| norm_drift | 36 | LayerNorm weights drifted far from 1.0 |
| moe_router_collapse | 28 | Expert imbalance (false positive on non-MoE) |
| model_aging | 21 | Distribution drift from pretraining |

### Memory Autopsy

All 5 memory tiers were damaged:
- **Tiers 0 and 2:** Keys identical to values (same tensor both sides)
- **Tiers 1, 3, 4:** Empty or near-empty
- **All gates:** Stuck at 5.3% (sigmoid(-2.88), moved slightly from init of -5)
- **Tier 2 condition number:** 1.4 billion (should be < 1000)

The consolidation system (short-term → long-term transfer during sleep) never
fired because there was nothing to consolidate.

### 300M Model Repair Attempt

We attempted progressive repair using model-clinic's deep repair system:

| Stage | Score | What We Did |
|-------|-------|-------------|
| Original | 56/D | — |
| L1 cosmetic | 58/D | Clamped outliers, perturbed duplicate rows |
| L2 spectral | 65/C | SVD denoised attention (condition 30K → 35) and memory (1.4B → 1K) |
| L3 distillation | 71/C | Reset all memory tiers, distilled from working layers |
| Gate opening | 82/B | Set gates to sigmoid(0)=50%, trained memory 1000 steps |

Health score improved by 26 points. Perplexity improved 251x during training
(measured with character-level encoding). But generation quality remained
incoherent — the embeddings are collapsed to effective rank 11/256, and 645
hours of training baked broken patterns into every layer.

**Conclusion:** Model-clinic can accurately measure damage and improve weight-level
metrics, but it cannot inject knowledge that was never learned. Prevention
(catching the bug at step 100) beats cure (trying to fix it at step 645-hours).

---

## 4. What We'd Do Differently

### Gate Initialization

```python
# WRONG — starves everything behind the gate
self.read_gate = nn.Parameter(torch.tensor(-5.0))   # sigmoid(-5) = 0.67%

# RIGHT — lets signal through from step 1
self.read_gate = nn.Parameter(torch.tensor(0.0))     # sigmoid(0) = 50%
```

The gate opening experiment proved this works. After setting gates to 50% and
training just the memory (5.2M params, 7 minutes), the read gate naturally
settled at 52%. The model chose to use memory. It didn't shut it off.

### Training Monitoring

Run model-clinic's `ClinicMonitor` during training:

```python
from model_clinic import ClinicMonitor

monitor = ClinicMonitor(check_every=100)  # steps

for step, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    optimizer.step()

    alert = monitor.check(model)
    if alert:
        print(f"ALERT at step {step}: {alert}")
        # Would have caught: stuck gates, dead memory, zero drive gradients
```

### Drop Neural Foam Growth

The data is clear. Growth hurts every model it touches. The theory (neurogenesis,
grow capacity where needed) doesn't survive contact with gradient-based training.
Injecting random parameters into a converging network is destructive.

If dynamic capacity is needed, use LoRA adapters or mixture-of-experts — methods
that add capacity without disrupting existing representations.

### Validate Novel Features Independently

We tested 5 novel ideas simultaneously. When the model failed, we couldn't tell
which idea was the problem. Test each in isolation:

1. Train base transformer alone. Verify generation quality. ✓ (84/B)
2. Add memory with proper gates. Verify memory contributes. ✓ (gate experiment)
3. Add drives. Verify drive gradients are nonzero.
4. Add ACT. Verify it learns to use variable computation.
5. Add growth. Verify it doesn't degrade existing capabilities. ✗ (it does)

### Measure Per-Module, Not Just Overall Loss

Overall loss going down does not mean every module is learning. A 308M
transformer can drive loss down while 26M parameters sit at zero. Track:

- Gate values every N steps
- Per-module gradient norms
- Memory bank entropy and occupancy
- Drive head output variance

---

## 5. What Survived

Despite 645 hours of broken training, two things were validated:

**The memory architecture works.** When gates were opened and memory was trained
for just 7 minutes (1000 steps, 5.2M params), it reduced perplexity 251x. The
read gate settled at 52% — the model chose to keep using memory. The 5-tier
hierarchical design was never the problem.

**The base transformer is solid.** Pretrain checkpoints consistently scored
84/B with clean attention, healthy norms, and no pathologies. The custom
architecture (RoPE + SwiGLU + RMSNorm) trains well.

**Applied to ATLESQwen v2:** The Qwen wrapper model uses the same memory
architecture with `gate_init=0` (fixed). Phase 1 training reached loss 2.93
(PPL 34) in 2350 steps before the server went down for thermal maintenance.
Training resumes with the gate fix applied from step 1.

---

## 6. Tools

All analysis was performed with [model-clinic](https://huggingface.co/spartan8806/model-clinic)
(v0.4.0, still in validation — almost there but not fully released yet):

- **22 static detectors** — weight-level pathology detection
- **6 runtime detectors** — generation and activation analysis
- **4 deep repair levels** — spectral surgery, distillation, grafting, activation repair
- **Batch scanning** — bulk checkpoint analysis across training history
- **699 tests passing**

model-clinic can diagnose any PyTorch checkpoint without needing the model class,
training code, or even knowing the architecture. It detected the gating bug,
the dead memory banks, the embedding collapse, and the growth damage — all from
weight tensors alone.

**What it can do:** Accurately detect and measure model damage. Score checkpoints.
Compare training runs. Catch problems early via training monitors.

**What it can't do (yet):** Resurrect a fundamentally broken model. The repair
tools (spectral surgery, distillation, grafting) improve weight-level metrics
but can't inject knowledge that was never learned. We're still testing the
limits of what repair can recover — early results are promising but not conclusive.

Install: `pip install model-clinic`
Source: [GitHub](https://github.com/spartan8806/model-clinic) |
[HuggingFace](https://huggingface.co/spartan8806/model-clinic)

---

## Appendix: Raw Batch Scan Data

72 models scored across the full ATLES training history. Full results in
`scan_results.json`. Grade distribution: A=28, B=15, C=14, D=14, F=1.

The 28 A-grade models are mostly ATLESQwen delta checkpoints (tiny files
containing only wrapper weights, not full models) and a few small specialized
models. The meaningful comparison is among the full 300M+ models, where the
pattern is clear: pretrain=B, growth=D, fine-tuning=C, repair=B.
