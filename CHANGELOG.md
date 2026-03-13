# Changelog

All notable changes to model-clinic are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.0] - 2026-03-11

### Added

**Health Score**
- `compute_health_score()` — 0-100 score with letter grade (A-F) and per-category breakdown (weights, stability, output, activations)
- `print_health_score()` — formatted ASCII bar chart output
- Per-condition penalty caps: WARN conditions capped at 15 pts, ERROR at 35 pts per category
- `HealthScore` type exported from public API

**HTML Report**
- `model-clinic report checkpoint.pt --output report.html` — full diagnostic HTML report
- 7 inline SVG visualizations: weight distribution histograms, layer norm drift bar chart, neuron activation histograms, attention entropy heatmap, dead neuron heatmap, before/after treatment comparison charts, health score gauge
- Suggested-fixes box: actionable recommendations derived directly from findings
- Self-contained single-file output (no external dependencies at render time)

**New CLI Commands** (added to v0.2.0's 10)
- `model-clinic validate` — verify a checkpoint loads and all tensors are finite
- `model-clinic report` — generate HTML diagnostic report
- `model-clinic compare` — compare health delta between two checkpoints (before/after metrics)
- `model-clinic demo` — examine a synthetic broken model without needing a real checkpoint

**Training Monitor**
- `ClinicMonitor` — call `monitor.check(model)` inside training loops to detect gradient explosion, neuron death, and layer collapse mid-training
- `ClinicTrainerCallback` — drop-in HuggingFace Trainer callback; plugs into any `transformers` training run
- `MonitorAlert` and `MonitorSummary` types exported from public API

**Treatment Pipelines**
- `create_pipeline([rx1, rx2, ...])` — chain multiple prescriptions into a single atomic operation
- `TreatmentPipeline` class with `.run()`, `.dry_run()`, and rollback support
- `examine_batch(paths, parallel=True)` — diagnose multiple checkpoints in one call
- `PipelineResult` type exported from public API

**Treatment Manifest**
- `TreatmentManifest` — records what changed, why, before/after checksums, timestamps
- Saved alongside treated checkpoint as `<name>.manifest.json`
- Enables full auditability of all applied fixes

**New Static Detectors** (expanded from 12 to 22)
- `moe_router_collapse` — detects expert imbalance in Mixture-of-Experts models (excludes non-MoE gate projections)
- `lora_merge_artifacts` — rank collapse detection via effective rank estimation (skips base models with no LoRA keys)
- `quantization_degradation` — post-GPTQ/AWQ/INT8/FP8 degradation checks (bf16 false-positive fixed)
- `head_redundancy` — identical or near-identical attention heads
- `token_collapse` — model consistently predicts the same token distribution
- `model_aging` — catastrophic forgetting / distribution drift from pretraining
- `representation_drift` — norm-based drift between consecutive layers
- `gradient_noise` — condition number estimation from weight SVD (embedding tables excluded; threshold 50K)
- `positional_encoding_issues` — anomalous positional embedding rows

**Synthetic Models**
- `make_healthy_mlp()` — deterministic healthy reference model for tests and demos
- `make_everything_broken()` — model with every detectable condition injected
- `SYNTHETIC_MODELS` dict — 11 named presets: healthy, dead-neurons, nan, exploding, norm-drift, collapsed, heavy-tails, duplicate-rows, stuck-gates, corrupted, everything-broken
- CLI: `model-clinic demo <preset>`, `model-clinic demo --list`, `model-clinic demo <preset> --treat`

**References System**
- Every condition links to relevant papers and guides via `get_references(condition)`
- Shown in `--explain` output and HTML reports

**CI / Packaging**
- `action.yml` — GitHub Actions CI plugin; gates PRs on model health score threshold
- JSON Schema for exam/report output (published in `schemas/`)
- `--explain` flag for `exam` and `treat` — shows why each fix is recommended and what the risk is
- `--verbose` / `-v` flag with per-detector progress indication
- `--example-prompts` flag bundles diverse test prompts for `--runtime` testing

**Calibration Fixes** (detector quality)
- `norm_drift` threshold raised from 0.5 to 1.5 (pretraining naturally drifts norms)
- `gradient_noise` threshold raised from 1K to 50K; embedding tables excluded
- `moe_router_collapse` excludes `gate_proj`/`gate_up` parameter names (non-MoE gates)
- `lora_merge_artifacts` skips checkpoints with no LoRA keys
- `quantization_degradation` no longer fires false positives on bf16 models

### Changed
- Health score penalty system uses per-condition caps to prevent single bad detector from dominating the score
- `--json` output now validates against published JSON Schema
- `pyproject.toml`: added `[project.urls]`, additional classifiers (Python 3.10/3.11/3.12, OS Independent, Debuggers, Typed), `wandb`/`mlflow`/`tensorboard` optional extras, sdist include list

### Stats
- 22 static detectors (was 12)
- 6 runtime detectors (unchanged)
- 322 tests passing (was 57)
- 15 CLI commands (was 10)
- 34 public API exports
- 27 source files, ~7600 lines of source
- 18 test files, ~3400 lines of tests

---

## [0.2.0] - 2025-12-01

### Added

**Static Detectors** (12 total)
- `nan_inf` — NaN or Inf values in any tensor
- `dead_neurons` — rows/columns with all-zero or near-zero activation
- `stuck_gate_closed` — gating parameters stuck in off position
- `stuck_gate_open` — gating parameters saturated in on position
- `exploding_norm` — parameter tensors with catastrophically large norms
- `vanishing_norm` — near-zero norms indicating dead or uninitialized parameters
- `heavy_tails` — kurtosis-based outlier detection (normal kurtosis ≈ 3)
- `norm_drift` — LayerNorm weight deviation from expected baseline of 1.0
- `saturated_weights` — tanh/sigmoid weights pushed into flat saturation region
- `identical_rows` — duplicate rows/columns indicating broken weight tying
- `attention_imbalance` — Q/K/V projection norm imbalance
- `dtype_mismatch` — mixed precision tensors within the same model

**Runtime Detectors** (6 total, requires transformers)
- `generation_collapse` — output token entropy below threshold (model always predicts same token)
- `low_coherence` — coherence scoring across diverse prompts
- `activation_nan_inf` — NaN/Inf in intermediate activations (hook-based)
- `activation_explosion` — layer activation norms far exceeding input norms
- `residual_explosion` — residual stream growing unboundedly through layers
- `weight_corruption` — all-zero tensors, constant tensors, majority-identical values

**Treatment System**
- `prescribe(findings)` — generate `Prescription` objects from findings
- `apply_treatment(state_dict, rx)` — apply a single prescription with before/after stats
- `rollback_treatment(state_dict, result)` — restore original values if treatment made things worse
- Conservative mode (`--conservative`) — only applies LOW risk fixes
- Dry-run mode (`--dry-run`) — shows what would change without modifying anything

**CLI Tools** (10 total)
- `model-clinic exam` — full diagnosis with optional treatment plan
- `model-clinic treat` — diagnose and apply fixes
- `model-xray` — per-parameter weight stats (shape, norm, sparsity, dtype)
- `model-diff` — parameter-by-parameter comparison between two checkpoints
- `model-health` — quick health check focused on dead neurons, norms, gates
- `model-surgery` — direct parameter modification (interactive REPL)
- `model-ablate` — systematically disable components and measure impact
- `model-neurons` — profile neuron activations across prompts
- `model-attention` — attention patterns per head per layer
- `model-logit-lens` — watch predictions form layer by layer

**Checkpoint Loading**
- `load_state_dict()` — unified loader for `.pt`/`.pth`, safetensors, HuggingFace hub, composite dicts
- `load_model()` — load full `nn.Module` with optional HuggingFace Auto classes
- `build_meta()` — extract `ModelMeta` (param count, dtype breakdown, architecture hints)
- `save_state_dict()` — save with optional safetensors format
- Nested checkpoint support: `model_state_dict`, `state_dict`, `module` keys auto-unwrapped
- Composite checkpoint support: multiple named state dicts in one file

**Output Formats**
- `--json` flag — machine-readable JSON output for CI pipelines
- `--quiet` flag — suppress progress output, results only

**Python API**
- `Finding`, `Prescription`, `TreatmentResult`, `ExamReport`, `ModelMeta` types
- `diagnose(state_dict)` — run all detectors, return list of `Finding`
- `REGISTRY` — custom condition registry for adding user-defined detectors and prescribers

**Extras**
- `[hf]` — HuggingFace transformers for runtime analysis and hub loading
- `[safetensors]` — safetensors file format support
- `[all]` — all optional dependencies

---

[0.3.0]: https://github.com/spartan8806/model-clinic/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/spartan8806/model-clinic/releases/tag/v0.2.0
