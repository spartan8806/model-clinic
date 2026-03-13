"""Tests for new detectors: weight_corruption, head_redundancy, positional_encoding_issues, and Tier 3."""

import torch
from model_clinic._types import Finding
from model_clinic.clinic import (
    detect_weight_corruption, detect_head_redundancy,
    detect_positional_issues, post_detect_head_redundancy,
    detect_token_collapse, detect_gradient_noise,
    _collect_layer_norms, post_detect_representation_drift,
    detect_moe_router_collapse, detect_lora_merge_artifacts,
    detect_quantization_degradation, _collect_model_aging,
    post_detect_model_aging,
    diagnose,
)
from model_clinic._health_score import CATEGORY_MAP, _categorize


class TestWeightCorruption:
    """Test the weight_corruption detector."""

    def test_all_zeros(self):
        t = torch.zeros(64, 32)
        findings = detect_weight_corruption("layer.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "weight_corruption"
        assert findings[0].details["reason"] == "all_zeros"
        assert findings[0].severity == "ERROR"

    def test_constant_value(self):
        t = torch.full((64, 32), 0.5)
        findings = detect_weight_corruption("layer.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "constant_value"

    def test_majority_same_value(self):
        t = torch.zeros(200)
        t[:50] = torch.randn(50)  # only 25% different
        findings = detect_weight_corruption("layer.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "majority_same_value"
        assert findings[0].details["fraction"] > 0.5

    def test_healthy_weights_no_finding(self):
        torch.manual_seed(42)
        t = torch.randn(64, 32)
        findings = detect_weight_corruption("layer.weight", t, {})
        assert len(findings) == 0

    def test_skips_small_tensors(self):
        t = torch.zeros(10)
        findings = detect_weight_corruption("small.param", t, {})
        assert len(findings) == 0

    def test_skips_bias_tensors(self):
        t = torch.zeros(200)
        findings = detect_weight_corruption("layer.bias", t, {})
        assert len(findings) == 0

    def test_skips_norm_tensors(self):
        t = torch.full((200,), 1.0)
        findings = detect_weight_corruption("final_norm.weight", t, {})
        assert len(findings) == 0


class TestHeadRedundancy:
    """Test the head_redundancy detector (per-tensor + post-scan)."""

    def test_collects_q_weights(self):
        ctx = {}
        t = torch.randn(128, 64)  # 2 heads of dim 64
        detect_head_redundancy("layers.0.attention.q_proj.weight", t, ctx)
        assert "_q_proj_weights" in ctx
        assert "layers.0.attention" in ctx["_q_proj_weights"]

    def test_skips_non_q_proj(self):
        ctx = {}
        t = torch.randn(128, 64)
        detect_head_redundancy("layers.0.attention.k_proj.weight", t, ctx)
        assert len(ctx.get("_q_proj_weights", {})) == 0

    def test_redundant_heads_detected(self):
        """Two identical heads should be flagged."""
        head_dim = 64
        n_heads = 2
        # Make both heads identical
        block = torch.randn(head_dim, 64)
        q_tensor = torch.cat([block, block], dim=0)
        ctx = {"_q_proj_weights": {"layers.0.attention": q_tensor.float()},
               "meta": {"hidden_size": 64}}
        findings = post_detect_head_redundancy(ctx)
        assert len(findings) == 1
        assert findings[0].condition == "head_redundancy"
        assert findings[0].details["num_heads"] == 2

    def test_distinct_heads_no_finding(self):
        """Two different heads should not be flagged."""
        torch.manual_seed(42)
        head_dim = 64
        q_tensor = torch.randn(128, 64)  # 2 distinct heads
        ctx = {"_q_proj_weights": {"layers.0.attention": q_tensor.float()},
               "meta": {"hidden_size": 64}}
        findings = post_detect_head_redundancy(ctx)
        assert len(findings) == 0

    def test_post_detect_empty_context(self):
        findings = post_detect_head_redundancy({})
        assert len(findings) == 0


class TestPositionalEncodingIssues:
    """Test the positional_encoding_issues detector."""

    def test_all_zeros_position_embedding(self):
        t = torch.zeros(512, 64)
        findings = detect_positional_issues("model.position_embeddings.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "all_zeros"
        assert findings[0].severity == "ERROR"

    def test_nan_in_rotary(self):
        t = torch.randn(64, 32)
        t[0, 0] = float("nan")
        findings = detect_positional_issues("layers.0.rotary_emb.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "nan_or_inf"

    def test_inf_in_rope(self):
        t = torch.randn(64, 32)
        t[0, 0] = float("inf")
        findings = detect_positional_issues("layers.0.rope.freqs", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "nan_or_inf"

    def test_constant_positions(self):
        t = torch.full((64, 32), 0.5)
        findings = detect_positional_issues("model.position_embed.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "constant_value"

    def test_duplicate_position_rows(self):
        t = torch.randn(64, 32)
        t[1] = t[0].clone()  # position 1 = position 0
        findings = detect_positional_issues("model.position_embedding.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details["reason"] == "duplicate_positions"

    def test_healthy_positions_no_finding(self):
        torch.manual_seed(42)
        t = torch.randn(512, 64)
        findings = detect_positional_issues("model.position_embed.weight", t, {})
        assert len(findings) == 0

    def test_skips_non_positional_tensors(self):
        t = torch.zeros(512, 64)
        findings = detect_positional_issues("model.attention.weight", t, {})
        assert len(findings) == 0


class TestNewDetectorCategories:
    """Verify new conditions are mapped in the health score CATEGORY_MAP."""

    def test_weight_corruption_in_weights(self):
        assert "weight_corruption" in CATEGORY_MAP["weights"]
        assert _categorize("weight_corruption") == "weights"

    def test_head_redundancy_in_weights(self):
        assert "head_redundancy" in CATEGORY_MAP["weights"]
        assert _categorize("head_redundancy") == "weights"

    def test_positional_encoding_in_stability(self):
        assert "positional_encoding_issues" in CATEGORY_MAP["stability"]
        assert _categorize("positional_encoding_issues") == "stability"


class TestNewDetectorsInDiagnose:
    """Verify new detectors run as part of full diagnosis."""

    def test_weight_corruption_found_in_diagnose(self):
        sd = {
            "layer.weight": torch.zeros(64, 32),
            "healthy.weight": torch.randn(64, 32),
        }
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "weight_corruption" in conditions

    def test_positional_issues_found_in_diagnose(self):
        sd = {
            "model.position_embedding.weight": torch.zeros(64, 32),
            "healthy.weight": torch.randn(64, 32),
        }
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "positional_encoding_issues" in conditions


# ── Tier 3 detector tests ────────────────────────────────────────────────


class TestTokenCollapse:
    """Test the token_collapse detector."""

    def test_identical_lm_head_rows(self):
        """All rows identical should flag token collapse."""
        row = torch.randn(64)
        t = row.unsqueeze(0).expand(100, -1).clone()
        findings = detect_token_collapse("lm_head.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "token_collapse"
        assert findings[0].severity == "WARN"
        assert findings[0].details["collapsed_pair_fraction"] > 0.10

    def test_distinct_lm_head_rows(self):
        """Distinct rows should not flag."""
        torch.manual_seed(42)
        t = torch.randn(100, 64)
        findings = detect_token_collapse("lm_head.weight", t, {})
        assert len(findings) == 0

    def test_skips_non_lm_head(self):
        """Should only check lm_head or output tensors."""
        row = torch.randn(64)
        t = row.unsqueeze(0).expand(100, -1).clone()
        findings = detect_token_collapse("layers.0.mlp.weight", t, {})
        assert len(findings) == 0

    def test_skips_1d(self):
        t = torch.randn(100)
        findings = detect_token_collapse("lm_head.weight", t, {})
        assert len(findings) == 0


class TestGradientNoise:
    """Test the gradient_noise detector."""

    def test_ill_conditioned_matrix(self):
        """Matrix with huge condition number should flag."""
        torch.manual_seed(42)
        # Create ill-conditioned matrix via SVD manipulation
        m, n = 64, 64
        u = torch.linalg.qr(torch.randn(m, m))[0]
        v = torch.linalg.qr(torch.randn(n, n))[0]
        s = torch.zeros(min(m, n))
        s[0] = 10000.0  # huge first singular value
        s[1:] = 0.001   # tiny rest
        t = u[:, :min(m, n)] @ torch.diag(s) @ v[:min(m, n), :]
        findings = detect_gradient_noise("layers.0.mlp.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "gradient_noise"
        assert findings[0].details["condition_number"] > 1000

    def test_well_conditioned_matrix(self):
        """Well-conditioned matrix should not flag."""
        torch.manual_seed(42)
        t = torch.randn(64, 64)
        findings = detect_gradient_noise("layers.0.mlp.weight", t, {})
        assert len(findings) == 0

    def test_skips_small_tensors(self):
        t = torch.randn(10, 10)
        findings = detect_gradient_noise("layer.weight", t, {})
        assert len(findings) == 0

    def test_skips_1d(self):
        t = torch.randn(2000)
        findings = detect_gradient_noise("layer.weight", t, {})
        assert len(findings) == 0


class TestRepresentationDrift:
    """Test representation drift post-scan detector."""

    def test_dramatic_norm_change(self):
        """Adjacent layers with 10x+ norm difference should flag."""
        ctx = {"_layer_weight_norms": {
            0: [0.01, 0.01, 0.01],
            1: [0.5, 0.5, 0.5],    # 50x jump
            2: [0.5, 0.5, 0.5],
        }}
        findings = post_detect_representation_drift(ctx)
        assert len(findings) == 1
        assert findings[0].condition == "representation_drift"
        assert findings[0].details["layer_a"] == 0
        assert findings[0].details["layer_b"] == 1
        assert findings[0].details["ratio"] > 10

    def test_stable_norms_no_finding(self):
        """Layers with similar norms should not flag."""
        ctx = {"_layer_weight_norms": {
            0: [0.5, 0.5],
            1: [0.6, 0.6],
            2: [0.55, 0.55],
        }}
        findings = post_detect_representation_drift(ctx)
        assert len(findings) == 0

    def test_empty_context(self):
        findings = post_detect_representation_drift({})
        assert len(findings) == 0

    def test_collector_populates_context(self):
        """_collect_layer_norms should populate the context dict."""
        ctx = {}
        t = torch.randn(64, 32)
        _collect_layer_norms("layers.0.mlp.weight", t, ctx)
        _collect_layer_norms("layers.1.mlp.weight", t, ctx)
        assert "_layer_weight_norms" in ctx
        assert 0 in ctx["_layer_weight_norms"]
        assert 1 in ctx["_layer_weight_norms"]


class TestMoERouterCollapse:
    """Test the moe_router_collapse detector."""

    def test_collapsed_router(self):
        """Router where one expert dominates should flag."""
        # One column has huge values, rest near zero
        t = torch.zeros(32, 8)
        t[:, 0] = 100.0  # all tokens route to expert 0
        findings = detect_moe_router_collapse("layers.0.moe.router.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "moe_router_collapse"
        assert findings[0].details["reason"] == "collapsed"

    def test_healthy_router(self):
        """Router with moderate entropy should not flag."""
        torch.manual_seed(42)
        t = torch.randn(32, 8) * 0.5  # moderate values, decent entropy
        findings = detect_moe_router_collapse("layers.0.moe.router.weight", t, {})
        # Should not flag as collapsed or near-uniform
        collapsed = [f for f in findings if f.details.get("reason") == "collapsed"]
        assert len(collapsed) == 0

    def test_skips_non_router(self):
        t = torch.randn(32, 8)
        findings = detect_moe_router_collapse("layers.0.mlp.weight", t, {})
        assert len(findings) == 0

    def test_skips_scalar_gate(self):
        """Scalar gates should be handled by stuck_gate detector, not this one."""
        t = torch.tensor(5.0)
        findings = detect_moe_router_collapse("layers.0.gate", t, {})
        assert len(findings) == 0

    def test_skips_1d_gate(self):
        """1D gate tensors should be skipped."""
        t = torch.randn(8)
        findings = detect_moe_router_collapse("layers.0.gate.weight", t, {})
        assert len(findings) == 0


class TestLoRAMergeArtifacts:
    """Test the lora_merge_artifacts detector."""

    def test_rank_deficient_matrix(self):
        """Low effective rank in attention projection should flag (with LoRA context)."""
        torch.manual_seed(42)
        # Create a rank-1 matrix (extreme LoRA artifact)
        a = torch.randn(64, 1)
        b = torch.randn(1, 64)
        t = a @ b  # rank 1
        # Provide LoRA context — detector skips base models without adapter keys
        ctx = {"_has_lora_keys": True}
        findings = detect_lora_merge_artifacts("layers.0.attention.q_proj.weight", t, ctx)
        assert len(findings) == 1
        assert findings[0].condition == "lora_merge_artifacts"
        assert findings[0].details["rank_ratio"] < 0.1

    def test_full_rank_matrix(self):
        """Full-rank matrix should not flag even with LoRA context."""
        torch.manual_seed(42)
        t = torch.randn(64, 64)
        ctx = {"_has_lora_keys": True}
        findings = detect_lora_merge_artifacts("layers.0.attention.q_proj.weight", t, ctx)
        assert len(findings) == 0

    def test_skips_non_attention_proj(self):
        """Should only check q/k/v/o_proj tensors."""
        torch.manual_seed(42)
        a = torch.randn(64, 1)
        b = torch.randn(1, 64)
        t = a @ b
        ctx = {"_has_lora_keys": True}
        findings = detect_lora_merge_artifacts("layers.0.mlp.weight", t, ctx)
        assert len(findings) == 0

    def test_skips_1d(self):
        t = torch.randn(64)
        findings = detect_lora_merge_artifacts("layers.0.attention.q_proj.weight", t, {})
        assert len(findings) == 0


class TestTier3DetectorCategories:
    """Verify Tier 3 conditions are mapped in the health score CATEGORY_MAP."""

    def test_token_collapse_in_output(self):
        assert "token_collapse" in CATEGORY_MAP["output"]
        assert _categorize("token_collapse") == "output"

    def test_gradient_noise_in_stability(self):
        assert "gradient_noise" in CATEGORY_MAP["stability"]
        assert _categorize("gradient_noise") == "stability"

    def test_representation_drift_in_stability(self):
        assert "representation_drift" in CATEGORY_MAP["stability"]
        assert _categorize("representation_drift") == "stability"

    def test_moe_router_collapse_in_weights(self):
        assert "moe_router_collapse" in CATEGORY_MAP["weights"]
        assert _categorize("moe_router_collapse") == "weights"

    def test_lora_merge_artifacts_in_weights(self):
        assert "lora_merge_artifacts" in CATEGORY_MAP["weights"]
        assert _categorize("lora_merge_artifacts") == "weights"


class TestTier3InDiagnose:
    """Verify Tier 3 detectors run as part of full diagnosis."""

    def test_token_collapse_found_in_diagnose(self):
        row = torch.randn(64)
        sd = {
            "lm_head.weight": row.unsqueeze(0).expand(100, -1).clone(),
        }
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "token_collapse" in conditions

    def test_representation_drift_found_in_diagnose(self):
        sd = {
            "layers.0.mlp.weight": torch.randn(64, 32) * 0.001,
            "layers.1.mlp.weight": torch.randn(64, 32) * 10.0,
        }
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "representation_drift" in conditions


# ── Quantization degradation detector tests ─────────────────────────────


class TestQuantizationDegradation:
    """Test the quantization_degradation detector."""

    def test_heavily_quantized_tensor(self):
        """Tensor with very few unique values should flag as WARN."""
        # Simulate INT4-like quantization: only 16 unique values
        t = torch.zeros(64, 64)
        values = torch.linspace(-1.0, 1.0, 16)
        flat = t.flatten()
        for i in range(flat.numel()):
            flat[i] = values[i % 16]
        t = flat.reshape(64, 64)
        findings = detect_quantization_degradation("layer.weight", t, {})
        assert len(findings) == 1
        assert findings[0].condition == "quantization_degradation"
        assert findings[0].severity == "WARN"
        assert findings[0].details["unique_ratio"] < 0.01

    def test_moderately_quantized_tensor(self):
        """Tensor with somewhat few unique values should flag as INFO."""
        # ~2% unique values
        t = torch.zeros(100, 100)
        values = torch.linspace(-1.0, 1.0, 200)  # 200 unique in 10000 = 2%
        flat = t.flatten()
        for i in range(flat.numel()):
            flat[i] = values[i % 200]
        t = flat.reshape(100, 100)
        findings = detect_quantization_degradation("layer.weight", t, {})
        assert len(findings) == 1
        assert findings[0].severity == "INFO"

    def test_grid_quantized_detection(self):
        """Uniformly spaced values should have grid_quantized=True."""
        t = torch.zeros(64, 64)
        values = torch.linspace(-1.0, 1.0, 8)  # 8 evenly spaced values
        flat = t.flatten()
        for i in range(flat.numel()):
            flat[i] = values[i % 8]
        t = flat.reshape(64, 64)
        findings = detect_quantization_degradation("layer.weight", t, {})
        assert len(findings) == 1
        assert findings[0].details.get("grid_quantized") is True

    def test_healthy_tensor_no_finding(self):
        """Normal float tensor should not flag."""
        torch.manual_seed(42)
        t = torch.randn(64, 64)
        findings = detect_quantization_degradation("layer.weight", t, {})
        assert len(findings) == 0

    def test_skips_small_tensors(self):
        """Tensors with fewer than 1000 elements should be skipped."""
        t = torch.zeros(10, 10)
        findings = detect_quantization_degradation("layer.weight", t, {})
        assert len(findings) == 0

    def test_skips_1d_tensors(self):
        """1D tensors should be skipped."""
        t = torch.zeros(2000)
        findings = detect_quantization_degradation("layer.weight", t, {})
        assert len(findings) == 0


# ── Model aging detector tests ──────────────────────────────────────────


class TestModelAging:
    """Test the model_aging post-scan detector."""

    def test_collapsed_embeddings(self):
        """Embedding with very low effective rank should flag."""
        torch.manual_seed(42)
        # Create a rank-1 embedding matrix
        a = torch.randn(256, 1)
        b = torch.randn(1, 64)
        embed = a @ b  # rank 1
        ctx = {"_aging_embed_weights": {"model.embed_tokens.weight": embed}}
        findings = post_detect_model_aging(ctx)
        aging_findings = [f for f in findings if f.details.get("reason") == "collapsed_embeddings"]
        assert len(aging_findings) == 1
        assert aging_findings[0].condition == "model_aging"
        assert aging_findings[0].severity == "WARN"
        assert aging_findings[0].details["rank_ratio"] < 0.05

    def test_healthy_embeddings_no_finding(self):
        """Full-rank embedding should not flag."""
        torch.manual_seed(42)
        embed = torch.randn(256, 64)
        ctx = {"_aging_embed_weights": {"model.embed_tokens.weight": embed}}
        findings = post_detect_model_aging(ctx)
        aging_findings = [f for f in findings if f.details.get("reason") == "collapsed_embeddings"]
        assert len(aging_findings) == 0

    def test_inverted_norm_gradient(self):
        """Early layers with much higher norms than later layers should flag."""
        ctx = {"_layer_weight_norms": {
            0: [10.0, 10.0],
            1: [9.0, 9.0],
            2: [8.0, 8.0],
            3: [7.0, 7.0],
            4: [1.0, 1.0],
            5: [1.0, 1.0],
            6: [1.0, 1.0],
            7: [1.0, 1.0],
        }}
        findings = post_detect_model_aging(ctx)
        aging_findings = [f for f in findings if f.details.get("reason") == "inverted_norm_gradient"]
        assert len(aging_findings) == 1
        assert aging_findings[0].details["ratio"] > 5.0

    def test_stable_norms_no_finding(self):
        """Layers with similar norms should not flag inverted gradient."""
        ctx = {"_layer_weight_norms": {
            0: [1.0, 1.0],
            1: [1.1, 1.1],
            2: [0.9, 0.9],
            3: [1.0, 1.0],
            4: [1.0, 1.0],
            5: [1.1, 1.1],
            6: [0.9, 0.9],
            7: [1.0, 1.0],
        }}
        findings = post_detect_model_aging(ctx)
        aging_findings = [f for f in findings if f.details.get("reason") == "inverted_norm_gradient"]
        assert len(aging_findings) == 0

    def test_token_merging_in_output(self):
        """Output projection with near-identical rows should flag."""
        row = torch.randn(64)
        output = row.unsqueeze(0).expand(100, -1).clone()
        ctx = {"_aging_output_weights": {"lm_head.weight": output}}
        findings = post_detect_model_aging(ctx)
        aging_findings = [f for f in findings if f.details.get("reason") == "token_merging"]
        assert len(aging_findings) == 1
        assert aging_findings[0].details["merged_pair_fraction"] > 0.05

    def test_distinct_output_rows_no_finding(self):
        """Distinct output rows should not flag token merging."""
        torch.manual_seed(42)
        output = torch.randn(100, 64)
        ctx = {"_aging_output_weights": {"lm_head.weight": output}}
        findings = post_detect_model_aging(ctx)
        aging_findings = [f for f in findings if f.details.get("reason") == "token_merging"]
        assert len(aging_findings) == 0

    def test_collector_populates_context(self):
        """_collect_model_aging should populate embed and output weight contexts."""
        ctx = {}
        embed = torch.randn(256, 64)
        output = torch.randn(100, 64)
        _collect_model_aging("model.embed_tokens.weight", embed, ctx)
        _collect_model_aging("lm_head.weight", output, ctx)
        assert "model.embed_tokens.weight" in ctx.get("_aging_embed_weights", {})
        assert "lm_head.weight" in ctx.get("_aging_output_weights", {})

    def test_collector_skips_1d(self):
        """1D tensors should not be collected."""
        ctx = {}
        _collect_model_aging("model.embed_tokens.weight", torch.randn(64), ctx)
        assert len(ctx.get("_aging_embed_weights", {})) == 0

    def test_empty_context(self):
        """Empty context should produce no findings."""
        findings = post_detect_model_aging({})
        assert len(findings) == 0


class TestNewDetectorCategories2:
    """Verify new conditions are mapped in the health score CATEGORY_MAP."""

    def test_quantization_degradation_in_weights(self):
        assert "quantization_degradation" in CATEGORY_MAP["weights"]
        assert _categorize("quantization_degradation") == "weights"

    def test_model_aging_in_weights(self):
        assert "model_aging" in CATEGORY_MAP["weights"]
        assert _categorize("model_aging") == "weights"


class TestNewDetectorsInDiagnose2:
    """Verify new detectors run as part of full diagnosis."""

    def test_quantization_degradation_found_in_diagnose(self):
        # Create a heavily quantized tensor
        t = torch.zeros(64, 64)
        values = torch.linspace(-1.0, 1.0, 8)
        flat = t.flatten()
        for i in range(flat.numel()):
            flat[i] = values[i % 8]
        sd = {"layer.weight": flat.reshape(64, 64)}
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "quantization_degradation" in conditions

    def test_model_aging_found_in_diagnose(self):
        torch.manual_seed(42)
        # Rank-1 embedding to trigger collapsed_embeddings
        a = torch.randn(256, 1)
        b = torch.randn(1, 64)
        sd = {
            "model.embed_tokens.weight": a @ b,
            "healthy.weight": torch.randn(64, 32),
        }
        findings = diagnose(sd)
        conditions = {f.condition for f in findings}
        assert "model_aging" in conditions
