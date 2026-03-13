"""Tests for model-clinic architecture profiles."""

import pytest
import torch

from model_clinic._profiles import (
    ArchProfile,
    get_profile,
    auto_detect_profile,
    list_profiles,
)


class TestGetProfile:
    """get_profile returns correct profiles."""

    def test_get_llm_profile(self):
        p = get_profile("llm")
        assert isinstance(p, ArchProfile)
        assert p.name == "llm"
        assert "dead_neurons" in p.detectors
        assert "q_proj" in p.key_layers

    def test_get_vit_profile(self):
        p = get_profile("vit")
        assert isinstance(p, ArchProfile)
        assert p.name == "vit"
        assert "patch_embed" in p.key_layers

    def test_get_diffusion_profile(self):
        p = get_profile("diffusion")
        assert isinstance(p, ArchProfile)
        assert p.name == "diffusion"
        assert "time_embed" in p.key_layers

    def test_unknown_profile_raises_valueerror(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent_arch")

    def test_unknown_profile_error_lists_available(self):
        with pytest.raises(ValueError, match="diffusion"):
            get_profile("cnn")


class TestListProfiles:
    """list_profiles returns all available names."""

    def test_returns_all_three(self):
        names = list_profiles()
        assert "llm" in names
        assert "vit" in names
        assert "diffusion" in names

    def test_returns_sorted(self):
        names = list_profiles()
        assert names == sorted(names)


class TestAutoDetectProfile:
    """auto_detect_profile heuristically identifies architectures."""

    def test_detects_llm_from_qwen_style_names(self):
        sd = {
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 64),
            "model.layers.0.mlp.up_proj.weight": torch.randn(256, 64),
            "model.layers.0.mlp.down_proj.weight": torch.randn(64, 256),
            "lm_head.weight": torch.randn(100, 64),
        }
        p = auto_detect_profile(sd)
        assert p is not None
        assert p.name == "llm"

    def test_detects_vit_from_patch_embed(self):
        sd = {
            "patch_embed.proj.weight": torch.randn(768, 3, 16, 16),
            "cls_token": torch.randn(1, 1, 768),
            "pos_embed": torch.randn(1, 197, 768),
            "blocks.0.attn.qkv.weight": torch.randn(2304, 768),
            "blocks.0.mlp.fc1.weight": torch.randn(3072, 768),
        }
        p = auto_detect_profile(sd)
        assert p is not None
        assert p.name == "vit"

    def test_detects_diffusion_from_unet(self):
        sd = {
            "unet.conv_in.weight": torch.randn(320, 4, 3, 3),
            "unet.time_embed.linear_1.weight": torch.randn(1280, 320),
            "unet.down_blocks.0.attentions.0.weight": torch.randn(320, 320),
        }
        p = auto_detect_profile(sd)
        assert p is not None
        assert p.name == "diffusion"

    def test_returns_none_for_unknown(self):
        sd = {
            "foo.bar.weight": torch.randn(32, 32),
            "baz.qux.bias": torch.randn(32),
        }
        p = auto_detect_profile(sd)
        assert p is None


class TestProfileDiagnose:
    """profile.diagnose() runs on state dicts."""

    def test_diagnose_runs_on_tiny_state_dict(self, tiny_state_dict):
        p = get_profile("llm")
        findings = p.diagnose(tiny_state_dict)
        assert isinstance(findings, list)
        # All findings should be Finding instances
        for f in findings:
            assert hasattr(f, "condition")
            assert hasattr(f, "severity")
            assert hasattr(f, "param_name")

    def test_diagnose_filters_to_profile_detectors(self, tiny_state_dict):
        p = get_profile("llm")
        findings = p.diagnose(tiny_state_dict)
        allowed = set(p.detectors)
        for f in findings:
            assert f.condition in allowed, (
                f"Finding {f.condition} not in LLM profile detectors"
            )

    def test_diagnose_suppresses_warnings(self):
        """LLM profile suppresses quantization_degradation."""
        p = get_profile("llm")
        assert "quantization_degradation" in p.warnings
        # Even if a quantization finding would normally fire, it gets filtered
        # We verify this by checking the suppression list is applied
        findings = p.diagnose({
            "layers.0.attention.q_proj.weight": torch.randn(64, 64),
        })
        quant_findings = [f for f in findings if f.condition == "quantization_degradation"]
        assert len(quant_findings) == 0

    def test_diagnose_catches_nan(self):
        """Profile should still find NaN values."""
        p = get_profile("llm")
        bad = torch.randn(32, 32)
        bad[0, 0] = float("nan")
        findings = p.diagnose({"layers.0.attention.q_proj.weight": bad})
        nan_findings = [f for f in findings if f.condition == "nan_inf"]
        assert len(nan_findings) >= 1

    def test_vit_profile_diagnose_runs(self):
        sd = {
            "patch_embed.proj.weight": torch.randn(64, 3, 4, 4),
            "cls_token": torch.randn(1, 1, 64),
            "blocks.0.attn.qkv.weight": torch.randn(192, 64),
            "blocks.0.norm1.weight": torch.ones(64),
        }
        p = get_profile("vit")
        findings = p.diagnose(sd)
        assert isinstance(findings, list)

    def test_diffusion_profile_diagnose_runs(self):
        sd = {
            "unet.conv_in.weight": torch.randn(32, 4, 3, 3),
            "unet.time_embed.linear_1.weight": torch.randn(128, 32),
            "unet.norm.weight": torch.ones(32),
        }
        p = get_profile("diffusion")
        findings = p.diagnose(sd)
        assert isinstance(findings, list)


class TestProfileDescribe:
    """profile.describe() returns readable text."""

    def test_describe_contains_name(self):
        p = get_profile("llm")
        desc = p.describe()
        assert "llm" in desc
        assert "Architecture Profile" in desc

    def test_describe_lists_detectors(self):
        p = get_profile("vit")
        desc = p.describe()
        assert "dead_neurons" in desc

    def test_describe_lists_key_layers(self):
        p = get_profile("diffusion")
        desc = p.describe()
        assert "time_embed" in desc


class TestProfileBaselines:
    """profile.healthy_baselines() returns metric ranges."""

    def test_llm_baselines_have_norm_range(self):
        b = get_profile("llm").healthy_baselines()
        assert "per_element_norm" in b
        assert b["per_element_norm"]["min"] == 0.01
        assert b["per_element_norm"]["max"] == 5.0

    def test_llm_baselines_have_kurtosis(self):
        b = get_profile("llm").healthy_baselines()
        assert "kurtosis" in b
        assert b["kurtosis"]["max"] == 50

    def test_vit_baselines_differ_from_llm(self):
        llm_b = get_profile("llm").healthy_baselines()
        vit_b = get_profile("vit").healthy_baselines()
        # ViT allows larger norms
        assert vit_b["per_element_norm"]["max"] > llm_b["per_element_norm"]["max"]

    def test_baselines_returns_dict_copy(self):
        p = get_profile("llm")
        b1 = p.healthy_baselines()
        b2 = p.healthy_baselines()
        assert b1 == b2
        b1["new_key"] = "test"
        assert "new_key" not in p.healthy_baselines()
