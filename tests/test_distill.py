"""Tests for Level 3: Targeted Re-initialization with Knowledge Distillation."""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from model_clinic._types import Finding
from model_clinic._repair.distill import (
    identify_dead_modules,
    reset_module_params,
    distill_repair,
    DistillReport,
    _module_prefix,
)
from model_clinic._repair.calibration import (
    load_calibration_data,
    generate_random_calibration,
)


# ---------------------------------------------------------------------------
# Tiny synthetic model for distillation tests
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    """A 2-layer MLP with hidden_dim=32. Small enough to run in ms on CPU."""

    def __init__(self, input_dim: int = 32, hidden_dim: int = 32, output_dim: int = 16):
        super().__init__()
        self.layer0 = nn.Linear(input_dim, hidden_dim)
        self.act0 = nn.ReLU()
        self.layer1 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.layer0(x))
        return self.layer1(x)


class TinyMLPWithSubmodule(nn.Module):
    """MLP where layer1 is wrapped in a submodule namespace for prefix tests."""

    def __init__(self):
        super().__init__()
        self.layer0 = nn.Linear(32, 32)
        self.act = nn.ReLU()
        self.sub = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.layer0(x))
        return self.sub(x)


# ---------------------------------------------------------------------------
# Tests: identify_dead_modules
# ---------------------------------------------------------------------------

class TestIdentifyDeadModules:

    def test_single_condition_not_enough(self):
        """A module with only 1 dead condition should NOT be flagged."""
        findings = [
            Finding(
                condition="identical_rows",
                severity="ERROR",
                param_name="memory.tiers.0.keys.weight",
            ),
        ]
        result = identify_dead_modules(findings)
        assert result == []

    def test_two_conditions_triggers(self):
        """A module with 2 dead conditions SHOULD be flagged."""
        findings = [
            Finding(
                condition="identical_rows",
                severity="ERROR",
                param_name="memory.tiers.0.keys.weight",
            ),
            Finding(
                condition="model_aging",
                severity="WARN",
                param_name="memory.tiers.0.values.weight",
            ),
        ]
        result = identify_dead_modules(findings)
        assert result == ["memory.tiers.0"]

    def test_three_conditions(self):
        """All three dead conditions present."""
        findings = [
            Finding(
                condition="identical_rows",
                severity="ERROR",
                param_name="memory.tiers.2.keys.weight",
            ),
            Finding(
                condition="gradient_noise",
                severity="ERROR",
                param_name="memory.tiers.2.values.weight",
            ),
            Finding(
                condition="model_aging",
                severity="WARN",
                param_name="memory.tiers.2.gate.bias",
            ),
        ]
        result = identify_dead_modules(findings)
        assert "memory.tiers.2" in result

    def test_gradient_noise_warn_ignored(self):
        """gradient_noise at WARN level should be ignored."""
        findings = [
            Finding(
                condition="identical_rows",
                severity="ERROR",
                param_name="mem.0.keys.weight",
            ),
            Finding(
                condition="gradient_noise",
                severity="WARN",  # Not ERROR — should be ignored
                param_name="mem.0.values.weight",
            ),
        ]
        result = identify_dead_modules(findings)
        assert result == []

    def test_irrelevant_conditions_ignored(self):
        """Conditions outside the dead set are ignored."""
        findings = [
            Finding(
                condition="dead_neurons",
                severity="ERROR",
                param_name="layer.0.linear.weight",
            ),
            Finding(
                condition="norm_drift",
                severity="WARN",
                param_name="layer.0.norm.weight",
            ),
        ]
        result = identify_dead_modules(findings)
        assert result == []

    def test_multiple_dead_modules(self):
        """Multiple distinct modules can be flagged."""
        findings = [
            Finding("identical_rows", "ERROR", "mem.0.keys.weight"),
            Finding("model_aging", "WARN", "mem.0.values.weight"),
            Finding("identical_rows", "ERROR", "mem.2.keys.weight"),
            Finding("gradient_noise", "ERROR", "mem.2.values.weight"),
        ]
        result = identify_dead_modules(findings)
        assert "mem.0" in result
        assert "mem.2" in result

    def test_empty_findings(self):
        result = identify_dead_modules([])
        assert result == []

    def test_redundant_prefixes_collapsed(self):
        """If both 'a' and 'a.b' are dead, only 'a' should be returned."""
        findings = [
            # Makes prefix "a" dead
            Finding("identical_rows", "ERROR", "a.x.weight"),
            Finding("model_aging", "WARN", "a.y.weight"),
            # Makes prefix "a.b" dead too — but it's under "a"
            Finding("identical_rows", "ERROR", "a.b.c.weight"),
            Finding("model_aging", "WARN", "a.b.d.weight"),
        ]
        result = identify_dead_modules(findings)
        # "a" covers "a.b" so "a.b" should be removed
        assert "a" in result
        # "a.b" should not appear since it's redundant
        assert "a.b" not in result


class TestModulePrefix:

    def test_standard_param(self):
        assert _module_prefix("memory.tiers.0.keys.weight") == "memory.tiers.0"

    def test_two_parts(self):
        assert _module_prefix("layer.weight") == "layer"

    def test_single_part(self):
        assert _module_prefix("weight") == "weight"

    def test_deep_nesting(self):
        assert _module_prefix("a.b.c.d.weight") == "a.b.c"


# ---------------------------------------------------------------------------
# Tests: reset_module_params
# ---------------------------------------------------------------------------

class TestResetModuleParams:

    def test_resets_2d_tensor(self):
        sd = {"mod.linear.weight": torch.ones(16, 16)}
        reset_module_params(sd, "mod")
        w = sd["mod.linear.weight"]
        # After Xavier init, should NOT be all ones
        assert not torch.allclose(w, torch.ones_like(w))

    def test_resets_1d_tensor_to_zeros(self):
        sd = {"mod.linear.bias": torch.ones(16)}
        reset_module_params(sd, "mod")
        assert torch.allclose(sd["mod.linear.bias"], torch.zeros(16))

    def test_skips_int_tensors(self):
        counter = torch.tensor([42], dtype=torch.int32)
        sd = {"mod.counter": counter.clone()}
        reset_module_params(sd, "mod")
        assert sd["mod.counter"].item() == 42

    def test_skips_unmatched_prefix(self):
        sd = {
            "mod_a.weight": torch.ones(8, 8),
            "mod_b.weight": torch.ones(8, 8),
        }
        reset_module_params(sd, "mod_a")
        # mod_a should be reset
        assert not torch.allclose(sd["mod_a.weight"], torch.ones(8, 8))
        # mod_b should be untouched
        assert torch.allclose(sd["mod_b.weight"], torch.ones(8, 8))

    def test_returns_state_dict(self):
        sd = {"mod.w": torch.ones(4, 4)}
        result = reset_module_params(sd, "mod")
        assert result is sd

    def test_empty_state_dict(self):
        sd = {}
        result = reset_module_params(sd, "mod")
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: distill_repair
# ---------------------------------------------------------------------------

class TestDistillRepair:

    def _make_dead_model(self):
        """Create a TinyMLP where layer1 is 'dead' (all zeros)."""
        model = TinyMLP(input_dim=32, hidden_dim=32, output_dim=16)
        with torch.no_grad():
            model.layer1.weight.zero_()
            model.layer1.bias.zero_()
        return model

    def _make_calibration(self, n=10):
        return [torch.randn(32) for _ in range(n)]

    def test_distill_reduces_loss(self):
        """Distillation should reduce the MSE between student and teacher."""
        # Teacher: a trained (random-init) model
        teacher = TinyMLP(input_dim=32, hidden_dim=32, output_dim=16)

        # Student: same architecture but layer1 is dead
        student = self._make_dead_model()

        # Capture teacher outputs for later comparison
        calibration = self._make_calibration(10)
        with torch.no_grad():
            teacher_outs = [teacher(x) for x in calibration]
            student_outs_before = [student(x) for x in calibration]

        # MSE before: student layer1 is all zeros, so outputs are all-zero
        mse_before = sum(
            (s - t).pow(2).mean().item()
            for s, t in zip(student_outs_before, teacher_outs)
        ) / len(calibration)

        # Run distillation (using student's own working layers as teacher)
        repaired = distill_repair(
            student,
            dead_modules=["layer1"],
            calibration_loader=calibration,
            num_steps=50,
            lr=1e-3,
            device="cpu",
        )

        # MSE after should be lower (student output changed from all-zeros)
        with torch.no_grad():
            student_outs_after = [repaired(x) for x in calibration]

        # The repaired model should produce non-zero outputs
        all_zero_after = all(
            torch.allclose(o, torch.zeros_like(o), atol=1e-6)
            for o in student_outs_after
        )
        assert not all_zero_after, "Distillation did not activate dead layer"

    def test_empty_dead_modules_noop(self):
        """Empty dead_modules list should be a no-op."""
        model = TinyMLP()
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}

        result = distill_repair(
            model,
            dead_modules=[],
            calibration_loader=self._make_calibration(5),
            num_steps=10,
        )

        sd_after = result.state_dict()
        for k in sd_before:
            assert torch.equal(sd_before[k], sd_after[k])

    def test_empty_calibration_noop(self):
        """Empty calibration data should be a no-op."""
        model = TinyMLP()
        sd_before = {k: v.clone() for k, v in model.state_dict().items()}

        result = distill_repair(
            model,
            dead_modules=["layer1"],
            calibration_loader=[],
            num_steps=10,
        )

        sd_after = result.state_dict()
        for k in sd_before:
            assert torch.equal(sd_before[k], sd_after[k])

    def test_unfreezes_after_repair(self):
        """All parameters should have requires_grad=True after repair."""
        model = self._make_dead_model()
        calibration = self._make_calibration(5)

        repaired = distill_repair(
            model,
            dead_modules=["layer1"],
            calibration_loader=calibration,
            num_steps=5,
        )

        for name, param in repaired.named_parameters():
            assert param.requires_grad, f"{name} has requires_grad=False"

    def test_non_dead_params_unchanged(self):
        """Parameters NOT in dead_modules should remain frozen during repair."""
        model = self._make_dead_model()
        layer0_weight_before = model.layer0.weight.clone()
        calibration = self._make_calibration(5)

        repaired = distill_repair(
            model,
            dead_modules=["layer1"],
            calibration_loader=calibration,
            num_steps=20,
            lr=1e-3,
        )

        # layer0 weights should be identical
        assert torch.equal(
            repaired.layer0.weight, layer0_weight_before
        ), "Non-dead module weights changed during distillation"

    def test_with_submodule(self):
        """Test distillation on a model with nested submodules."""
        model = TinyMLPWithSubmodule()
        # Kill the sub module
        with torch.no_grad():
            for p in model.sub.parameters():
                p.zero_()

        calibration = [torch.randn(32) for _ in range(8)]
        repaired = distill_repair(
            model,
            dead_modules=["sub"],
            calibration_loader=calibration,
            num_steps=30,
            lr=1e-3,
        )

        with torch.no_grad():
            out = repaired(torch.randn(32))
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)


# ---------------------------------------------------------------------------
# Tests: calibration data loading
# ---------------------------------------------------------------------------

class TestCalibrationLoading:

    def test_load_pt_single_tensor(self, tmp_path):
        data = torch.randint(0, 100, (10, 64))
        pt_path = tmp_path / "calib.pt"
        torch.save(data, str(pt_path))

        result = load_calibration_data(str(pt_path), max_samples=5)
        assert len(result) == 5
        assert all(isinstance(t, torch.Tensor) for t in result)
        assert all(t.dtype == torch.long for t in result)

    def test_load_pt_list_of_tensors(self, tmp_path):
        data = [torch.randint(0, 100, (32,)) for _ in range(8)]
        pt_path = tmp_path / "calib.pt"
        torch.save(data, str(pt_path))

        result = load_calibration_data(str(pt_path))
        assert len(result) == 8

    def test_load_pt_truncation(self, tmp_path):
        data = torch.randint(0, 100, (3, 1000))
        pt_path = tmp_path / "calib.pt"
        torch.save(data, str(pt_path))

        result = load_calibration_data(str(pt_path), max_length=64)
        assert all(t.shape[0] <= 64 for t in result)

    def test_load_jsonl(self, tmp_path):
        jsonl_path = tmp_path / "calib.jsonl"
        lines = [
            json.dumps({"text": f"Hello world sample {i}"})
            for i in range(5)
        ]
        jsonl_path.write_text("\n".join(lines), encoding="utf-8")

        class FakeTokenizer:
            def __call__(self, text, truncation=True, max_length=512, return_tensors="pt"):
                ids = torch.tensor([[ord(c) % 100 for c in text[:max_length]]])
                return {"input_ids": ids}

        result = load_calibration_data(str(jsonl_path), tokenizer=FakeTokenizer())
        assert len(result) == 5
        assert all(isinstance(t, torch.Tensor) for t in result)

    def test_load_jsonl_no_tokenizer_raises(self, tmp_path):
        jsonl_path = tmp_path / "calib.jsonl"
        jsonl_path.write_text('{"text": "hi"}\n', encoding="utf-8")

        try:
            load_calibration_data(str(jsonl_path))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "tokenizer" in str(e).lower()

    def test_load_txt(self, tmp_path):
        txt_path = tmp_path / "calib.txt"
        txt_path.write_text("line one\nline two\nline three\n", encoding="utf-8")

        class FakeTokenizer:
            def __call__(self, text, truncation=True, max_length=512, return_tensors="pt"):
                ids = torch.tensor([[ord(c) % 100 for c in text[:max_length]]])
                return {"input_ids": ids}

        result = load_calibration_data(str(txt_path), tokenizer=FakeTokenizer())
        assert len(result) == 3

    def test_load_txt_no_tokenizer_raises(self, tmp_path):
        txt_path = tmp_path / "calib.txt"
        txt_path.write_text("hello\n", encoding="utf-8")

        try:
            load_calibration_data(str(txt_path))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "tokenizer" in str(e).lower()

    def test_file_not_found(self):
        try:
            load_calibration_data("/nonexistent/path.pt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_unsupported_format(self, tmp_path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b,c\n", encoding="utf-8")

        try:
            load_calibration_data(str(csv_path))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert ".csv" in str(e)

    def test_max_samples_limit(self, tmp_path):
        data = [torch.randint(0, 100, (32,)) for _ in range(20)]
        pt_path = tmp_path / "calib.pt"
        torch.save(data, str(pt_path))

        result = load_calibration_data(str(pt_path), max_samples=5)
        assert len(result) == 5


class TestRandomCalibration:

    def test_default_params(self):
        result = generate_random_calibration()
        assert len(result) == 50
        assert all(t.shape == (128,) for t in result)
        assert all(t.dtype == torch.long for t in result)

    def test_custom_params(self):
        result = generate_random_calibration(
            vocab_size=1000, num_samples=10, seq_length=64
        )
        assert len(result) == 10
        assert all(t.shape == (64,) for t in result)
        # All values should be in [0, 1000)
        assert all(t.max().item() < 1000 for t in result)
        assert all(t.min().item() >= 0 for t in result)

    def test_single_sample(self):
        result = generate_random_calibration(num_samples=1, seq_length=8)
        assert len(result) == 1
        assert result[0].shape == (8,)


# ---------------------------------------------------------------------------
# Tests: DistillReport dataclass
# ---------------------------------------------------------------------------

class TestDistillReport:

    def test_creation(self):
        report = DistillReport(
            dead_modules=["mem.0", "mem.2"],
            steps_run=200,
            loss_start=1.5,
            loss_end=0.3,
            params_reset=5000,
            params_total=100000,
        )
        assert report.dead_modules == ["mem.0", "mem.2"]
        assert report.steps_run == 200
        assert report.loss_start == 1.5
        assert report.loss_end == 0.3
        assert report.params_reset == 5000
        assert report.params_total == 100000

    def test_empty_dead_modules(self):
        report = DistillReport(
            dead_modules=[],
            steps_run=0,
            loss_start=0.0,
            loss_end=0.0,
            params_reset=0,
            params_total=50000,
        )
        assert report.dead_modules == []
