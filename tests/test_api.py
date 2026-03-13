"""Tests for the public API — imports, types, and round-trip workflows."""

import torch


class TestPublicAPI:
    """Verify all documented symbols are importable."""

    def test_version(self):
        import model_clinic
        assert model_clinic.__version__ == "0.3.0"

    def test_types_importable(self):
        from model_clinic import Finding, Prescription, TreatmentResult, ExamReport, ModelMeta
        assert Finding is not None
        assert Prescription is not None
        assert TreatmentResult is not None
        assert ExamReport is not None
        assert ModelMeta is not None

    def test_clinic_importable(self):
        from model_clinic import diagnose, prescribe, apply_treatment, rollback_treatment
        assert callable(diagnose)
        assert callable(prescribe)
        assert callable(apply_treatment)
        assert callable(rollback_treatment)

    def test_loader_importable(self):
        from model_clinic import load_state_dict, load_model, build_meta, save_state_dict
        assert callable(load_state_dict)
        assert callable(load_model)
        assert callable(build_meta)
        assert callable(save_state_dict)

    def test_eval_importable(self):
        from model_clinic import generate, eval_coherence, eval_perplexity, eval_logit_entropy, eval_diversity
        assert callable(generate)
        assert callable(eval_coherence)
        assert callable(eval_perplexity)
        assert callable(eval_logit_entropy)
        assert callable(eval_diversity)

    def test_all_matches_exports(self):
        import model_clinic
        for name in model_clinic.__all__:
            assert hasattr(model_clinic, name), f"{name} in __all__ but not exported"


class TestExamReport:
    """Test ExamReport serialization."""

    def test_to_dict(self):
        from model_clinic import Finding, Prescription, TreatmentResult, ExamReport, ModelMeta

        meta = ModelMeta(source="test", num_params=1000, num_tensors=5,
                         hidden_size=64, num_layers=2, vocab_size=100)
        f = Finding("dead_neurons", "WARN", "layer.0.weight",
                    {"dead_count": 3, "pct": 0.05})
        rx = Prescription("reinit_dead", "Fix dead", "low", f, "reinit_dead",
                          {"indices": [0, 1, 2]})
        tr = TreatmentResult(rx, True, "Fixed 3 dead neurons")

        report = ExamReport(
            model_path="test.pt",
            meta=meta,
            findings=[f],
            prescriptions=[rx],
            treatments=[tr],
            before_score=3.0,
            after_score=5.0,
        )

        d = report.to_dict()
        assert d["model"] == "test.pt"
        assert d["meta"]["num_params"] == 1000
        assert len(d["findings"]) == 1
        assert d["findings"][0]["condition"] == "dead_neurons"
        assert len(d["prescriptions"]) == 1
        assert d["prescriptions"][0]["action"] == "reinit_dead"
        assert len(d["treatments"]) == 1
        assert d["treatments"][0]["success"] is True
        assert d["before_score"] == 3.0
        assert d["after_score"] == 5.0

    def test_to_dict_empty(self):
        from model_clinic import ExamReport, ModelMeta
        report = ExamReport(model_path="empty.pt", meta=ModelMeta())
        d = report.to_dict()
        assert d["model"] == "empty.pt"
        assert len(d["findings"]) == 0
        assert len(d["prescriptions"]) == 0


class TestEndToEnd:
    """Full pipeline round-trip tests."""

    def test_diagnose_prescribe_treat(self):
        from model_clinic import diagnose, prescribe, apply_treatment

        sd = {
            "layer.0.weight": torch.randn(64, 64),
            "bad_norm.weight": torch.full((64,), 0.2),  # norm drift
            "bad_param": torch.randn(32, 32),
        }
        sd["bad_param"][5, 5] = float("nan")

        # Diagnose
        findings = diagnose(sd)
        assert len(findings) > 0

        # Prescribe
        rxs = prescribe(findings)
        assert len(rxs) > 0

        # Treat
        for rx in rxs:
            result = apply_treatment(sd, rx)
            assert result.success or "not found" in result.description.lower() or "Skipped" in result.description

        # Verify NaN fixed
        assert not torch.isnan(sd["bad_param"]).any()

    def test_diagnose_healthy_returns_minimal(self):
        from model_clinic import diagnose

        sd = {
            "layer.0.weight": torch.randn(64, 64),
            "layer.0.bias": torch.randn(64),
            "norm.weight": torch.ones(64),
        }
        findings = diagnose(sd)
        errors = [f for f in findings if f.severity == "ERROR"]
        assert len(errors) == 0
