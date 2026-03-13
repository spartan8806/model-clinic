"""Tests for the treatment manifest module."""

import json
import tempfile
from io import StringIO
from pathlib import Path

import torch

from model_clinic._types import Finding, Prescription, TreatmentResult
from model_clinic._manifest import TreatmentManifest, _tensor_checksum


def _make_treatment(name="reset_norm", action="reset_norm_weights",
                    target="final_norm.weight", risk="low",
                    success=True, description="Fixed norm"):
    """Helper to create a TreatmentResult."""
    finding = Finding(condition="norm_drift", severity="WARN", param_name=target)
    rx = Prescription(
        name=name,
        description=description,
        risk=risk,
        finding=finding,
        action=action,
    )
    backup = torch.ones(64) * 2.5  # original tensor
    return TreatmentResult(prescription=rx, success=success,
                           description=description, backup=backup)


class TestTensorChecksum:

    def test_deterministic(self):
        t = torch.randn(32, 32)
        assert _tensor_checksum(t) == _tensor_checksum(t)

    def test_length_is_8(self):
        t = torch.randn(10)
        assert len(_tensor_checksum(t)) == 8

    def test_different_tensors_different_checksums(self):
        a = torch.randn(32)
        b = torch.randn(32)
        assert _tensor_checksum(a) != _tensor_checksum(b)

    def test_hex_chars_only(self):
        t = torch.randn(10)
        cs = _tensor_checksum(t)
        assert all(c in "0123456789abcdef" for c in cs)


class TestTreatmentManifest:

    def test_empty_manifest(self):
        m = TreatmentManifest()
        d = m.to_dict()
        assert d["treatments"] == []
        assert d["summary"]["total_applied"] == 0
        assert d["summary"]["total_failed"] == 0

    def test_record_successful_treatment(self):
        m = TreatmentManifest()
        result = _make_treatment()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(result, sd)

        d = m.to_dict()
        assert len(d["treatments"]) == 1
        assert d["treatments"][0]["success"] is True
        assert d["treatments"][0]["prescription"] == "reset_norm"
        assert d["treatments"][0]["target"] == "final_norm.weight"
        assert d["treatments"][0]["risk"] == "low"
        assert d["summary"]["total_applied"] == 1
        assert d["summary"]["total_failed"] == 0

    def test_record_failed_treatment(self):
        m = TreatmentManifest()
        result = _make_treatment(success=False)
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(result, sd)

        d = m.to_dict()
        assert d["summary"]["total_applied"] == 0
        assert d["summary"]["total_failed"] == 1

    def test_checksums_recorded(self):
        m = TreatmentManifest()
        result = _make_treatment()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(result, sd)

        entry = m.to_dict()["treatments"][0]
        assert entry["checksum_before"] is not None
        assert entry["checksum_after"] is not None
        assert len(entry["checksum_before"]) == 8
        assert len(entry["checksum_after"]) == 8
        # Before and after should differ (backup vs current)
        assert entry["checksum_before"] != entry["checksum_after"]

    def test_multiple_treatments(self):
        m = TreatmentManifest()
        sd = {
            "final_norm.weight": torch.ones(64),
            "wrapper/gate": torch.tensor(0.0),
        }
        m.record(_make_treatment(name="fix_norm", risk="low"), sd)
        m.record(_make_treatment(name="fix_gate", risk="medium",
                                  target="wrapper/gate"), sd)

        d = m.to_dict()
        assert len(d["treatments"]) == 2
        assert d["summary"]["total_applied"] == 2
        assert d["summary"]["risk_breakdown"] == {"low": 1, "medium": 1}

    def test_risk_breakdown(self):
        m = TreatmentManifest()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(_make_treatment(risk="low"), sd)
        m.record(_make_treatment(risk="low"), sd)
        m.record(_make_treatment(risk="high"), sd)

        d = m.to_dict()
        assert d["summary"]["risk_breakdown"]["low"] == 2
        assert d["summary"]["risk_breakdown"]["high"] == 1

    def test_version_in_manifest(self):
        from model_clinic import __version__
        m = TreatmentManifest()
        d = m.to_dict()
        assert d["model_clinic_version"] == __version__

    def test_timestamp_in_manifest(self):
        m = TreatmentManifest()
        d = m.to_dict()
        assert "timestamp" in d
        assert "T" in d["timestamp"]  # ISO format

    def test_save_and_load(self, tmp_path):
        m = TreatmentManifest()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(_make_treatment(), sd)

        path = tmp_path / "manifest.json"
        m.save(str(path))

        with open(path) as f:
            loaded = json.load(f)

        assert loaded["summary"]["total_applied"] == 1
        assert len(loaded["treatments"]) == 1
        assert loaded["treatments"][0]["prescription"] == "reset_norm"

    def test_json_serializable(self):
        m = TreatmentManifest()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(_make_treatment(), sd)
        # Should not raise
        output = json.dumps(m.to_dict(), default=str)
        assert "reset_norm" in output


class TestPrintSummary:

    def test_prints_without_error(self):
        m = TreatmentManifest()
        buf = StringIO()
        m.print_summary(file=buf)
        output = buf.getvalue()
        assert "Treatment Manifest" in output

    def test_prints_treatment_details(self):
        m = TreatmentManifest()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(_make_treatment(), sd)

        buf = StringIO()
        m.print_summary(file=buf)
        output = buf.getvalue()
        assert "reset_norm" in output
        assert "OK" in output

    def test_prints_failed(self):
        m = TreatmentManifest()
        sd = {"final_norm.weight": torch.ones(64)}
        m.record(_make_treatment(success=False), sd)

        buf = StringIO()
        m.print_summary(file=buf)
        output = buf.getvalue()
        assert "FAIL" in output
        assert "Failed:   1" in output
