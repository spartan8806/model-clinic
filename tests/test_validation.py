"""Tests for treatment validation reporting (model_clinic._validation)."""

import io
import os
import tempfile

import torch

from model_clinic import ValidationReport, HealthScore
from model_clinic._validation import print_validation_report
from model_clinic._synthetic import make_everything_broken
from model_clinic import clinic


# ── derived-metric tests ──────────────────────────────────────────────────

def test_health_delta():
    r = ValidationReport(health_before=HealthScore(62, grade="D"),
                         health_after=HealthScore(78, grade="C"))
    assert r.health_delta == 16


def test_health_delta_none_when_missing():
    assert ValidationReport(health_before=HealthScore(50)).health_delta is None
    assert ValidationReport().health_delta is None


def test_ppl_factor():
    r = ValidationReport(ppl_before=100.0, ppl_after=25.0)
    assert r.ppl_factor == 4.0


def test_ppl_factor_guards_zero():
    assert ValidationReport(ppl_before=100.0, ppl_after=0.0).ppl_factor is None
    assert ValidationReport(ppl_before=100.0).ppl_factor is None


def test_coherence_ratios():
    r = ValidationReport(coherence_before=(2, 5), coherence_after=(4, 5))
    assert r.coherence_ratio_before == 0.4
    assert r.coherence_ratio_after == 0.8


def test_coherence_ratio_guards_zero_total():
    assert ValidationReport(coherence_before=(0, 0)).coherence_ratio_before is None


# ── verdict tests ─────────────────────────────────────────────────────────

def test_verdict_improved_by_health():
    r = ValidationReport(health_before=HealthScore(60), health_after=HealthScore(75))
    assert r.verdict() == "IMPROVED"


def test_verdict_regressed_by_health():
    r = ValidationReport(health_before=HealthScore(75), health_after=HealthScore(60))
    assert r.verdict() == "REGRESSED"


def test_verdict_neutral():
    r = ValidationReport(health_before=HealthScore(70), health_after=HealthScore(70))
    assert r.verdict() == "NEUTRAL"


def test_verdict_inconclusive_without_health():
    assert ValidationReport().verdict() == "INCONCLUSIVE"


def test_verdict_rolled_back_takes_precedence():
    r = ValidationReport(health_before=HealthScore(60), health_after=HealthScore(80),
                         rolled_back=True, rollback_reason="x")
    assert r.verdict() == "ROLLED BACK"


def test_runtime_signal_outranks_health():
    # Health improved but PPL regressed -> runtime evidence wins -> REGRESSED
    r = ValidationReport(health_before=HealthScore(60), health_after=HealthScore(80),
                         ppl_before=50.0, ppl_after=90.0)
    assert r.verdict() == "REGRESSED"


def test_runtime_improvement():
    r = ValidationReport(health_before=HealthScore(70), health_after=HealthScore(70),
                         coherence_before=(1, 5), coherence_after=(5, 5))
    assert r.verdict() == "IMPROVED"


# ── serialization ─────────────────────────────────────────────────────────

def test_to_dict_roundtrip():
    r = ValidationReport(n_applied=2, n_total=4,
                         health_before=HealthScore(62, grade="D"),
                         health_after=HealthScore(78, grade="C"),
                         ppl_before=100.0, ppl_after=25.0,
                         coherence_before=(2, 5), coherence_after=(4, 5))
    d = r.to_dict()
    assert d["applied"] == 2
    assert d["health_delta"] == 16
    assert d["ppl_factor"] == 4.0
    assert d["verdict"] == "IMPROVED"
    assert d["coherence_after"] == 0.8


# ── rendering (must not raise on any console) ──────────────────────────────

def test_print_does_not_crash():
    buf = io.StringIO()
    r = ValidationReport(n_applied=2, n_total=4,
                         health_before=HealthScore(62, grade="D"),
                         health_after=HealthScore(78, grade="C"),
                         ppl_before=11726.0, ppl_after=46.6,
                         coherence_before=(2, 5), coherence_after=(4, 5))
    print_validation_report(r, file=buf)
    text = buf.getvalue()
    assert "TREATMENT VALIDATION" in text
    assert "62/D" in text and "78/C" in text
    assert "VERDICT: IMPROVED" in text


def test_print_health_only():
    """No runtime metrics — health-only report still renders."""
    buf = io.StringIO()
    r = ValidationReport(n_applied=3, n_total=3,
                         health_before=HealthScore(34, grade="F"),
                         health_after=HealthScore(59, grade="D"))
    print_validation_report(r, file=buf)
    text = buf.getvalue()
    assert "34/F" in text and "59/D" in text
    assert "PPL" not in text  # PPL row omitted when not measured


# ── integration: treat raises health on a broken model, statically ────────

def test_treat_improves_health_static(capsys):
    sd = make_everything_broken()
    tmp = os.path.join(tempfile.gettempdir(), "mc_val_test.pt")
    torch.save(sd, tmp)
    try:
        class Args:
            model = tmp; hf = False; runtime = False; quiet = False; verbose = False
            only = None; conservative = False; dry_run = False; test = False
            no_rollback = False; save = None; manifest = None; export = None
            example_prompts = False
        clinic.run_treat(Args())
        out = capsys.readouterr().out
        assert "TREATMENT VALIDATION" in out
        assert "VERDICT: IMPROVED" in out
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
