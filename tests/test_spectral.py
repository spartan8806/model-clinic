"""Tests for spectral surgery — SVD-based denoising and analysis."""

import torch
from model_clinic._types import Finding, Prescription
from model_clinic._repair.spectral import (
    spectral_denoise,
    spectral_analysis,
    spectral_denoise_with_report,
    SpectralReport,
)
from model_clinic.clinic import prescribe, apply_treatment, diagnose


class TestSpectralAnalysis:
    """Test spectral_analysis returns correct metrics."""

    def test_analysis_basic(self):
        t = torch.randn(64, 32)
        info = spectral_analysis(t)
        assert "singular_values" in info
        assert "effective_rank" in info
        assert "condition_number" in info
        assert "energy_distribution" in info
        assert "total_energy" in info
        assert "shape" in info
        assert info["shape"] == [64, 32]
        assert info["effective_rank"] > 0
        assert info["condition_number"] > 0
        assert info["total_energy"] > 0

    def test_analysis_1d_skip(self):
        t = torch.randn(64)
        info = spectral_analysis(t)
        assert info["effective_rank"] == 0
        assert info["condition_number"] == 1.0

    def test_analysis_identity(self):
        t = torch.eye(32)
        info = spectral_analysis(t)
        # Identity has condition number 1
        assert abs(info["condition_number"] - 1.0) < 1e-4
        assert info["effective_rank"] == 32

    def test_analysis_rank_deficient(self):
        # Create a rank-1 matrix
        a = torch.randn(64, 1)
        b = torch.randn(1, 32)
        t = a @ b
        info = spectral_analysis(t)
        assert info["effective_rank"] == 1

    def test_analysis_energy_distribution(self):
        t = torch.randn(32, 32)
        info = spectral_analysis(t)
        ed = info["energy_distribution"]
        # Should be monotonically increasing and end at 1.0
        assert ed[-1].item() > 0.999
        for i in range(1, len(ed)):
            assert ed[i] >= ed[i - 1] - 1e-6

    def test_analysis_zero_tensor(self):
        t = torch.zeros(32, 32)
        info = spectral_analysis(t)
        assert info["effective_rank"] == 0


class TestSpectralDenoise:
    """Test spectral_denoise on matrices with known properties."""

    def test_high_condition_drops(self):
        """Matrix with high condition number should have it reduced."""
        # Build a matrix with known high condition: S = [1000, 1, 0.001]
        U = torch.linalg.qr(torch.randn(64, 64))[0][:, :32]
        V = torch.linalg.qr(torch.randn(32, 32))[0]
        S = torch.zeros(32)
        S[0] = 1000.0
        S[1] = 100.0
        S[2] = 10.0
        for i in range(3, 32):
            S[i] = 0.001  # very small — noise
        t = U @ torch.diag(S) @ V

        # Verify it has high condition number before
        sv_before = torch.linalg.svdvals(t)
        sv_pos = sv_before[sv_before > 1e-10]
        cond_before = (sv_pos[0] / sv_pos[-1]).item()
        assert cond_before > 100_000

        denoised = spectral_denoise(t, max_condition=1000)

        # Check condition number dropped. Use spectral_analysis for a clean
        # measurement — numerical SVD of a truncated reconstruction has
        # float32 noise SVs that aren't meaningful signal.
        info = spectral_analysis(denoised)
        # The effective rank should be much smaller than original
        assert info["effective_rank"] <= 5
        # Verify the noise SVs were removed: the denoised matrix should
        # differ from the original
        assert not torch.equal(t, denoised)
        # Relative reconstruction error should be small (we kept 99%+ energy)
        error = (t - denoised).norm() / t.norm()
        assert error < 0.05

    def test_energy_preserved(self):
        """Energy threshold should preserve most of the signal."""
        U = torch.linalg.qr(torch.randn(64, 64))[0][:, :32]
        V = torch.linalg.qr(torch.randn(32, 32))[0]
        S = torch.zeros(32)
        S[0] = 100.0
        S[1] = 50.0
        S[2] = 10.0
        for i in range(3, 32):
            S[i] = 0.001
        t = U @ torch.diag(S) @ V

        denoised = spectral_denoise(t, energy_threshold=0.99, max_condition=1000)

        # Frobenius error should be small
        error = (t - denoised).norm() / t.norm()
        assert error < 0.05  # < 5% relative error

    def test_min_rank_ratio_prevents_over_truncation(self):
        """min_rank_ratio should prevent removing too many singular values."""
        # Create matrix where condition-based truncation would want rank=1
        # but min_rank_ratio=0.5 should keep at least 16
        U = torch.linalg.qr(torch.randn(32, 32))[0]
        V = torch.linalg.qr(torch.randn(32, 32))[0]
        S = torch.zeros(32)
        S[0] = 1e6
        for i in range(1, 32):
            S[i] = 1e-3
        t = U @ torch.diag(S) @ V

        denoised = spectral_denoise(t, max_condition=100, min_rank_ratio=0.5)

        # Should keep at least 16 singular values (50% of 32)
        sv_after = torch.linalg.svdvals(denoised)
        nonzero_svs = (sv_after > 1e-10).sum().item()
        assert nonzero_svs >= 16

    def test_1d_tensor_skip(self):
        """1D tensors should be returned unchanged."""
        t = torch.randn(64)
        result = spectral_denoise(t)
        assert torch.equal(t, result)

    def test_small_tensor_skip(self):
        """Tensors smaller than 2x2 should be returned unchanged."""
        t = torch.randn(1, 64)
        result = spectral_denoise(t)
        assert torch.equal(t, result)

    def test_well_conditioned_noop(self):
        """Well-conditioned tensor should not be modified."""
        # Identity-like matrix (condition = 1)
        t = torch.eye(32) + torch.randn(32, 32) * 0.01
        original = t.clone()
        result = spectral_denoise(t, max_condition=1000)
        # Should be identical (no change needed)
        assert torch.equal(result, original)

    def test_dtype_preserved(self):
        """Output dtype should match input dtype."""
        U = torch.linalg.qr(torch.randn(32, 32))[0]
        V = torch.linalg.qr(torch.randn(32, 32))[0]
        S = torch.zeros(32)
        S[0] = 1e6
        for i in range(1, 32):
            S[i] = 1e-3
        t = (U @ torch.diag(S) @ V).half()

        result = spectral_denoise(t)
        assert result.dtype == torch.float16

    def test_shape_preserved(self):
        """Output shape should match input shape."""
        U = torch.linalg.qr(torch.randn(64, 64))[0][:, :32]
        V = torch.linalg.qr(torch.randn(32, 32))[0]
        S = torch.zeros(32)
        S[0] = 1e6
        for i in range(1, 32):
            S[i] = 1e-3
        t = U @ torch.diag(S) @ V
        assert t.shape == (64, 32)

        result = spectral_denoise(t)
        assert result.shape == (64, 32)


class TestSpectralReport:
    """Test spectral_denoise_with_report returns correct report."""

    def test_report_fields(self):
        U = torch.linalg.qr(torch.randn(32, 32))[0]
        V = torch.linalg.qr(torch.randn(32, 32))[0]
        S = torch.zeros(32)
        S[0] = 1e5
        S[1] = 1e2
        for i in range(2, 32):
            S[i] = 1e-3
        t = U @ torch.diag(S) @ V

        denoised, report = spectral_denoise_with_report(t, "test.weight")
        assert isinstance(report, SpectralReport)
        assert report.param_name == "test.weight"
        assert report.effective_rank < report.original_rank
        assert report.energy_retained > 0.9
        assert report.condition_before > 10_000
        assert report.condition_after < report.condition_before
        assert report.frobenius_error < 0.1

    def test_report_noop(self):
        """Well-conditioned tensor should report no change."""
        t = torch.eye(32) + torch.randn(32, 32) * 0.01
        _, report = spectral_denoise_with_report(t, "identity.weight")
        assert report.effective_rank == report.original_rank
        assert report.energy_retained == 1.0
        assert report.frobenius_error == 0.0

    def test_report_1d(self):
        t = torch.randn(64)
        _, report = spectral_denoise_with_report(t, "bias")
        assert report.energy_retained == 1.0


class TestPrescriptionIntegration:
    """Test spectral_denoise integrates with prescribe() and apply_treatment()."""

    def test_prescribe_generates_spectral_rx(self):
        """gradient_noise finding with condition > 10K should get spectral_denoise Rx."""
        finding = Finding(
            condition="gradient_noise",
            severity="WARN",
            param_name="layers.0.self_attn.q_proj.weight",
            details={
                "condition_number": 500_000,
                "max_sv": 100.0,
                "min_sv": 0.0002,
                "shape": [512, 512],
            },
        )
        prescriptions = prescribe([finding])
        assert len(prescriptions) == 1
        rx = prescriptions[0]
        assert rx.action == "spectral_denoise"
        assert rx.params["energy_threshold"] == 0.99
        assert rx.params["max_condition"] == 1000

    def test_prescribe_advisory_for_moderate_condition(self):
        """gradient_noise finding with condition < 10K should get advisory."""
        finding = Finding(
            condition="gradient_noise",
            severity="WARN",
            param_name="layers.0.weight",
            details={
                "condition_number": 5_000,
                "max_sv": 10.0,
                "min_sv": 0.002,
                "shape": [64, 64],
            },
        )
        prescriptions = prescribe([finding])
        assert len(prescriptions) == 1
        rx = prescriptions[0]
        assert rx.action == "advisory"

    def test_apply_treatment_spectral(self):
        """apply_treatment with spectral_denoise action should fix condition number."""
        # Create a poorly-conditioned matrix
        U = torch.linalg.qr(torch.randn(64, 64))[0]
        V = torch.linalg.qr(torch.randn(64, 64))[0]
        S = torch.zeros(64)
        S[0] = 1e5
        S[1] = 1e3
        S[2] = 10.0
        for i in range(3, 64):
            S[i] = 1e-4
        tensor = U @ torch.diag(S) @ V
        original = tensor.clone()
        param_name = "layers.0.self_attn.q_proj.weight"

        state_dict = {param_name: tensor}

        finding = Finding(
            condition="gradient_noise",
            severity="WARN",
            param_name=param_name,
            details={"condition_number": 1e9, "max_sv": 1e5, "min_sv": 1e-4,
                      "shape": [64, 64]},
        )
        rx = Prescription(
            name="spectral_denoise",
            description="Test spectral denoise",
            risk="medium",
            finding=finding,
            action="spectral_denoise",
            params={"energy_threshold": 0.99, "max_condition": 1000, "min_rank_ratio": 0.1},
        )

        result = apply_treatment(state_dict, rx)
        assert result.success
        assert "Spectral denoise" in result.description

        # Verify the tensor was actually modified
        treated = state_dict[param_name]
        assert not torch.equal(treated, original)

        # Use spectral_analysis to check the effective rank dropped
        info = spectral_analysis(treated)
        assert info["effective_rank"] <= 10  # was 64, should be much smaller

        # Reconstruction error should be small
        error = (original - treated).norm() / original.norm()
        assert error < 0.05

    def test_apply_treatment_noop_for_good_tensor(self):
        """apply_treatment should not modify a well-conditioned tensor."""
        t = torch.eye(32) + torch.randn(32, 32) * 0.01
        param_name = "healthy.weight"
        state_dict = {param_name: t.clone()}

        finding = Finding("gradient_noise", "WARN", param_name,
                          {"condition_number": 50001, "max_sv": 1.0, "min_sv": 0.9,
                           "shape": [32, 32]})
        rx = Prescription("spectral_denoise", "Test", "medium", finding,
                          "spectral_denoise",
                          {"energy_threshold": 0.99, "max_condition": 1000,
                           "min_rank_ratio": 0.1})

        result = apply_treatment(state_dict, rx)
        assert result.success
        assert "well-conditioned" in result.description

    def test_end_to_end_diagnose_prescribe_treat(self):
        """Full pipeline: diagnose a bad tensor, prescribe, and treat."""
        # Build a state dict with one badly-conditioned matrix
        U = torch.linalg.qr(torch.randn(256, 256))[0]
        V = torch.linalg.qr(torch.randn(256, 256))[0]
        S = torch.zeros(256)
        S[0] = 1e5
        S[1] = 1e3
        for i in range(2, 256):
            S[i] = 1e-5
        bad_tensor = U @ torch.diag(S) @ V
        state_dict = {"layers.0.self_attn.q_proj.weight": bad_tensor}

        # Diagnose
        findings = diagnose(state_dict)
        grad_noise = [f for f in findings if f.condition == "gradient_noise"]
        # Should detect high condition number
        assert len(grad_noise) > 0

        # Prescribe
        prescriptions = prescribe(findings)
        spectral_rx = [rx for rx in prescriptions if rx.action == "spectral_denoise"]
        assert len(spectral_rx) > 0

        # Treat
        for rx in spectral_rx:
            result = apply_treatment(state_dict, rx)
            assert result.success
