"""Tests for the health score module."""

from io import StringIO

from model_clinic._types import Finding, HealthScore
from model_clinic._health_score import compute_health_score, print_health_score


def _finding(condition, severity="WARN", param="test.weight"):
    return Finding(condition=condition, severity=severity, param_name=param)


class TestHealthScore:

    def test_perfect_score_no_findings(self):
        score = compute_health_score([])
        assert score.overall == 100
        assert score.grade == "A"
        assert all(v == 100 for v in score.categories.values())

    def test_single_error_deducts_25(self):
        score = compute_health_score([_finding("dead_neurons", "ERROR")])
        assert score.categories["weights"] == 75
        # No runtime findings -> weights gets 40/70 proportion
        assert score.categories["stability"] == 100

    def test_single_warn_deducts_10(self):
        score = compute_health_score([_finding("norm_drift", "WARN")])
        assert score.categories["stability"] == 90

    def test_single_info_deducts_2(self):
        score = compute_health_score([_finding("heavy_tails", "INFO")])
        assert score.categories["weights"] == 98

    def test_category_floor_at_zero(self):
        # With penalty caps, many findings of the SAME condition can't zero a category.
        # Use many DIFFERENT weight conditions to actually floor it.
        findings = (
            [_finding("dead_neurons", "ERROR")] * 5 +        # cap 35 → weights -35
            [_finding("exploding_norm", "ERROR")] * 5 +      # cap 35 → weights -35
            [_finding("heavy_tails", "ERROR")] * 5 +         # cap 35 → weights -35
            [_finding("saturated_weights", "ERROR")] * 5     # cap 35 → weights -35
        )
        score = compute_health_score(findings)
        assert score.categories["weights"] == 0

    def test_multiple_categories(self):
        findings = [
            _finding("dead_neurons", "ERROR"),
            _finding("nan_inf", "ERROR"),
        ]
        score = compute_health_score(findings)
        assert score.categories["weights"] == 75
        assert score.categories["stability"] == 75

    def test_runtime_categories_weighted_when_present(self):
        findings = [
            _finding("generation_collapse", "ERROR", "model"),
        ]
        score = compute_health_score(findings)
        # output category hit, runtime is present
        assert score.categories["output"] == 75
        # Overall uses full 4-category weighting
        # weights=100*.4 + stability=100*.3 + output=75*.2 + activations=100*.1
        # = 40 + 30 + 15 + 10 = 95
        assert score.overall == 95

    def test_no_runtime_redistributes_weights(self):
        # Only static findings — output and activations get 0 weight
        findings = [_finding("dead_neurons", "ERROR")]
        score = compute_health_score(findings)
        # weights=75, stability=100, no runtime
        # Redistributed: weights=0.4/0.7, stability=0.3/0.7
        # = 75*(4/7) + 100*(3/7) = 42.86 + 42.86 = 85.71 -> 86
        assert score.overall == 86

    def test_grade_a(self):
        score = compute_health_score([])
        assert score.grade == "A"

    def test_grade_b(self):
        # Need overall 80-89
        findings = [_finding("dead_neurons", "ERROR")]
        score = compute_health_score(findings)
        assert score.grade == "B"

    def test_grade_c(self):
        # 2 ERRORs on weights (50) + 1 WARN on stability (90)
        # 50*(4/7) + 90*(3/7) = 28.57 + 38.57 = 67 -> C
        findings = [
            _finding("dead_neurons", "ERROR"),
            _finding("exploding_norm", "ERROR"),
            _finding("norm_drift", "WARN"),
        ]
        score = compute_health_score(findings)
        assert score.grade == "C"

    def test_grade_d(self):
        # 3 different ERROR conditions on weights → weights=25
        # 25*(4/7) + 100*(3/7) = 14.29 + 42.86 = 57 → D
        findings = [
            _finding("dead_neurons", "ERROR"),
            _finding("exploding_norm", "ERROR"),
            _finding("heavy_tails", "ERROR"),
        ]
        score = compute_health_score(findings)
        assert score.grade == "D"

    def test_grade_f(self):
        # Floor both weights and stability with multiple distinct ERROR conditions
        findings = (
            [_finding("dead_neurons", "ERROR")] * 3 +
            [_finding("exploding_norm", "ERROR")] * 3 +
            [_finding("heavy_tails", "ERROR")] * 3 +
            [_finding("saturated_weights", "ERROR")] * 3 +
            [_finding("nan_inf", "ERROR")] * 3 +
            [_finding("stuck_gate_closed", "ERROR")] * 3 +
            [_finding("stuck_gate_open", "ERROR")] * 3 +
            [_finding("gradient_noise", "ERROR")] * 3
        )
        score = compute_health_score(findings)
        assert score.grade == "F"

    def test_summary_healthy(self):
        score = compute_health_score([])
        assert "healthy" in score.summary.lower()

    def test_summary_unhealthy_mentions_worst(self):
        findings = [_finding("dead_neurons", "ERROR")] * 4
        score = compute_health_score(findings)
        assert "weights" in score.summary

    def test_all_static_conditions(self):
        conditions = [
            "dead_neurons", "vanishing_norm", "exploding_norm",
            "heavy_tails", "saturated_weights", "identical_rows",
            "nan_inf", "norm_drift", "stuck_gate_closed", "stuck_gate_open",
        ]
        findings = [_finding(c, "WARN") for c in conditions]
        score = compute_health_score(findings)
        assert 0 <= score.overall <= 100
        assert score.categories["weights"] == 40   # 6 WARNs = -60
        assert score.categories["stability"] == 60  # 4 WARNs = -40

    def test_all_runtime_conditions(self):
        conditions = [
            "generation_collapse", "low_coherence", "low_entropy",
            "response_uniformity", "activation_nan", "activation_inf",
            "activation_explosion", "activation_collapse",
            "residual_explosion", "residual_collapse",
        ]
        findings = [_finding(c, "WARN", "model") for c in conditions]
        score = compute_health_score(findings)
        assert score.categories["output"] == 60   # 4 WARNs = -40
        assert score.categories["activations"] == 40  # 6 WARNs = -60

    def test_unknown_condition_defaults_to_weights(self):
        score = compute_health_score([_finding("something_new", "WARN")])
        assert score.categories["weights"] == 90

    def test_return_type(self):
        score = compute_health_score([])
        assert isinstance(score, HealthScore)
        assert isinstance(score.overall, int)
        assert isinstance(score.categories, dict)
        assert isinstance(score.grade, str)
        assert isinstance(score.summary, str)


class TestPrintHealthScore:

    def test_prints_without_error(self):
        score = compute_health_score([])
        buf = StringIO()
        print_health_score(score, file=buf)
        output = buf.getvalue()
        assert "100/100" in output
        assert "A" in output

    def test_prints_all_categories(self):
        score = compute_health_score([_finding("dead_neurons", "WARN")])
        buf = StringIO()
        print_health_score(score, file=buf)
        output = buf.getvalue()
        for cat in ["weights", "stability", "output", "activations"]:
            assert cat in output

    def test_prints_grade_f(self):
        # Use multiple distinct ERROR conditions to actually reach F grade
        findings = (
            [_finding("dead_neurons", "ERROR")] * 3 +
            [_finding("exploding_norm", "ERROR")] * 3 +
            [_finding("heavy_tails", "ERROR")] * 3 +
            [_finding("saturated_weights", "ERROR")] * 3 +
            [_finding("nan_inf", "ERROR")] * 3 +
            [_finding("stuck_gate_closed", "ERROR")] * 3 +
            [_finding("stuck_gate_open", "ERROR")] * 3 +
            [_finding("gradient_noise", "ERROR")] * 3
        )
        score = compute_health_score(findings)
        buf = StringIO()
        print_health_score(score, file=buf)
        output = buf.getvalue()
        assert "F" in output
