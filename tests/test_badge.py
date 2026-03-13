"""Tests for the badge generation module."""

from urllib.parse import unquote

import pytest

from model_clinic._types import HealthScore
from model_clinic._badge import (
    generate_badge_url,
    generate_badge_svg,
    generate_model_card_snippet,
    save_badge_svg,
    GRADE_COLORS,
    GRADE_COLOR_NAMES,
)
from model_clinic._health_score import compute_health_score
from model_clinic._types import Finding


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_score(overall, grade):
    """Build a minimal HealthScore without running the full scorer."""
    return HealthScore(
        overall=overall,
        categories={"weights": overall, "stability": overall,
                    "output": overall, "activations": overall},
        grade=grade,
        summary=f"Grade {grade}",
    )


def _finding(condition, severity="WARN", param="test.weight"):
    return Finding(condition=condition, severity=severity, param_name=param)


# ── generate_badge_url ────────────────────────────────────────────────────────

class TestGenerateBadgeUrl:

    def test_returns_string(self):
        score = _make_score(76, "C")
        url = generate_badge_url(score)
        assert isinstance(url, str)

    def test_starts_with_shields_io(self):
        score = _make_score(76, "C")
        url = generate_badge_url(score)
        assert url.startswith("https://img.shields.io/badge/")

    def test_contains_label(self):
        score = _make_score(90, "A")
        url = generate_badge_url(score)
        assert "model--clinic" in url

    def test_contains_score(self):
        score = _make_score(76, "C")
        url = generate_badge_url(score)
        # URL-encoded: "76/100 C" -> "76%2F100%20C"
        decoded = unquote(url)
        assert "76/100" in decoded
        assert " C" in decoded

    def test_grade_a_is_brightgreen(self):
        score = _make_score(95, "A")
        url = generate_badge_url(score)
        assert url.endswith("brightgreen")

    def test_grade_b_is_green(self):
        score = _make_score(82, "B")
        url = generate_badge_url(score)
        assert url.endswith("green")

    def test_grade_c_is_yellow(self):
        score = _make_score(70, "C")
        url = generate_badge_url(score)
        assert url.endswith("yellow")

    def test_grade_d_is_orange(self):
        score = _make_score(55, "D")
        url = generate_badge_url(score)
        assert url.endswith("orange")

    def test_grade_f_is_red(self):
        score = _make_score(30, "F")
        url = generate_badge_url(score)
        assert url.endswith("red")

    def test_all_grades_produce_different_colors(self):
        grades = [("A", 95), ("B", 82), ("C", 70), ("D", 55), ("F", 30)]
        urls = [generate_badge_url(_make_score(s, g)) for g, s in grades]
        # All URLs are distinct (different colors)
        assert len(set(urls)) == len(urls)

    def test_url_encoding_special_chars(self):
        """Score with slash and space must be URL-encoded."""
        score = _make_score(100, "A")
        url = generate_badge_url(score)
        # The raw message "100/100 A" must not appear unencoded in the URL
        # (slash and space should be percent-encoded)
        path_part = url.split("/badge/")[1]
        # After stripping label and color, slash in score should be encoded
        assert "/100 A" not in path_part  # space must be encoded
        assert "100/100" not in path_part  # slash must be encoded

    def test_url_no_spaces(self):
        score = _make_score(76, "C")
        url = generate_badge_url(score)
        assert " " not in url


# ── generate_badge_svg ────────────────────────────────────────────────────────

class TestGenerateBadgeSvg:

    def test_returns_string(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert isinstance(svg, str)

    def test_non_empty(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert len(svg) > 0

    def test_contains_svg_open_tag(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert "<svg" in svg

    def test_contains_svg_close_tag(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert "</svg>" in svg

    def test_contains_label_text(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert "model-clinic" in svg

    def test_contains_score_text(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert "76/100" in svg
        assert " C" in svg

    def test_grade_a_color_in_svg(self):
        score = _make_score(95, "A")
        svg = generate_badge_svg(score)
        assert GRADE_COLORS["A"] in svg

    def test_grade_b_color_in_svg(self):
        score = _make_score(82, "B")
        svg = generate_badge_svg(score)
        assert GRADE_COLORS["B"] in svg

    def test_grade_c_color_in_svg(self):
        score = _make_score(70, "C")
        svg = generate_badge_svg(score)
        assert GRADE_COLORS["C"] in svg

    def test_grade_d_color_in_svg(self):
        score = _make_score(55, "D")
        svg = generate_badge_svg(score)
        assert GRADE_COLORS["D"] in svg

    def test_grade_f_color_in_svg(self):
        score = _make_score(30, "F")
        svg = generate_badge_svg(score)
        assert GRADE_COLORS["F"] in svg

    def test_svg_has_width_attribute(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert 'width="' in svg

    def test_svg_has_height_20(self):
        score = _make_score(76, "C")
        svg = generate_badge_svg(score)
        assert 'height="20"' in svg

    def test_svg_different_scores_different_widths(self):
        # A 3-digit score should produce a wider badge than a 2-digit one
        # (more characters in the message half)
        score_wide = _make_score(100, "A")
        score_narrow = _make_score(1, "F")
        svg_wide = generate_badge_svg(score_wide)
        svg_narrow = generate_badge_svg(score_narrow)
        # Extract the outer width value
        import re
        w_wide = int(re.search(r'<svg[^>]+width="(\d+)"', svg_wide).group(1))
        w_narrow = int(re.search(r'<svg[^>]+width="(\d+)"', svg_narrow).group(1))
        assert w_wide >= w_narrow


# ── generate_model_card_snippet ───────────────────────────────────────────────

class TestGenerateModelCardSnippet:

    def _score_and_findings(self):
        findings = [
            _finding("dead_neurons", "ERROR"),
            _finding("norm_drift", "WARN"),
            _finding("heavy_tails", "INFO"),
        ]
        health = compute_health_score(findings)
        return health, findings

    def test_returns_string(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert isinstance(snippet, str)

    def test_contains_badge_url(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        expected_url = generate_badge_url(health)
        assert expected_url in snippet

    def test_contains_shields_io_image(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert "![model-clinic]" in snippet
        assert "img.shields.io" in snippet

    def test_contains_category_table_header(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert "| Category | Score |" in snippet

    def test_contains_all_categories(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        for cat in ["weights", "stability", "output", "activations"]:
            assert cat in snippet

    def test_contains_overall_score(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert str(health.overall) in snippet

    def test_contains_grade(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert health.grade in snippet

    def test_contains_pypi_link(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert "pypi.org/project/model-clinic" in snippet

    def test_contains_model_clinic_link_text(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert "model-clinic" in snippet

    def test_contains_model_name_when_provided(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings, model_name="MyModel")
        assert "MyModel" in snippet

    def test_no_model_name_when_not_provided(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings, model_name="")
        # Should still be valid markdown without a model name line
        assert "## Model Health" in snippet

    def test_contains_findings_section(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        # At least one finding condition should appear
        assert "dead_neurons" in snippet

    def test_top_5_findings_limit(self):
        """With 10 findings, only top 5 should appear by severity order."""
        many_findings = (
            [_finding("dead_neurons", "ERROR")] * 3 +
            [_finding("norm_drift", "WARN")] * 4 +
            [_finding("heavy_tails", "INFO")] * 3
        )
        health = compute_health_score(many_findings)
        snippet = generate_model_card_snippet(health, many_findings)
        # Count bullet points in findings section
        bullet_count = snippet.count("- `ERROR`") + snippet.count("- `WARN`") + snippet.count("- `INFO`")
        assert bullet_count <= 5

    def test_empty_findings_no_findings_section(self):
        health = compute_health_score([])
        snippet = generate_model_card_snippet(health, [])
        # No findings bullets when no findings
        assert "- `ERROR`" not in snippet
        assert "- `WARN`" not in snippet

    def test_markdown_h2_header(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert snippet.startswith("## Model Health")

    def test_contains_version(self):
        health, findings = self._score_and_findings()
        snippet = generate_model_card_snippet(health, findings)
        assert "v0.3.0" in snippet


# ── Color mapping correctness ─────────────────────────────────────────────────

class TestColorMapping:

    def test_all_grades_have_hex_color(self):
        for grade in ("A", "B", "C", "D", "F"):
            assert grade in GRADE_COLORS
            assert GRADE_COLORS[grade].startswith("#")

    def test_all_grades_have_color_name(self):
        for grade in ("A", "B", "C", "D", "F"):
            assert grade in GRADE_COLOR_NAMES
            assert isinstance(GRADE_COLOR_NAMES[grade], str)

    def test_grade_a_is_bright(self):
        assert GRADE_COLORS["A"] == "#4c1"
        assert GRADE_COLOR_NAMES["A"] == "brightgreen"

    def test_grade_f_is_red(self):
        assert GRADE_COLORS["F"] == "#e05d44"
        assert GRADE_COLOR_NAMES["F"] == "red"

    def test_hex_colors_are_unique(self):
        colors = list(GRADE_COLORS.values())
        assert len(colors) == len(set(colors))

    def test_color_names_are_unique(self):
        names = list(GRADE_COLOR_NAMES.values())
        assert len(names) == len(set(names))


# ── save_badge_svg ────────────────────────────────────────────────────────────

class TestSaveBadgeSvg:

    def test_saves_file(self, tmp_path):
        score = _make_score(76, "C")
        out = tmp_path / "badge.svg"
        save_badge_svg(score, str(out))
        assert out.exists()

    def test_saved_file_contains_svg(self, tmp_path):
        score = _make_score(76, "C")
        out = tmp_path / "badge.svg"
        save_badge_svg(score, str(out))
        content = out.read_text(encoding="utf-8")
        assert "<svg" in content
        assert "</svg>" in content

    def test_saved_content_matches_generate(self, tmp_path):
        score = _make_score(82, "B")
        out = tmp_path / "badge.svg"
        save_badge_svg(score, str(out))
        expected = generate_badge_svg(score)
        actual = out.read_text(encoding="utf-8")
        assert actual == expected


# ── Integration: compute_health_score -> badge ────────────────────────────────

class TestBadgeIntegration:

    def test_perfect_score_badge_url(self):
        health = compute_health_score([])
        url = generate_badge_url(health)
        assert "100" in unquote(url)
        assert url.endswith("brightgreen")

    def test_broken_model_badge_url(self):
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
        health = compute_health_score(findings)
        url = generate_badge_url(health)
        assert url.endswith("red")

    def test_model_card_snippet_is_valid_markdown(self):
        findings = [_finding("norm_drift", "WARN")]
        health = compute_health_score(findings)
        snippet = generate_model_card_snippet(health, findings, model_name="test-model")
        # Must start with a heading
        assert snippet.startswith("##")
        # Must contain a table separator
        assert "|---" in snippet
