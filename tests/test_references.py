"""Tests for model-clinic condition references."""

from model_clinic._references import (
    CONDITION_REFERENCES,
    get_references,
    format_references,
)


# All conditions that should have references
EXPECTED_CONDITIONS = [
    # Static
    "dead_neurons",
    "stuck_gate_closed",
    "stuck_gate_open",
    "exploding_norm",
    "vanishing_norm",
    "heavy_tails",
    "norm_drift",
    "saturated_weights",
    "nan_inf",
    "identical_rows",
    "attention_imbalance",
    "dtype_mismatch",
    "weight_corruption",
    "head_redundancy",
    "positional_encoding_issues",
    "token_collapse",
    "gradient_noise",
    "representation_drift",
    "moe_router_collapse",
    "lora_merge_artifacts",
    # Runtime
    "generation_collapse",
    "low_coherence",
    "activation_nan",
    "activation_inf",
    "activation_explosion",
    "activation_collapse",
    "residual_explosion",
    "residual_collapse",
]


class TestConditionCoverage:
    """Every registered condition must have at least one reference."""

    def test_all_conditions_have_references(self):
        for condition in EXPECTED_CONDITIONS:
            refs = CONDITION_REFERENCES.get(condition)
            assert refs is not None, f"Missing references for condition: {condition}"
            assert len(refs) >= 1, f"Need at least 1 reference for: {condition}"


class TestGetReferences:
    """get_references returns correct types."""

    def test_known_condition_returns_list(self):
        refs = get_references("dead_neurons")
        assert isinstance(refs, list)
        assert len(refs) >= 1

    def test_unknown_condition_returns_empty_list(self):
        refs = get_references("nonexistent_condition_xyz")
        assert isinstance(refs, list)
        assert len(refs) == 0

    def test_each_reference_has_required_fields(self):
        for condition, refs in CONDITION_REFERENCES.items():
            for ref in refs:
                assert "title" in ref, f"{condition}: reference missing 'title'"
                assert "url" in ref, f"{condition}: reference missing 'url'"


class TestFormatReferences:
    """format_references returns properly formatted strings."""

    def test_known_condition_has_title_and_url(self):
        output = format_references("dead_neurons")
        assert isinstance(output, str)
        assert "dead_neurons" in output
        assert "arxiv" in output.lower() or "http" in output.lower()

    def test_unknown_condition_returns_empty_string(self):
        output = format_references("nonexistent_condition_xyz")
        assert output == ""

    def test_format_includes_note_when_present(self):
        output = format_references("dead_neurons")
        refs = get_references("dead_neurons")
        for ref in refs:
            if ref.get("note"):
                assert ref["note"] in output


class TestReferenceQuality:
    """References should not have empty or placeholder values."""

    def test_no_empty_titles(self):
        for condition, refs in CONDITION_REFERENCES.items():
            for ref in refs:
                assert ref["title"].strip(), f"{condition}: empty title"

    def test_no_empty_urls(self):
        for condition, refs in CONDITION_REFERENCES.items():
            for ref in refs:
                assert ref["url"].strip(), f"{condition}: empty URL"

    def test_urls_look_valid(self):
        for condition, refs in CONDITION_REFERENCES.items():
            for ref in refs:
                url = ref["url"]
                assert url.startswith("http"), (
                    f"{condition}: URL does not start with http: {url}"
                )
