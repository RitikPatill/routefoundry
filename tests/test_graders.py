"""Grader contract tests.

Fairness cases matter as much as correctness cases here: a grader that quietly punishes
one model family's output format would bias every routing conclusion drawn from it.
"""

from __future__ import annotations

import pytest

from routefoundry.graders import (
    GraderError,
    grade,
    grade_contains_all,
    grade_exact_number,
    grade_exact_string,
    grade_json_field,
    grade_regex,
    normalise_response,
)


class TestNormaliseResponse:
    def test_removes_reasoning_block(self) -> None:
        raw = "<think>The user wants 2+2. That is 4.</think>4"
        assert normalise_response(raw) == "4"

    @pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "scratchpad"])
    def test_removes_every_supported_reasoning_tag(self, tag: str) -> None:
        assert normalise_response(f"<{tag}>hidden work</{tag}> positive") == "positive"

    def test_unclosed_reasoning_block_leaves_no_answer(self) -> None:
        # The model exhausted its token budget mid-thought; there is no answer to credit.
        assert normalise_response("<think>Let me work through this slowly") == ""

    def test_strips_code_fence_and_preamble(self) -> None:
        assert normalise_response("```python\nreturn 7\n```") == "return 7"
        assert normalise_response("The answer is: 42") == "42"
        assert normalise_response("Final answer = spam") == "spam"

    def test_rejects_non_string(self) -> None:
        with pytest.raises(GraderError):
            normalise_response(7)  # type: ignore[arg-type]


class TestExactNumber:
    def test_plain_and_wrapped_answers(self) -> None:
        assert grade_exact_number("18", "18").correct
        assert grade_exact_number("The answer is 18.", "18").correct
        assert grade_exact_number("<think>6*3=18</think>18", "18").correct

    def test_anchors_on_final_number_not_intermediate_working(self) -> None:
        # Intermediate arithmetic mentions 18; the stated answer is wrong and must fail.
        assert not grade_exact_number("6 times 3 is 18, so the total is 20.", "18").correct
        assert grade_exact_number("First 5, then 13, total 18", "18").correct

    def test_thousands_separator_and_negatives(self) -> None:
        assert grade_exact_number("1,250", "1250").correct
        assert grade_exact_number("-7", "-7").correct

    def test_tolerance(self) -> None:
        assert not grade_exact_number("3.14", "3.14159").correct
        assert grade_exact_number("3.14", "3.14159", tolerance="0.01").correct

    def test_missing_or_empty(self) -> None:
        assert grade_exact_number("no idea", "5").reason == "no number in response"
        assert grade_exact_number("", "5").reason == "empty response"

    def test_invalid_specification(self) -> None:
        with pytest.raises(GraderError):
            grade_exact_number("1", "not-a-number")
        with pytest.raises(GraderError):
            grade_exact_number("1", "1", tolerance="-1")


class TestExactString:
    def test_case_and_punctuation_tolerance(self) -> None:
        assert grade_exact_string("Positive", "positive").correct
        assert grade_exact_string('"positive."', "positive").correct
        assert grade_exact_string("**positive**", "positive").correct

    def test_final_line_match_for_chatty_models(self) -> None:
        assert grade_exact_string("Sure! Here it is:\npositive", "positive").correct

    def test_first_line_match_when_the_model_answers_then_explains(self) -> None:
        # Observed with gemma:2b: the correct label, then an unprompted justification.
        # Scoring this wrong would penalise verbosity rather than capability.
        response = "Negative.\n\nThe message indicates the package arrived late and was damaged."
        assert grade_exact_string(response, "negative").correct

    def test_rejects_wrong_label_and_label_lists(self) -> None:
        assert not grade_exact_string("negative", "positive").correct
        # Listing every option must not earn credit.
        assert not grade_exact_string("positive, negative, neutral", "positive").correct

    def test_rejects_a_line_that_merely_contains_the_label(self) -> None:
        # A whole-line match is required, so a negated or hedged line earns nothing.
        assert not grade_exact_string("not negative\nstill unsure", "negative").correct
        assert not grade_exact_string("either positive or negative", "negative").correct

    def test_empty_expected_is_a_configuration_error(self) -> None:
        with pytest.raises(GraderError):
            grade_exact_string("x", "  ")


class TestJsonField:
    def test_extracts_field_from_surrounding_prose(self) -> None:
        response = 'Here you go: {"invoice": "INV-42", "due": "2026-01-01"}'
        assert grade_json_field(response, "INV-42", field="invoice").correct

    def test_non_string_values_are_compared_by_rendering(self) -> None:
        assert grade_json_field('{"count": 7}', "7", field="count").correct

    def test_failure_modes(self) -> None:
        assert grade_json_field("no json here", "x", field="a").reason == "no JSON object in response"
        assert grade_json_field('{"a": }', "x", field="a").reason == "invalid JSON in response"
        assert grade_json_field('{"b": 1}', "x", field="a").reason == "field missing from JSON"

    def test_field_name_is_required(self) -> None:
        with pytest.raises(GraderError):
            grade_json_field('{"a": 1}', "1")


class TestContainsAll:
    def test_all_required_substrings(self) -> None:
        assert grade_contains_all("uses map and filter", "x", required="map,filter").correct

    def test_partial_credit_is_reported_but_not_correct(self) -> None:
        result = grade_contains_all("uses map only", "x", required="map,filter")
        assert not result.correct
        assert result.score == pytest.approx(0.5)

    def test_requires_at_least_one_substring(self) -> None:
        with pytest.raises(GraderError):
            grade_contains_all("text", "", required=" , ")


class TestRegex:
    def test_match_and_mismatch(self) -> None:
        assert grade_regex("line 3 is wrong", "x", pattern=r"\bline\s*3\b").correct
        assert not grade_regex("line 4 is wrong", "x", pattern=r"\bline\s*3\b").correct

    def test_invalid_and_oversized_patterns_are_rejected(self) -> None:
        with pytest.raises(GraderError):
            grade_regex("x", "x", pattern="(unclosed")
        with pytest.raises(GraderError):
            grade_regex("x", "x", pattern="a" * 501)


class TestGradeDispatch:
    def test_dispatches_each_named_grader(self) -> None:
        assert grade("18", "18", "exact_number").correct
        assert grade("positive", "positive", "exact_string").correct
        assert grade('{"a": "b"}', "b", "json_field", "a").correct
        assert grade("map filter", "x", "contains_all", "map,filter").correct
        assert grade("line 3", "x", "regex", r"line\s*3").correct

    def test_unknown_grader_names_the_valid_options(self) -> None:
        with pytest.raises(GraderError, match="exact_number"):
            grade("x", "y", "vibes")
