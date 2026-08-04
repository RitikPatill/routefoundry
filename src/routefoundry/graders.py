"""Deterministic graders for auto-gradable tasks.

Grading decides whether a measured routing result is credible, so these functions are
deliberately conservative and explicit rather than clever.

Two failure modes matter more than raw strictness:

* **Reasoning models.**  Models such as DeepSeek-R1 emit a ``<think>`` block before the
  answer.  Scoring that verbatim would penalise a correct answer for its format and would
  silently bias every downstream routing conclusion.  :func:`normalise_response` removes
  reasoning blocks and common wrappers before any comparison.
* **False credit.**  A grader that accepts any response containing the right token would
  reward a model that lists every option.  Numeric and string graders therefore anchor on
  the *final* answer, and ``contains_all`` is reserved for cases where a single token is
  genuinely too brittle.

Every grader returns :class:`GradeResult` with a score in ``[0, 1]`` and a short,
non-sensitive reason so an audit can explain why a response was judged incorrect.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

GRADER_NAMES: Final = (
    "exact_number",
    "exact_string",
    "json_field",
    "contains_all",
    "regex",
)

MAX_RESPONSE_CHARS: Final = 20_000
MAX_PATTERN_CHARS: Final = 500

# Reasoning-model scratchpads. Content between the tags is never graded.
_REASONING_BLOCK: Final = re.compile(
    r"<(think|thinking|reasoning|scratchpad)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
# An unterminated block means the model ran out of tokens while still reasoning: there is
# no answer after it, so everything from the opening tag onwards is dropped.
_UNCLOSED_REASONING: Final = re.compile(
    r"<(think|thinking|reasoning|scratchpad)\b[^>]*>.*\Z",
    re.IGNORECASE | re.DOTALL,
)
_CODE_FENCE: Final = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?|\n?```\s*$")
# Handles "answer: X", "the answer is X", and "Final answer = X" in one pass; the verb and
# the separator are independently optional because models mix all three forms.
_PREAMBLE: Final = re.compile(
    r"^\s*(?:the\s+)?(?:final\s+)?(?:answer|result|output|value|label)\b"
    r"\s*(?:is\b)?\s*[:=]?\s*",
    re.IGNORECASE,
)
_NUMBER: Final = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+")
_PUNCTUATION_EDGE: Final = re.compile(r"^[\s\"'`*_(\[{]+|[\s\"'`*_.,;:!?)\]}]+$")


class GraderError(ValueError):
    """Raised when a grader is configured with an unusable specification."""


@dataclass(frozen=True, slots=True)
class GradeResult:
    """The outcome of grading one response."""

    score: float
    correct: bool
    reason: str


def normalise_response(text: str) -> str:
    """Strip reasoning blocks, code fences, and answer preambles from a raw response.

    Truncation is applied first so a pathological response cannot make the regular
    expressions do unbounded work.
    """

    if not isinstance(text, str):
        raise GraderError("response must be a string")

    cleaned = text[:MAX_RESPONSE_CHARS]
    cleaned = _REASONING_BLOCK.sub(" ", cleaned)
    cleaned = _UNCLOSED_REASONING.sub(" ", cleaned)
    cleaned = cleaned.strip()
    cleaned = _CODE_FENCE.sub("", cleaned).strip()
    cleaned = _PREAMBLE.sub("", cleaned).strip()
    return cleaned


def _canonical(text: str) -> str:
    """Case-fold and strip surrounding punctuation for tolerant string comparison."""

    collapsed = re.sub(r"\s+", " ", text).strip()
    return _PUNCTUATION_EDGE.sub("", collapsed).casefold()


def _parse_number(token: str) -> float | None:
    try:
        value = float(token.replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def grade_exact_number(response: str, expected: str, *, tolerance: str | None = None) -> GradeResult:
    """Grade a numeric answer, anchoring on the LAST number in the response.

    Anchoring on the last number follows how models actually write: intermediate
    arithmetic appears first and the final answer appears last.  A model that merely
    happens to mention the target number mid-working does not get credit.
    """

    target = _parse_number(expected)
    if target is None:
        raise GraderError(f"expected value {expected!r} is not a finite number")

    allowance = 0.0
    if tolerance not in (None, ""):
        parsed = _parse_number(str(tolerance))
        if parsed is None or parsed < 0:
            raise GraderError(f"tolerance {tolerance!r} must be a non-negative number")
        allowance = parsed

    cleaned = normalise_response(response)
    if not cleaned:
        return GradeResult(0.0, False, "empty response")

    matches = _NUMBER.findall(cleaned)
    if not matches:
        return GradeResult(0.0, False, "no number in response")

    observed = _parse_number(matches[-1])
    if observed is None:
        return GradeResult(0.0, False, "unparseable number in response")

    if abs(observed - target) <= allowance:
        return GradeResult(1.0, True, "numeric match")
    return GradeResult(0.0, False, "numeric mismatch")


def grade_exact_string(response: str, expected: str, *, _arg: str | None = None) -> GradeResult:
    """Grade a short string answer, tolerant of case, wrapping, and edge punctuation.

    A multi-line response is accepted when its final non-empty line matches, which covers
    models that restate the question before answering.
    """

    cleaned = normalise_response(response)
    if not cleaned:
        return GradeResult(0.0, False, "empty response")

    target = _canonical(expected)
    if not target:
        raise GraderError("expected value must not be empty")

    if _canonical(cleaned) == target:
        return GradeResult(1.0, True, "exact match")

    lines = [line for line in cleaned.splitlines() if line.strip()]
    if lines and _canonical(lines[-1]) == target:
        return GradeResult(1.0, True, "final-line match")
    return GradeResult(0.0, False, "string mismatch")


def grade_json_field(response: str, expected: str, *, field: str | None = None) -> GradeResult:
    """Grade one field of a JSON object emitted by the model."""

    if not field:
        raise GraderError("json_field grader requires the field name in grader_arg")

    cleaned = normalise_response(response)
    if not cleaned:
        return GradeResult(0.0, False, "empty response")

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return GradeResult(0.0, False, "no JSON object in response")

    try:
        payload = json.loads(cleaned[start : end + 1])
    except (json.JSONDecodeError, ValueError, RecursionError):
        return GradeResult(0.0, False, "invalid JSON in response")
    if not isinstance(payload, Mapping):
        return GradeResult(0.0, False, "JSON value is not an object")
    if field not in payload:
        return GradeResult(0.0, False, "field missing from JSON")

    value = payload[field]
    rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    if _canonical(str(rendered)) == _canonical(expected):
        return GradeResult(1.0, True, "field match")
    return GradeResult(0.0, False, "field mismatch")


def grade_contains_all(response: str, expected: str, *, required: str | None = None) -> GradeResult:
    """Grade by requiring every comma-separated substring to appear.

    Used only where a single exact token would be unfairly brittle.  Partial credit is
    reported so an audit can distinguish "close" from "absent"; :attr:`GradeResult.correct`
    still requires every substring.
    """

    needles = [item.strip() for item in (required or expected).split(",") if item.strip()]
    if not needles:
        raise GraderError("contains_all grader requires at least one substring")

    haystack = _canonical(normalise_response(response))
    if not haystack:
        return GradeResult(0.0, False, "empty response")

    found = sum(1 for needle in needles if _canonical(needle) in haystack)
    score = found / len(needles)
    if found == len(needles):
        return GradeResult(1.0, True, "all substrings present")
    return GradeResult(score, False, f"{found}/{len(needles)} substrings present")


def grade_regex(response: str, expected: str, *, pattern: str | None = None) -> GradeResult:
    """Grade with a case-insensitive regular expression supplied by the task."""

    source = pattern or expected
    if not source:
        raise GraderError("regex grader requires a pattern")
    if len(source) > MAX_PATTERN_CHARS:
        raise GraderError("regex pattern is too long")

    try:
        compiled = re.compile(source, re.IGNORECASE)
    except re.error as error:
        raise GraderError(f"invalid regex pattern: {error}") from None

    cleaned = normalise_response(response)
    if not cleaned:
        return GradeResult(0.0, False, "empty response")
    if compiled.search(cleaned):
        return GradeResult(1.0, True, "pattern match")
    return GradeResult(0.0, False, "pattern mismatch")


_GRADERS: Final = {
    "exact_number": grade_exact_number,
    "exact_string": grade_exact_string,
    "json_field": grade_json_field,
    "contains_all": grade_contains_all,
    "regex": grade_regex,
}


def grade(response: str, expected: str, grader: str, grader_arg: str | None = None) -> GradeResult:
    """Grade ``response`` against ``expected`` using the named grader."""

    function = _GRADERS.get(grader)
    if function is None:
        raise GraderError(f"unknown grader {grader!r}; expected one of {', '.join(GRADER_NAMES)}")
    return function(response, expected, **{_ARG_NAME[grader]: grader_arg})  # type: ignore[operator]


_ARG_NAME: Final = {
    "exact_number": "tolerance",
    "exact_string": "_arg",
    "json_field": "field",
    "contains_all": "required",
    "regex": "pattern",
}


__all__ = [
    "GRADER_NAMES",
    "GradeResult",
    "GraderError",
    "grade",
    "grade_contains_all",
    "grade_exact_number",
    "grade_exact_string",
    "grade_json_field",
    "grade_regex",
    "normalise_response",
]
