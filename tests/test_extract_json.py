"""Regression tests for `_extract_json` JSON extraction from model responses.

`_extract_json` is the core of `TransformerModel.structured_output`: it recovers
a JSON object/array from a chatty/reasoning model reply. These cases pin the
behaviour that a balanced-span fallback must be string-aware (a brace or bracket
inside a JSON string value must not close the span) and must anchor on the first
delimiter (a top-level array of objects must not collapse to its inner object).
"""

import json

import pytest

from strands_transformers.models.transformers import _extract_json


def _parsed(text):
    return json.loads(_extract_json(text))


# ── happy paths that short-circuit on the early json.loads ──
@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 2}\n```', {"a": 2}),
    ],
)
def test_clean_and_fenced(text, expected):
    assert _parsed(text) == expected


# ── reasoning-model prefixes ──
@pytest.mark.parametrize(
    "text,expected",
    [
        ('<think>reasoning</think>{"a": 3}', {"a": 3}),
        ('<think>unterminated reasoning {"a": 4}', {"a": 4}),
        ('Here is the JSON: {"a": 5} done', {"a": 5}),
        ('x {"a": {"b": 1}} y', {"a": {"b": 1}}),
    ],
)
def test_prose_and_think_prefixes(text, expected):
    assert _parsed(text) == expected


# ── regression: braces/brackets INSIDE string values must not close the span ──
@pytest.mark.parametrize(
    "text,expected",
    [
        ('Sure! {"note": "use } carefully"} done', {"note": "use } carefully"}),
        (
            'Here is the JSON:\n{"command": "echo }", "ok": true}\nend',
            {"command": "echo }", "ok": True},
        ),
        (
            'prefix {"a": {"b": "v}"}, "c": 1} suffix',
            {"a": {"b": "v}"}, "c": 1},
        ),
        ('data {"path": "a/b]c", "n": 2}', {"path": "a/b]c", "n": 2}),
        (
            r'{"s": "he said \"}\" loudly", "n": 3}',
            {"s": 'he said "}" loudly', "n": 3},
        ),
    ],
)
def test_brace_inside_string_literal(text, expected):
    assert _parsed(text) == expected


# ── regression: a top-level array of objects must not collapse to inner object ──
@pytest.mark.parametrize(
    "text,expected",
    [
        ('result: [{"k": "}"}]', [{"k": "}"}]),
        ('Here: [{"a": 1}, {"b": 2}]', [{"a": 1}, {"b": 2}]),
        ('{"items": [1, 2], "n": 3}', {"items": [1, 2], "n": 3}),
    ],
)
def test_earliest_delimiter_anchoring(text, expected):
    assert _parsed(text) == expected


def test_no_json_returns_text_untouched():
    assert _extract_json("no json here") == "no json here"
