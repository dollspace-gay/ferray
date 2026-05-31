"""Verify ferray.char classify predicates match numpy.char element by element.

Divergence cluster #912 (refs #836): the `is*` classify family in
``ferray-strings`` must mirror CPython ``str.is*()`` (which ``numpy.char``
delegates to per element). These assertions compare ``ferray.char`` against
``numpy.char`` directly (the live oracle, numpy 2.4.x / Unicode 15.1.0), so
they are non-tautological (R-CHAR-3): the expected side is computed by numpy,
never literal-copied from the ferray side.

Covers the predicates fixed in #912 (isdecimal, isalpha, isalnum, istitle,
isupper, islower, isspace) plus the previously-fixed isdigit (#911) and
isnumeric guards to assert no regression.
"""

import numpy as np
import pytest

import ferray as fr

# A truth set spanning every category-sensitive corner the predicates care
# about: ASCII, accented Latin, CJK, titlecase digraphs (Lt), roman numerals
# (Nl with Other_Upper/Lowercase), circled letters/digits (So), superscripts
# and fractions (No), non-ASCII decimals (fullwidth, Arabic-Indic), and a
# spread of Unicode whitespace.
TRUTH_SET = [
    "",
    "abc",
    "ABC",
    "Hello",
    "Hello World",
    "hELLO",
    "abc123",
    "a1",
    "A1",
    "ab c",
    "café",
    "日本",
    "ǅ",        # U+01C5 titlecase Lt
    "ǅx",
    "ǅX",
    "Ǆ",        # U+01C4 uppercase Lu digraph
    "ǆ",        # U+01C6 lowercase Ll digraph
    "Ⅻ",        # U+216B roman 12 (Nl, Uppercase property)
    "ⅻ",        # U+217B roman 12 lower (Nl, Lowercase property)
    "Ⓐ",        # U+24B6 circled A (So, Uppercase property)
    "②",        # U+2461 circled 2 (digit)
    "²",        # U+00B2 superscript 2 (digit, not decimal)
    "³⁴",       # superscripts
    "½",        # U+00BD fraction (numeric, not digit)
    "①",        # U+2460 circled 1 (digit)
    "０",        # U+FF10 fullwidth 0 (decimal)
    "０９",
    "٣",        # U+0663 Arabic-Indic 3 (decimal)
    "123",
    "12.3",
    "12a",
    " ",   # NBSP (White_Space)
    " ",   # line separator
    " ",   # paragraph separator
    "\x1c",     # file separator (bidi B -> isspace True)
    "\x1f",     # unit separator
    "​",   # zero-width space (NOT isspace)
    " ",
    "\t\n",
    "a b",
]

PREDICATES = [
    "isalpha",
    "isdigit",
    "isspace",
    "isupper",
    "islower",
    "isalnum",
    "isnumeric",
    "isdecimal",
    "istitle",
]


@pytest.mark.parametrize("name", PREDICATES)
def test_char_classify_matches_numpy(name):
    arr = TRUTH_SET
    fr_fn = getattr(fr.char, name)
    np_fn = getattr(np.char, name)
    fr_res = np.asarray(fr_fn(np.array(arr)))
    np_res = np.asarray(np_fn(np.array(arr)))
    mism = [
        (repr(s), bool(f), bool(g))
        for s, f, g in zip(arr, fr_res, np_res)
        if bool(f) != bool(g)
    ]
    assert not mism, f"{name} mismatches (str, ferray, numpy): {mism}"
