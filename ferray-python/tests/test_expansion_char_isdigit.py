"""Expansion regression: ``ferray.char.isdigit`` vs ``numpy.char.isdigit``.

Pins crosslink blocker #911 (refs #836): ferray-strings ``isdigit`` was too
narrow (ASCII-only), returning ``False`` for Unicode Digit-class characters
(superscripts ``²³``, circled ``①``) that Python/numpy
``str.isdigit`` returns ``True`` for.

Python/numpy ``str.isdigit`` is exactly Unicode ``Numeric_Type`` of Decimal
OR Digit -- broader than ``isdecimal`` (Decimal only) and narrower than
``isnumeric`` (Decimal+Digit+Numeric). Expected values come from the live
numpy oracle (R-CHAR-3: never copied from the ferray side).
"""

import numpy as np
import pytest

import ferray as fr


# The truth set spans: ASCII, superscripts/subscripts, circled,
# parenthesized, fullwidth, Arabic-Indic decimal (all Digit/Decimal -> True),
# fractions and roman numerals (Numeric, not Digit -> False), letters,
# empty, and mixed strings (-> False).
_TRUTH_SET = [
    "0",          # ascii digit          -> True
    "9",
    "123",        # ascii run            -> True
    "²³",  # superscript 2 3   -> True (Digit, not Decimal)
    "⁴",     # superscript 4        -> True
    "₀",     # subscript 0          -> True
    "①",     # circled 1            -> True
    "⓪",     # circled 0            -> True
    "⑴",     # parenthesized 1      -> True
    "０９",  # fullwidth 0 9     -> True (Decimal)
    "٣",     # arabic-indic 3       -> True (Decimal)
    "½",     # fraction 1/2         -> False (Numeric, not Digit)
    "⅓",     # fraction 1/3         -> False
    "Ⅻ",     # roman numeral XII    -> False (Numeric, not Digit)
    "abc",        # letters              -> False
    "12a",        # mixed                -> False
    "12.3",       # has a dot            -> False
    "",           # empty                -> False
]


@pytest.mark.parametrize("s", _TRUTH_SET)
def test_char_isdigit_matches_numpy_per_element(s):
    arr = np.array([s])
    fr_res = np.asarray(fr.char.isdigit(arr))
    np_res = np.asarray(np.char.isdigit(arr))
    assert fr_res.shape == np_res.shape, (fr_res.shape, np_res.shape)
    assert fr_res.dtype.kind == np_res.dtype.kind == "b", (
        fr_res.dtype,
        np_res.dtype,
    )
    np.testing.assert_array_equal(
        fr_res, np_res, err_msg=f"isdigit divergence for {s!r}"
    )


def test_char_isdigit_matches_numpy_whole_array():
    arr = np.array(_TRUTH_SET)
    fr_res = np.asarray(fr.char.isdigit(arr))
    np_res = np.asarray(np.char.isdigit(arr))
    np.testing.assert_array_equal(fr_res, np_res)


def test_char_isdigit_does_not_regress_isdecimal_isnumeric():
    """isdigit changing must not shift the sibling predicates: superscript
    is digit-but-not-decimal; fraction is numeric-but-not-digit."""
    arr = np.array(["²", "½", "①", "Ⅻ", "0123456789"])
    np.testing.assert_array_equal(
        np.asarray(fr.char.isdecimal(arr)), np.asarray(np.char.isdecimal(arr))
    )
    np.testing.assert_array_equal(
        np.asarray(fr.char.isnumeric(arr)), np.asarray(np.char.isnumeric(arr))
    )
    np.testing.assert_array_equal(
        np.asarray(fr.char.isdigit(arr)), np.asarray(np.char.isdigit(arr))
    )
