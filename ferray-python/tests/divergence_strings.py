"""Adversarial divergence pins for ``ferray.strings`` vs the NumPy oracle.

Audit target: the ``ferray.strings.*`` function surface registered by
``ferray-python/src/strings.rs`` (which re-registers the shared
``#[pyfunction]`` set from ``ferray-python/src/char.rs`` under the NumPy 2.0
canonical ``numpy.strings`` namespace). The underlying kernels live in the
``ferray-strings`` crate (``ferray-strings/src/classify.rs``).

Each test pins a CONFIRMED divergence between ``ferray.strings.*`` and the
installed ``numpy.strings.*`` (numpy 2.4.4, CPython 3.13 / Unicode 15.1.0 =
live oracle). The numpy result is computed LIVE in every assertion
(R-CHAR-3 — no literal-copied expected values). Tests are EXPECTED TO FAIL
until the binding / underlying ``ferray-strings`` crate is fixed.

Root cause (both pins): ``ferray-strings/src/classify.rs:2414`` implements
``isnumeric`` via Rust ``char::is_numeric`` (the ``Nd|Nl|No`` general
categories tracked against a *newer* Unicode revision than 15.1.0), instead
of an oracle-derived ``Numeric_Type = Decimal|Digit|Numeric`` table the way
the other is* predicates (isalpha/isdigit/isdecimal) already are. The
file's own top-of-module note claims ``char::is_numeric`` "coincides with
Numeric_Type Decimal|Digit|Numeric" — the sweeps below show it does NOT.
``isalnum`` (``classify.rs:2401``) composes ``c.is_numeric()`` and inherits
the same error.

Upstream: ``numpy/_core/strings.py`` ``isnumeric`` / ``isalnum`` (lines 50 /
45 of ``__all__``) delegate per element to CPython ``str.isnumeric()`` /
``str.isalnum()`` (Unicode 15.1.0).

Run:  PYTHONPATH=python python3 -m pytest tests/divergence_strings.py -q
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. isnumeric: CJK ideographic numerals (Numeric_Type=Numeric) are MISSED.
#    U+4E00 CJK ONE 一 -> numpy True, ferray False.
#    Rust char::is_numeric (Nd|Nl|No) excludes CJK numerals, which are
#    general category Lo but carry Numeric_Type=Numeric, so CPython /
#    numpy report them numeric.
# ---------------------------------------------------------------------------


def test_isnumeric_cjk_ideographic_numeral():
    a = np.array(["一"], dtype="U1")  # 一 CJK ONE, Numeric_Type=Numeric
    expected = np.strings.isnumeric(a)  # live oracle -> True
    out = np.asarray(fr.strings.isnumeric(a))
    assert expected[0], "oracle precondition: numpy classifies U+4E00 numeric"
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 2. isnumeric: codepoints assigned AFTER Unicode 15.1.0 are WRONGLY numeric.
#    U+10D40 GARAY DIGIT ZERO (assigned Unicode 16.0) -> numpy False (its
#    15.1.0 table is unassigned), ferray True (Rust std tracks a newer
#    Unicode revision that has assigned it Nd). Reverse-direction error.
# ---------------------------------------------------------------------------


def test_isnumeric_post_15_1_codepoint_not_numeric():
    a = np.array(["\U00010d40"], dtype="U1")  # Garay digit zero, U16.0
    expected = np.strings.isnumeric(a)  # live oracle -> False on Unicode 15.1.0
    out = np.asarray(fr.strings.isnumeric(a))
    assert not expected[0], "oracle precondition: numpy 15.1.0 leaves U+10D40 unassigned"
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 3. isalnum inherits the isnumeric error: U+10D40 is alphanumeric in ferray
#    (because its isnumeric is wrongly True) but not in numpy.
#    isalnum = isalpha | isdecimal | isdigit | isnumeric (classify.rs:2401).
# ---------------------------------------------------------------------------


def test_isalnum_inherits_isnumeric_post_15_1_error():
    a = np.array(["\U00010d40"], dtype="U1")  # Garay digit zero, U16.0
    expected = np.strings.isalnum(a)  # live oracle -> False
    out = np.asarray(fr.strings.isalnum(a))
    assert not expected[0], "oracle precondition: numpy classifies U+10D40 non-alnum"
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 4. Full-BMP-and-beyond sweep: isnumeric must match the numpy oracle on
#    EVERY assigned codepoint. Pins the whole 184-codepoint divergence set,
#    not just the two representatives above. Expected mask is the live
#    oracle (R-CHAR-3).
# ---------------------------------------------------------------------------


def test_isnumeric_full_codepoint_sweep():
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    arr = np.array([chr(c) for c in cps], dtype="U1")
    expected = np.strings.isnumeric(arr)  # live oracle over all codepoints
    out = np.asarray(fr.strings.isnumeric(arr))
    mismatches = np.where(out != expected)[0]
    assert mismatches.size == 0, (
        f"{mismatches.size} isnumeric codepoint mismatches vs numpy oracle; "
        f"first: U+{cps[mismatches[0]]:04X} "
        f"numpy={bool(expected[mismatches[0]])} ferray={bool(out[mismatches[0]])}"
    )


# ---------------------------------------------------------------------------
# 5. Full sweep for isalnum (downstream of isnumeric).
# ---------------------------------------------------------------------------


def test_isalnum_full_codepoint_sweep():
    cps = [c for c in range(0x110000) if not (0xD800 <= c <= 0xDFFF)]
    arr = np.array([chr(c) for c in cps], dtype="U1")
    expected = np.strings.isalnum(arr)  # live oracle
    out = np.asarray(fr.strings.isalnum(arr))
    mismatches = np.where(out != expected)[0]
    assert mismatches.size == 0, (
        f"{mismatches.size} isalnum codepoint mismatches vs numpy oracle; "
        f"first: U+{cps[mismatches[0]]:04X} "
        f"numpy={bool(expected[mismatches[0]])} ferray={bool(out[mismatches[0]])}"
    )
