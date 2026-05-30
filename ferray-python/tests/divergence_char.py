"""Adversarial divergence tests for ferray.char vs the NumPy oracle.

Each test pins a CONFIRMED divergence between ``ferray.char.*`` and
``numpy.char.* / numpy.strings.*`` (numpy 2.4.x). The numpy result is
computed LIVE as the oracle. Tests assert value + dtype + shape, or a
matching exception type, and are expected to FAIL until the binding /
underlying ferray-strings crate is fixed.

Upstream citations point at /home/doll/numpy-ref/numpy/_core/strings.py.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. Result dtype: count returns intp (int64), not uint64
# ---------------------------------------------------------------------------


def test_count_returns_signed_int_dtype():
    """numpy/_core/strings.py:405 `def count(...)` returns an int array
    (``_count_ufunc`` output dtype is the platform ``intp`` == int64),
    NEVER unsigned. ferray.char.count returns uint64.
    """
    arr = np.array(["hello", "foo", "barbaz"])
    out = fr.char.count(arr, "o")
    expected = np.char.count(arr, "o")
    assert out.dtype == expected.dtype, (
        f"count dtype {out.dtype!r} != numpy {expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 2. Result dtype: str_len returns intp (int64), not uint64
# ---------------------------------------------------------------------------


def test_str_len_returns_signed_int_dtype():
    """numpy str_len (numpy/_core/strings.py, `str_len` ufunc) returns an
    int64 array. ferray.char.str_len returns uint64.
    """
    arr = np.array(["a", "ab", "abc"])
    out = fr.char.str_len(arr)
    expected = np.char.str_len(arr)
    assert out.dtype == expected.dtype, (
        f"str_len dtype {out.dtype!r} != numpy {expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 3. multiply by a negative count -> empty strings, not OverflowError
# ---------------------------------------------------------------------------


def test_multiply_negative_yields_empty_strings():
    """numpy/_core/strings.py:155 `Values in i of less than 0 are treated
    as 0 (which yields an empty string).` and :195 `i = np.maximum(i, 0)`.
    ferray.char.multiply(a, -2) raises OverflowError instead.
    """
    arr = np.array(["ab", "cd"])
    expected = np.char.multiply(arr, -2)
    out = fr.char.multiply(arr, -2)
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 4. startswith honors the start/end range arguments
# ---------------------------------------------------------------------------


def test_startswith_with_start():
    """numpy/_core/strings.py:450 `def startswith(a, prefix, start=0, end=None)`
    and :482 example `np.strings.startswith(s, 'o', start=1, end=2)`.
    ferray.char.startswith accepts no start/end -> TypeError.
    """
    arr = np.array(["hello", "help"])
    expected = np.char.startswith(arr, "ello", 1)
    out = fr.char.startswith(arr, "ello", start=1)
    np.testing.assert_array_equal(out, expected)


def test_endswith_with_end():
    """numpy/_core/strings.py:491 `def endswith(a, suffix, start=0, end=None)`
    and :523 example `np.strings.endswith(s, 'a', start=1, end=2)`.
    ferray.char.endswith accepts no start/end -> TypeError.
    """
    arr = np.array(["hello"])
    expected = np.char.endswith(arr, "ell", 0, 4)
    out = fr.char.endswith(arr, "ell", end=4)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 5. find / count honor the start/end range arguments
# ---------------------------------------------------------------------------


def test_find_with_start():
    """numpy/_core/strings.py:256 `def find(a, sub, start=0, end=None)`.
    With start, the search window shifts. ferray.char.find takes no start.
    """
    arr = np.array(["hello", "help"])
    expected = np.char.find(arr, "l", 3)
    out = fr.char.find(arr, "l", start=3)
    np.testing.assert_array_equal(out, expected)


def test_count_with_start_end():
    """numpy/_core/strings.py:405 `def count(a, sub, start=0, end=None)`
    and :439 example `np.strings.count(c, 'A', start=1, end=4)`.
    ferray.char.count takes no start/end.
    """
    arr = np.array(["aAaAaA"])
    expected = np.char.count(arr, "A", 1, 4)
    out = fr.char.count(arr, "A", start=1, end=4)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 6. comparison ops broadcast a scalar string against an array
# ---------------------------------------------------------------------------


def test_equal_broadcasts_scalar():
    """numpy.char.equal broadcasts like any ufunc: a 0-d string vs a 1-d
    array produces a per-element bool array. ferray.char.equal raises
    ValueError on the shape mismatch instead of broadcasting.
    """
    a = np.array(["a", "b"])
    b = np.array("a")
    expected = np.char.equal(a, b)
    out = fr.char.equal(a, b)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 7. add broadcasts a length-1 array against a longer array
# ---------------------------------------------------------------------------


def test_add_broadcasts():
    """numpy.char.add follows ufunc broadcasting: shape (2,) + shape (1,)
    -> (2,). ferray.char.add (fs::add_same) requires identical shapes and
    raises ValueError.
    """
    a = np.array(["a", "b"])
    b = np.array(["x"])
    expected = np.char.add(a, b)
    out = fr.char.add(a, b)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 8. empty string array preserves the unicode dtype, not float64
# ---------------------------------------------------------------------------


def test_upper_empty_array_preserves_string_dtype():
    """numpy/_core/strings.py:1086 `def upper(a)` preserves the unicode
    dtype on an empty input (np.char.upper(empty <U5) -> <U5). ferray
    returns dtype float64 (the empty Python list -> numpy.asarray default).
    """
    arr = np.array([], dtype="<U5")
    expected = np.char.upper(arr)
    out = fr.char.upper(arr)
    assert out.dtype.kind == expected.dtype.kind, (
        f"upper(empty) dtype {out.dtype!r} != numpy {expected.dtype!r}"
    )


# ---------------------------------------------------------------------------
# 9. 0-d scalar input returns a 0-d array, not a 1-d array
# ---------------------------------------------------------------------------


def test_upper_0d_returns_0d():
    """numpy preserves rank: np.char.upper(np.array('hi')) is a 0-d array
    (ndim == 0). ferray.char.upper returns a 1-d array of length 1.
    """
    arr = np.array("hi")
    expected = np.char.upper(arr)
    out = fr.char.upper(arr)
    assert out.ndim == expected.ndim, (
        f"upper(0-d) ndim {out.ndim} != numpy {expected.ndim}"
    )
    assert out.shape == expected.shape
