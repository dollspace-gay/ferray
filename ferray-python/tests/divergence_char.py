"""Adversarial divergence pins for ``ferray.char`` vs the NumPy oracle.

Each test pins a CONFIRMED divergence between ``ferray.char.*`` and
``numpy.char.* / numpy.strings.*`` (numpy 2.4.5, installed = live oracle).
The numpy result is computed LIVE in every assertion (R-CHAR-3 — no
literal-copied expected values). Tests assert value AND dtype AND shape,
or a matching exception type, and are EXPECTED TO FAIL until the binding
(`ferray-python/src/char.rs`) / underlying `ferray-strings` crate is fixed.

Target under audit: ferray-python/src/char.rs.
Upstream citations point at /home/doll/numpy-ref/numpy/_core/strings.py and
/home/doll/numpy-ref/numpy/_core/defchararray.py.

Run:  PYTHONPATH=python python3 -m pytest tests/divergence_char.py -q
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. count result dtype: numpy returns intp/int64, ferray returns uint64
# ---------------------------------------------------------------------------


def test_count_returns_signed_int_dtype():
    """numpy/_core/strings.py:41 imports `count as _count_ufunc`; its output
    dtype is the platform default integer (int64 on this host), NEVER
    unsigned. char.rs:129 builds `ArrayD::<u64>` -> uint64.
    """
    arr = np.array(["hello", "foo", "barbaz"])
    expected = np.char.count(arr, "o")  # live oracle
    out = fr.char.count(arr, "o")
    assert out.dtype == expected.dtype, (
        f"count dtype {out.dtype!r} != numpy {expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 2. str_len result dtype: numpy returns int64, ferray returns uint64
# ---------------------------------------------------------------------------


def test_str_len_returns_signed_int_dtype():
    """numpy/_core/strings.py:57 imports `str_len` ufunc; output dtype is
    int64. char.rs:177 builds `ArrayD::<u64>` -> uint64.
    """
    arr = np.array(["a", "ab", "abc"])
    expected = np.char.str_len(arr)  # live oracle
    out = fr.char.str_len(arr)
    assert out.dtype == expected.dtype, (
        f"str_len dtype {out.dtype!r} != numpy {expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 3. multiply by a negative count -> empty strings, not OverflowError
# ---------------------------------------------------------------------------


def test_multiply_negative_yields_empty_strings():
    """numpy/_core/strings.py: `multiply` clamps i to >= 0 (negative repeat
    counts yield the empty string), it does not raise. char.rs:224 binds the
    count as Rust `usize`, so PyO3 raises OverflowError on a negative int.
    """
    arr = np.array(["ab", "cd"])
    expected = np.char.multiply(arr, -2)  # live oracle: array(['',''])
    out = fr.char.multiply(arr, -2)
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 4. startswith / endswith honor the start/end range arguments
# ---------------------------------------------------------------------------


def test_startswith_with_start():
    """numpy/_core/defchararray startswith(a, prefix, start=0, end=None).
    char.rs:145 `startswith(py, a, prefix)` exposes no start/end -> TypeError.
    """
    arr = np.array(["hello", "help"])
    expected = np.char.startswith(arr, "ello", 1)  # live oracle
    out = fr.char.startswith(arr, "ello", start=1)
    np.testing.assert_array_equal(out, expected)


def test_endswith_with_end():
    """numpy endswith(a, suffix, start=0, end=None). char.rs:159 exposes no
    start/end -> TypeError on the `end` kwarg.
    """
    arr = np.array(["hello"])
    expected = np.char.endswith(arr, "ell", 0, 4)  # live oracle
    out = fr.char.endswith(arr, "ell", end=4)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 5. find / count honor the start/end range arguments
# ---------------------------------------------------------------------------


def test_find_with_start():
    """numpy find(a, sub, start=0, end=None): with start the search window
    shifts. char.rs:135 `find(py, a, sub)` takes no start -> TypeError.
    """
    arr = np.array(["hello", "help"])
    expected = np.char.find(arr, "l", 3)  # live oracle
    out = fr.char.find(arr, "l", start=3)
    np.testing.assert_array_equal(out, expected)


def test_count_with_start_end():
    """numpy count(a, sub, start=0, end=None). char.rs:121 `count(py, a, sub)`
    takes no start/end -> TypeError.
    """
    arr = np.array(["aAaAaA"])
    expected = np.char.count(arr, "A", 1, 4)  # live oracle
    out = fr.char.count(arr, "A", start=1, end=4)
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 6. comparison ops broadcast a scalar string against an array
# ---------------------------------------------------------------------------


def test_equal_broadcasts_scalar():
    """numpy.char.equal is a ufunc and broadcasts: a 0-d string vs a 1-d
    array -> per-element bool array. char.rs `equal` -> fs::equal requires
    identical shapes and raises ValueError on the mismatch.
    """
    a = np.array(["a", "b"])
    b = np.array("a")
    expected = np.char.equal(a, b)  # live oracle: array([True, False])
    out = fr.char.equal(a, b)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 7. add broadcasts a length-1 array against a longer array
# ---------------------------------------------------------------------------


def test_add_broadcasts():
    """numpy.char.add follows ufunc broadcasting: (2,) + (1,) -> (2,).
    char.rs:208 `add` -> fs::add_same requires identical shapes, raising
    ValueError on the (2,) vs (1,) mismatch.
    """
    a = np.array(["a", "b"])
    b = np.array(["x"])
    expected = np.char.add(a, b)  # live oracle: array(['ax', 'bx'])
    out = fr.char.add(a, b)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 8. empty string array preserves the unicode dtype, not float64
# ---------------------------------------------------------------------------


def test_upper_empty_array_preserves_string_dtype():
    """numpy upper(a) preserves the unicode dtype kind on an empty input
    (np.char.upper(empty <U5).dtype.kind == 'U'). char.rs:56
    `string_array_to_pyarray` round-trips an empty Python list through
    `numpy.asarray`, which defaults to dtype float64 (kind 'f').
    """
    arr = np.array([], dtype="<U5")
    expected = np.char.upper(arr)  # live oracle: kind 'U'
    out = fr.char.upper(arr)
    assert out.dtype.kind == expected.dtype.kind, (
        f"upper(empty) dtype {out.dtype!r} != numpy {expected.dtype!r}"
    )


# ---------------------------------------------------------------------------
# 9. 0-d scalar input returns a 0-d array, not a 1-d array
# ---------------------------------------------------------------------------


def test_upper_0d_returns_0d():
    """numpy preserves rank: np.char.upper(np.array('hi')).ndim == 0.
    char.rs:56 `string_array_to_pyarray` returns the flat list unreshaped
    (shape.len() <= 1 short-circuit), yielding a 1-d length-1 array.
    """
    arr = np.array("hi")
    expected = np.char.upper(arr)  # live oracle: 0-d
    out = fr.char.upper(arr)
    assert out.ndim == expected.ndim, (
        f"upper(0-d) ndim {out.ndim} != numpy {expected.ndim}"
    )
    assert out.shape == expected.shape


# ---------------------------------------------------------------------------
# 10. string->string ops preserve the INPUT itemsize (<U width)
# ---------------------------------------------------------------------------


def test_lower_preserves_input_itemsize():
    """numpy's case ops cannot grow a string, so the output dtype keeps the
    INPUT itemsize: np.char.lower(<U10) -> <U10. char.rs:56
    `string_array_to_pyarray` round-trips through `numpy.asarray(list)`,
    which picks the MINIMAL <U width from the actual contents (<U2 here),
    silently shrinking the dtype across the PyO3 boundary (R-CODE-4).
    """
    arr = np.array(["hi", "x"], dtype="<U10")
    expected = np.char.lower(arr)  # live oracle: dtype <U10
    out = fr.char.lower(arr)
    assert out.dtype == expected.dtype, (
        f"lower itemsize {out.dtype!r} != numpy {expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)


def test_strip_preserves_input_itemsize():
    """numpy strip keeps the input itemsize even though stripping shrinks the
    content: np.char.strip(<U6 'xyhixy', chars='xy') -> dtype <U6 'hi'.
    char.rs strip path shrinks the dtype to <U2 via the list round-trip.
    """
    arr = np.array(["xyhixy", "xy"])  # dtype <U6
    expected = np.char.strip(arr, chars="xy")  # live oracle: dtype <U6
    out = fr.char.strip(arr, chars="xy")
    assert out.dtype == expected.dtype, (
        f"strip itemsize {out.dtype!r} != numpy {expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)
