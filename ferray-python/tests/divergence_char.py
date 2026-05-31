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


# ---------------------------------------------------------------------------
# 11. char.split honors maxsplit (binding currently ignores it)
# ---------------------------------------------------------------------------


def test_split_honors_maxsplit():
    """numpy/_core/defchararray.py:1081 `split(self, sep, maxsplit)` ->
    strings.py:1441 `_vec_string(a, object_, 'split', [sep] + maxsplit)`,
    i.e. CPython `str.split(sep, maxsplit)` semantics: at most `maxsplit`
    splits are performed and the remainder stays joined.
    char.rs:1026 `let _ = maxsplit;` discards the argument, so ferray always
    performs a full split.

    Tracking: #1032 (-l blocker).
    """
    arr = np.array(["a,b,c,d"])
    expected = np.char.split(arr, ",", maxsplit=2)  # live: [['a','b','c,d']]
    out = fr.char.split(arr, ",", maxsplit=2)
    assert out[0] == expected[0], (
        f"split maxsplit ferray={out[0]} != numpy={expected[0]}"
    )


# ---------------------------------------------------------------------------
# 12. multiply by an integer ARRAY broadcasts element-wise
# ---------------------------------------------------------------------------


def test_multiply_by_integer_array():
    """numpy/_core/strings.py:150 `multiply(a, i)`: `i` is `array_like, with
    any integer dtype` and broadcasts element-wise. The docstring example
    (strings.py:177-180) shows `i = np.array([1,2,3])` -> ['a','bb','ccc'].
    char.rs:442 routes `n` through `coerce_multiply_count`, which only accepts
    a scalar integer and raises TypeError on an integer ndarray.

    Tracking: #1033 (-l blocker).
    """
    arr = np.array(["a", "b", "c"])
    i = np.array([1, 2, 3])
    expected = np.char.multiply(arr, i)  # live: ['a','bb','ccc'], <U3
    out = fr.char.multiply(arr, i)
    assert out.dtype == expected.dtype
    np.testing.assert_array_equal(out, expected)


def test_multiply_by_integer_array_2d():
    """Same as above for a 2-D operand broadcasting against a (3,) integer
    array (strings.py:182-184 docstring example). ferray rejects the integer
    ndarray operand entirely.

    Tracking: #1033 (-l blocker).
    """
    arr = np.array(["a", "b", "c", "d", "e", "f"]).reshape(2, 3)
    i = np.array([1, 2, 3])
    expected = np.char.multiply(arr, i)  # live oracle
    out = fr.char.multiply(arr, i)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 13. center: odd padding goes on the RIGHT (str.center semantics)
# ---------------------------------------------------------------------------


def test_center_odd_padding_on_right():
    """numpy/_core/strings.py:693 `center` delegates to the `_center` ufunc,
    which mirrors CPython `str.center`: when total padding is odd the EXTRA
    pad char goes on the RIGHT. `'ab'.center(5)` == '  ab ' (2 left, 1 right).
    ferray produces ' ab  ' (1 left, 2 right) — the asymmetric padding is on
    the wrong side.

    Tracking: #1034 (-l blocker).
    """
    arr = np.array(["ab"])
    expected = np.char.center(arr, 5)  # live oracle: '  ab '
    out = fr.char.center(arr, 5)
    assert out[0] == expected[0], (
        f"center ferray={out[0]!r} != numpy={expected[0]!r}"
    )


def test_center_odd_padding_with_fillchar():
    """`'ab'.center(7,'*')` == '***ab**' (3 left, 2 right). ferray puts the
    extra fill on the left ('**ab***'). strings.py:693 / str.center.

    Tracking: #1034 (-l blocker).
    """
    arr = np.array(["ab"])
    expected = np.char.center(arr, 7, "*")  # live oracle: '***ab**'
    out = fr.char.center(arr, 7, "*")
    assert out[0] == expected[0], (
        f"center ferray={out[0]!r} != numpy={expected[0]!r}"
    )


def test_center_0d_odd_padding():
    """0-d scalar input, odd padding: numpy '  ab ' vs ferray ' ab  '.
    Pins the padding direction independent of itemsize/rank handling.

    Tracking: #1034 (-l blocker).
    """
    arr = np.array("ab")
    expected = np.char.center(arr, 5)  # live oracle: '  ab '
    out = fr.char.center(arr, 5)
    assert str(out) == str(expected), (
        f"center 0d ferray={out!r} != numpy={expected!r}"
    )


# ---------------------------------------------------------------------------
# 14. expandtabs reserves the worst-case itemsize (<U width)
# ---------------------------------------------------------------------------


def test_expandtabs_reserves_itemsize():
    """numpy/_core/strings.py `expandtabs` allocates the output dtype from the
    worst-case tab expansion width, so np.char.expandtabs(<U3 'a\\tb', 4) keeps
    dtype <U11 even though the content is 'a   b' (5 chars). char.rs:727 lets
    the list round-trip through numpy.asarray, shrinking the dtype to the
    minimal content width <U5.

    Tracking: #1035 (-l blocker).
    """
    arr = np.array(["a\tb"])  # dtype <U3
    expected = np.char.expandtabs(arr, 4)  # live oracle: dtype <U11
    out = fr.char.expandtabs(arr, 4)
    assert out.dtype == expected.dtype, (
        f"expandtabs itemsize ferray={out.dtype!r} != numpy={expected.dtype!r}"
    )
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# 15. strings.partition: the empty `after` field has <U0 itemsize, not <U1
# ---------------------------------------------------------------------------


def test_partition_empty_after_field_is_U0():
    """When the separator is absent, numpy.strings.partition returns the
    3-tuple (a, '', '') where the empty trailing `after` field has dtype <U0
    (zero itemsize), since its width is derived from the (absent) match. The
    live oracle: np.strings.partition(['abc'], 'X')[2].dtype == '<U0'.
    char.rs `partition_tuple` -> `string_array_to_pyarray(.., force_width=None)`
    falls back to a <U1 minimum on the empty field, so ferray reports <U1.

    Tracking: #1036 (-l blocker).
    """
    arr = np.array(["abc"])
    _before, _sep, after = np.strings.partition(arr, "X")  # live oracle
    _fb, _fsep, faf = fr.strings.partition(arr, "X")
    assert faf.dtype == after.dtype, (
        f"partition after dtype ferray={faf.dtype!r} != numpy={after.dtype!r}"
    )
    np.testing.assert_array_equal(faf, after)
