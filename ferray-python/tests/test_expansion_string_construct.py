"""String (``<U`` unicode / ``<S`` bytes) input coercion for the main-array
construction surface (REQ-1, blocker #959, epic #958).

``fr.array``/``asarray``/``zeros``/``ones``/``empty``/``full``/``*_like`` accept
fixed-width string dtypes (``dtype='U3'``/``'S4'``) and string-list inputs
(``fr.array(['ab','cd'])`` -> numpy ``<U2``), delegating construction to numpy
(which owns the U/S width inference + parse + fill) and returning the numpy
string ndarray unchanged. Strings never enter the Rust library — they ride the
binding boundary as numpy ndarrays (.design/ferray-core-string.md).

Every expected value is derived LIVE from numpy in-test (R-CHAR-3): each case
compares ``fr.<fn>(...)`` against the corresponding ``np.<fn>(...)`` call (values
+ dtype), never a literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


def _same(got, expected):
    """ferray result equals the numpy oracle in both values and dtype."""
    g = np.asarray(got)
    assert np.array_equal(g, expected), f"values: {g!r} != {expected!r}"
    assert g.dtype == expected.dtype, f"dtype: {g.dtype} != {expected.dtype}"


# ---------------------------------------------------------------------------
# array / asarray — width inference + explicit width + bytes
# ---------------------------------------------------------------------------


def test_array_string_list_infers_u2():
    # numpy infers <U2 (width = longest element) — derived live.
    _same(fr.array(["ab", "cd"]), np.array(["ab", "cd"]))
    assert str(np.asarray(fr.array(["ab", "cd"])).dtype) == "<U2"


def test_array_string_list_infers_max_width():
    # mixed-length list -> width = the longest (<U3 for 'abc').
    _same(fr.array(["a", "abc"]), np.array(["a", "abc"]))


def test_array_explicit_u3_fixed_width():
    _same(fr.array(["ab"], dtype="U3"), np.array(["ab"], dtype="U3"))


def test_array_bytes_explicit_s4():
    _same(fr.array([b"ab"], dtype="S4"), np.array([b"ab"], dtype="S4"))


def test_array_bytes_list_infers_s_width():
    _same(fr.array([b"ab", b"cde"]), np.array([b"ab", b"cde"]))


def test_array_2d_string():
    _same(
        fr.array([["ab", "c"], ["d", "ef"]]),
        np.array([["ab", "c"], ["d", "ef"]]),
    )


def test_array_empty_list_explicit_u3():
    _same(fr.array([], dtype="U3"), np.array([], dtype="U3"))


def test_array_non_ascii_unicode():
    _same(fr.array(["héllo", "naïve"]), np.array(["héllo", "naïve"]))


def test_array_explicit_widen_u5():
    # explicit width wider than the data -> numpy keeps the requested width.
    _same(fr.array(["ab", "cd"], dtype="U5"), np.array(["ab", "cd"], dtype="U5"))


def test_asarray_preserves_numpy_string_dtype_and_width():
    src = np.array(["xy", "z"])  # <U2
    _same(fr.asarray(src), np.asarray(src))
    assert str(np.asarray(fr.asarray(src)).dtype) == "<U2"


def test_asarray_preserves_bytes_array():
    src = np.array([b"ab", b"cd"], dtype="S4")
    _same(fr.asarray(src), np.asarray(src))


def test_asarray_numpy_string_array_with_dtype_recast():
    src = np.array(["abcd", "ef"])  # <U4
    _same(fr.asarray(src, dtype="U2"), np.asarray(src, dtype="U2"))


# ---------------------------------------------------------------------------
# zeros / ones / empty — string dtype shape creation
# ---------------------------------------------------------------------------


def test_zeros_u5_is_empty_strings():
    _same(fr.zeros(3, dtype="U5"), np.zeros(3, dtype="U5"))


def test_zeros_s4_is_empty_bytes():
    _same(fr.zeros(2, dtype="S4"), np.zeros(2, dtype="S4"))


def test_ones_u5_fills_one_strings():
    # numpy fills '1' for ones on a string dtype — derived live.
    _same(fr.ones(2, dtype="U5"), np.ones(2, dtype="U5"))


def test_empty_u5_defined_buffer_is_empty_strings():
    # the binding's empty -> numpy zeros (defined buffer) contract.
    _same(fr.empty(2, dtype="U5"), np.zeros(2, dtype="U5"))


def test_zeros_2d_shape_string():
    _same(fr.zeros((2, 3), dtype="U4"), np.zeros((2, 3), dtype="U4"))


# ---------------------------------------------------------------------------
# full / full_like — fill semantics + width truncation
# ---------------------------------------------------------------------------


def test_full_u4_fill():
    _same(fr.full(2, "hi", dtype="U4"), np.full(2, "hi", dtype="U4"))


def test_full_string_fill_no_dtype_infers_width():
    # numpy infers <U2 from the fill 'hi' when no dtype is given.
    _same(fr.full(2, "hi"), np.full(2, "hi"))


def test_full_bytes_fill_no_dtype_infers_width():
    _same(fr.full(2, b"hi"), np.full(2, b"hi"))


def test_full_bytes_explicit_s4():
    _same(fr.full(2, b"hi", dtype="S4"), np.full(2, b"hi", dtype="S4"))


def test_full_fill_truncates_to_explicit_width():
    # numpy truncates a too-long fill to the requested width.
    _same(fr.full(3, "longfill", dtype="U4"), np.full(3, "longfill", dtype="U4"))


# ---------------------------------------------------------------------------
# *_like — preserve the source string dtype + width
# ---------------------------------------------------------------------------


def test_zeros_like_preserves_string_dtype():
    s = np.array(["ab", "cd"])
    _same(fr.zeros_like(s), np.zeros_like(s))


def test_ones_like_preserves_string_dtype():
    s = np.array(["ab", "cd"])
    _same(fr.ones_like(s), np.ones_like(s))


def test_empty_like_preserves_string_dtype():
    s = np.array(["ab", "cd"])
    _same(fr.empty_like(s), np.zeros_like(s))


def test_full_like_truncates_fill_to_source_width():
    s = np.array(["ab", "cd"])  # <U2
    _same(fr.full_like(s, "longfill"), np.full_like(s, "longfill"))


def test_full_like_explicit_dtype_overrides_width():
    s = np.array(["ab", "cd"])
    _same(fr.full_like(s, "longfill", dtype="U6"), np.full_like(s, "longfill", dtype="U6"))


def test_like_bytes_source_preserved():
    s = np.array([b"ab", b"cd"], dtype="S4")
    _same(fr.zeros_like(s), np.zeros_like(s))
    _same(fr.full_like(s, b"x"), np.full_like(s, b"x"))


# ---------------------------------------------------------------------------
# Real / numeric / datetime / float16 dtypes must be UNCHANGED by the branch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["float64", "int64", "float32", "uint8", "bool"])
def test_real_zeros_unregressed(dt):
    _same(fr.zeros(3, dtype=dt), np.zeros(3, dtype=dt))


def test_real_full_unregressed():
    _same(fr.full(2, 5), np.full(2, 5))
    _same(fr.full(2, 1.5, dtype="float64"), np.full(2, 1.5, dtype="float64"))


def test_real_array_unregressed():
    _same(fr.array([1, 2, 3]), np.array([1, 2, 3]))
    _same(fr.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_float16_unregressed():
    _same(fr.array([1.5, 2.5], dtype="float16"), np.array([1.5, 2.5], dtype="float16"))
    _same(fr.zeros(2, dtype="float16"), np.zeros(2, dtype="float16"))


def test_datetime_unregressed():
    exp = np.array(["2020-01-01"], dtype="datetime64[D]")
    _same(fr.array(["2020-01-01"], dtype="datetime64[D]"), exp)
    _same(fr.zeros(2, dtype="datetime64[s]"), np.zeros(2, dtype="datetime64[s]"))


def test_real_like_unregressed():
    a = np.array([1, 2, 3])
    _same(fr.zeros_like(a), np.zeros_like(a))
    _same(fr.full_like(a, 7), np.full_like(a, 7))
