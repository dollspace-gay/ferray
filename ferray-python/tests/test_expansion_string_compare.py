"""String (``<U`` unicode / ``<S`` bytes) comparison for the main-array surface
(REQ-3, blocker #961, epic #958 — the LAST string blocker).

Each comparison op (``fr.less``/``greater``/``less_equal``/``greater_equal``/
``equal``/``not_equal``) detects both operands as fixed-width string operands
(dtype kind in ``{'U','S'}``, via ``is_string_compare``) ahead of its real-only
dtype-dispatch body and delegates to numpy's own ``np.<op>(x1, x2)`` on the
ORIGINAL operands, returning numpy's bool result directly (no transport —
strings never enter the Rust library, .design/ferray-core-string.md). numpy owns
the codepoint-lexicographic compare, the broadcasting (string array vs string
scalar), and width-independence (``<U2`` vs ``<U3``).

Every expected value is derived LIVE from numpy in-test (R-CHAR-3): each case
compares ``fr.<op>(...)`` against the corresponding ``np.<op>(...)`` call (values
+ bool dtype), never a literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr


# (fr op, np op) pairs — the six comparison ufuncs REQ-3 covers.
_OPS = [
    (fr.less, np.less),
    (fr.greater, np.greater),
    (fr.less_equal, np.less_equal),
    (fr.greater_equal, np.greater_equal),
    (fr.equal, np.equal),
    (fr.not_equal, np.not_equal),
]
_OP_IDS = ["less", "greater", "less_equal", "greater_equal", "equal", "not_equal"]


def _same(got, expected):
    """ferray result equals the numpy oracle in both values and dtype."""
    g = np.asarray(got)
    e = np.asarray(expected)
    assert np.array_equal(g, e), f"values: {g!r} != {e!r}"
    assert g.dtype == e.dtype, f"dtype: {g.dtype} != {e.dtype}"


def _U(*items):
    return np.array(list(items))


def _S(*items):
    return np.array([s.encode() for s in items])


# ---------------------------------------------------------------------------
# Core: each op on two <U arrays → bool, lexicographic by codepoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_unicode_array_vs_array(fr_op, np_op):
    a = _U("abc", "xyz", "abc")
    b = _U("abd", "abc", "abc")
    _same(fr_op(a, b), np_op(a, b))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_unicode_result_is_bool(fr_op, np_op):
    a = _U("a", "b", "c")
    b = _U("b", "b", "a")
    out = np.asarray(fr_op(a, b))
    assert out.dtype == np.bool_
    _same(out, np_op(a, b))


# Mirror the goal.md live probe verbatim (R-CHAR-3):
#   a=['abc','xyz','abc']; b=['abd','abc','abc']
#   a<b a>b a==b a!=b a<='abc'
def test_goal_probe_matches_numpy():
    a = _U("abc", "xyz", "abc")
    b = _U("abd", "abc", "abc")
    _same(fr.less(a, b), np.less(a, b))
    _same(fr.greater(a, b), np.greater(a, b))
    _same(fr.equal(a, b), np.equal(a, b))
    _same(fr.not_equal(a, b), np.not_equal(a, b))
    _same(fr.less_equal(a, "abc"), np.less_equal(a, "abc"))


# ---------------------------------------------------------------------------
# Broadcasting: string array vs string scalar
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_broadcast_array_vs_str_scalar(fr_op, np_op):
    a = _U("a", "b", "c")
    _same(fr_op(a, "b"), np_op(a, "b"))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_broadcast_str_scalar_vs_array(fr_op, np_op):
    a = _U("a", "b", "c")
    _same(fr_op("b", a), np_op("b", a))


def test_equal_broadcast_scalar_matches_numpy():
    # AC-3 explicit case.
    a = _U("a", "b")
    _same(fr.equal(a, "a"), np.equal(a, "a"))


# ---------------------------------------------------------------------------
# Different widths: <U2 vs <U3 (compare by value, width-independent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_different_unicode_widths(fr_op, np_op):
    a = _U("ab", "cd")  # <U2
    b = _U("abc", "cd")  # <U3
    assert a.dtype != b.dtype
    _same(fr_op(a, b), np_op(a, b))


# ---------------------------------------------------------------------------
# Bytes <S arrays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_bytes_array_vs_array(fr_op, np_op):
    a = _S("a", "z", "m")
    b = _S("b", "a", "m")
    _same(fr_op(a, b), np_op(a, b))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_bytes_broadcast_scalar(fr_op, np_op):
    a = _S("a", "b", "c")
    _same(fr_op(a, b"b"), np_op(a, b"b"))


# ---------------------------------------------------------------------------
# Edge cases: empty strings, non-ASCII, 2-D, empty array
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_empty_strings(fr_op, np_op):
    a = _U("", "a", "")
    b = _U("", "b", "x")
    _same(fr_op(a, b), np_op(a, b))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_non_ascii_unicode(fr_op, np_op):
    a = _U("é", "a", "ω")
    b = _U("z", "b", "ω")
    _same(fr_op(a, b), np_op(a, b))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_two_dimensional(fr_op, np_op):
    a = np.array([["ab", "cd"], ["ef", "gh"]])
    b = np.array([["ab", "ce"], ["ef", "gg"]])
    _same(fr_op(a, b), np_op(a, b))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_empty_string_array(fr_op, np_op):
    a = np.array([], dtype="U3")
    b = np.array([], dtype="U3")
    _same(fr_op(a, b), np_op(a, b))


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_scalar_str_vs_scalar_str(fr_op, np_op):
    # both 0-d / Python str scalars — np.asarray('a').dtype.kind == 'U'
    _same(fr_op("abc", "abd"), np_op("abc", "abd"))


# ---------------------------------------------------------------------------
# Mixed-kind U-vs-S: numpy raises UFuncTypeError; delegation must match numpy.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fr_op,np_op", _OPS, ids=_OP_IDS)
def test_mixed_unicode_vs_bytes_matches_numpy(fr_op, np_op):
    a = _U("a", "b")
    b = _S("a", "b")
    # numpy has no <U-vs-<S compare loop → UFuncTypeError. ferray delegates the
    # same call, so it must raise the SAME error type (match numpy, not ferray's
    # old `unsupported dtype`). Confirm numpy raises, then ferray raises too.
    with pytest.raises(Exception) as np_exc:
        np_op(a, b)
    with pytest.raises(type(np_exc.value)):
        fr_op(a, b)


# ---------------------------------------------------------------------------
# Real / numeric / datetime comparison paths UNREGRESSED
# ---------------------------------------------------------------------------


def test_numeric_compare_unregressed():
    a = np.array([1, 2, 3])
    b = np.array([3, 2, 1])
    _same(fr.less(a, b), np.less(a, b))
    _same(fr.equal(a, b), np.equal(a, b))
    _same(fr.greater_equal(a, b), np.greater_equal(a, b))


def test_float_compare_unregressed():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 1.0, 2.0])
    _same(fr.less(a, b), np.less(a, b))
    _same(fr.not_equal(a, b), np.not_equal(a, b))


def test_datetime_compare_unregressed():
    a = np.array(["2020-01-01", "2021-06-01"], dtype="datetime64[D]")
    b = np.array(["2020-06-01", "2021-06-01"], dtype="datetime64[D]")
    _same(fr.less(a, b), np.less(a, b))
    _same(fr.equal(a, b), np.equal(a, b))
