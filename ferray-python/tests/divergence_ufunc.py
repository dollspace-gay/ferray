"""Adversarial divergence pins for the ferray ufunc surface.

Each test computes the numpy oracle value LIVE and asserts ferray matches
(value + dtype + shape, or matching exception type). Every test is expected
to FAIL against current ferray (these are pinned divergences, not regressions).

Upstream citations refer to the numpy working tree at /home/doll/numpy-ref.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# DIV-1: np.divide uses TrueDivisionTypeResolver — integer inputs promote to
# float64 and the quotient is a true (non-floor) division.
#
# numpy cite: numpy/_core/code_generators/generate_umath.py:422
#   "          'PyUFunc_TrueDivisionTypeResolver',"
# and ufunc_docstrings.py:1106-1107 "The ``true_divide(x1, x2)`` function is
# an alias for ``divide(x1, x2)``."
# Expected: np.divide([1,2,3],[2,2,2]) -> [0.5, 1.0, 1.5] dtype float64.
# Actual (ferray): integer floor-style result [0,1,1] dtype int64.
# ---------------------------------------------------------------------------
def test_divide_int_promotes_to_float_true_division():
    a = np.array([1, 2, 3])
    b = np.array([2, 2, 2])
    expected = np.divide(a, b)
    got = fr.divide(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# PROMOTE-1: mixed int + float promotes to float64 (the float operand's
# fractional part must survive). ferray coerces x2 to x1's int dtype.
#
# numpy cite: numpy/_core/code_generators/ufunc_docstrings.py:3129
#   add_newdoc('numpy._core.umath', 'multiply' ...) — TD(no_bool_times_obj)
#   in generate_umath.py:392 covers fdFD + ints, with the result type
#   following numpy's common-type promotion (int + float -> float64).
# Expected: np.add([1,2],[1.5,2.5]) -> [2.5, 4.5] dtype float64.
# Actual (ferray): [2, 4] dtype int64 (second operand truncated to int).
# ---------------------------------------------------------------------------
def test_add_mixed_int_float_promotes_to_float64():
    a = np.array([1, 2])
    b = np.array([1.5, 2.5])
    expected = np.add(a, b)
    got = fr.add(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# CLIP-1 (seed): np.clip supports integer arrays.
#
# numpy cite: numpy/_core/fromnumeric.py:2208
#   "def clip(a, a_min=np._NoValue, a_max=np._NoValue, out=None, *, ..."
#   clip is "Equivalent to ... np.minimum(a_max, np.maximum(a, a_min))"
#   (line 2218) — and maximum/minimum support ints.
# Expected: np.clip([1,5,10], 2, 8) -> [2,5,8] dtype int64.
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
def test_clip_int_array():
    src = np.array([1, 5, 10])
    expected = np.clip(src, 2, 8)
    got = fr.clip(src, 2, 8)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# CLIP-2: np.clip accepts array-valued a_min / a_max bounds (broadcast).
#
# numpy cite: numpy/_core/fromnumeric.py:2226
#   "    a_min, a_max : array_like or None"
# Expected: np.clip([1.,5.,10.],[2.,2.,2.],[8.,8.,8.]) -> [2.,5.,8.].
# Actual (ferray): TypeError "argument 'a_min': only 0-dimensional arrays
#   can be converted to Python scalars" (a_min typed as f64).
# ---------------------------------------------------------------------------
def test_clip_array_bounds():
    src = np.array([1.0, 5.0, 10.0])
    lo = np.array([2.0, 2.0, 2.0])
    hi = np.array([8.0, 8.0, 8.0])
    expected = np.clip(src, lo, hi)
    got = fr.clip(src, lo, hi)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# CLIP-3: np.clip accepts None for a missing bound (clip one-sided).
#
# numpy cite: numpy/_core/fromnumeric.py:2228
#   "the corresponding edge. If both ``a_min`` and ``a_max`` are ``None``,"
# Expected: np.clip([1.,5.,10.], None, 8.0) -> [1.,5.,8.].
# Actual (ferray): TypeError "argument 'a_min': must be real number, not
#   NoneType" (a_min typed as non-optional f64).
# ---------------------------------------------------------------------------
def test_clip_none_lower_bound():
    src = np.array([1.0, 5.0, 10.0])
    expected = np.clip(src, None, 8.0)
    got = fr.clip(src, None, 8.0)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# MAXMIN-INT: maximum/minimum support integer dtypes, returning that int dtype.
#
# numpy cite: numpy/_core/code_generators/generate_umath.py:661 'maximum'
#   and docstrings.py:2490 — element-wise max over array_like (ints included).
# Expected: np.maximum([1,5],[3,2]) -> [3,5] dtype int64.
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["maximum", "minimum"])
def test_maximum_minimum_int(name):
    a = np.array([1, 5])
    b = np.array([3, 2])
    expected = getattr(np, name)(a, b)
    got = getattr(fr, name)(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# POWER-INT: power supports integer dtypes.
#
# numpy cite: numpy/_core/code_generators/generate_umath.py:484
#   "          TD(ints)," — the first registered loop for 'power'.
# Expected: np.power([2,3],[2,3]) -> [4,27] dtype int64.
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
def test_power_int():
    a = np.array([2, 3])
    b = np.array([2, 3])
    expected = np.power(a, b)
    got = fr.power(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# UNARY-INT: absolute / negative / sign / floor / ceil / trunc / rint accept
# integer inputs (numpy preserves int dtype for the algebraic ones and returns
# the same int dtype for floor/ceil/trunc/rint).
#
# numpy cite: numpy/_core/code_generators/generate_umath.py:500 'absolute'
#   "TD(bints + flts + timedeltaonly ...)"; :520 'negative' "TD(ints + flts
#   ...)"; :538 'sign' "TD(ints + flts ...)"; floor at :1011.
# Expected: np.negative([1,-2,3]) -> [-1,2,-3] int64; np.absolute([-1,-2,3])
#   -> [1,2,3] int64; np.sign([-5,0,7]) -> [-1,0,1] int64; np.floor([1,2,3])
#   -> [1,2,3] int64.
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, src",
    [
        ("negative", [1, -2, 3]),
        ("absolute", [-1, -2, 3]),
        ("sign", [-5, 0, 7]),
        ("floor", [1, 2, 3]),
        ("ceil", [1, 2, 3]),
        ("trunc", [1, 2, 3]),
    ],
)
def test_unary_int_supported(name, src):
    a = np.array(src)
    expected = getattr(np, name)(a)
    got = getattr(fr, name)(a)
    assert got.dtype == expected.dtype, (name, got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# SCALAR-RET: a ufunc on scalar / 0-d inputs returns a numpy scalar, not a
# 0-d ndarray (the $OUT_SCALAR contract).
#
# numpy cite: numpy/_core/code_generators/ufunc_docstrings.py:1093-1095
#   "    y : ndarray or scalar\n ... $OUT_SCALAR_2"
#   and divide example 1112-1113: "np.divide(2.0, 4.0)" -> "0.5" (scalar).
# Expected: type(np.add(np.array(1.0), np.array(2.0))) is np.float64.
# Actual (ferray): returns a 0-d numpy.ndarray.
# ---------------------------------------------------------------------------
def test_binary_scalar_return_type():
    expected = np.add(np.array(1.0), np.array(2.0))
    got = fr.add(np.array(1.0), np.array(2.0))
    assert np.isscalar(got) or isinstance(got, np.floating), type(got)
    assert type(got) is type(expected), (type(got), type(expected))


# ---------------------------------------------------------------------------
# MISSING-FN: top-level ufunc aliases/functions present in numpy but absent
# from ferray.
#
# numpy cite: generate_umath.py:405 'floor_divide', :419 'divide' (true_divide
#   aliased per comment :404), :490 'float_power', :1039 'remainder' (mod is
#   an alias of remainder).
# Expected: fr exposes floor_divide / true_divide / mod / remainder /
#   float_power with numpy-matching results.
# Actual (ferray): AttributeError — not registered.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["true_divide", "floor_divide", "mod", "remainder", "float_power"])
def test_missing_top_level_function(name):
    assert hasattr(fr, name), f"ferray is missing top-level {name}"


def test_true_divide_value():
    a = np.array([1, 2])
    b = np.array([2, 2])
    expected = np.true_divide(a, b)
    got = fr.true_divide(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


def test_floor_divide_value():
    a = np.array([7, 8])
    b = np.array([2, 3])
    expected = np.floor_divide(a, b)
    got = fr.floor_divide(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)
