"""Phase-2 parity tests for the ferray ufunc surface."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Unary float ufuncs — shape preservation + numeric agreement with numpy
# ---------------------------------------------------------------------------

UNARY_FLOAT_FUNCS = [
    ("sin", -np.pi, np.pi),
    ("cos", -np.pi, np.pi),
    ("tan", -1.0, 1.0),
    ("sinh", -2.0, 2.0),
    ("cosh", -2.0, 2.0),
    ("tanh", -2.0, 2.0),
    ("arcsin", -0.99, 0.99),
    ("arccos", -0.99, 0.99),
    ("arctan", -10.0, 10.0),
    ("arcsinh", -3.0, 3.0),
    ("arccosh", 1.01, 5.0),
    ("arctanh", -0.99, 0.99),
    ("exp", -3.0, 3.0),
    ("exp2", -3.0, 3.0),
    ("expm1", -3.0, 3.0),
    ("log", 0.01, 100.0),
    ("log1p", 0.0, 100.0),
    ("log2", 0.01, 100.0),
    ("log10", 0.01, 100.0),
    ("sqrt", 0.0, 100.0),
    ("cbrt", -27.0, 27.0),
    ("square", -10.0, 10.0),
    ("reciprocal", 0.5, 5.0),
    ("negative", -10.0, 10.0),
    ("positive", -10.0, 10.0),
    ("absolute", -10.0, 10.0),
    ("fabs", -10.0, 10.0),
    ("sign", -10.0, 10.0),
    ("floor", -5.7, 5.7),
    ("ceil", -5.7, 5.7),
    ("round", -5.7, 5.7),
    ("trunc", -5.7, 5.7),
    ("rint", -5.7, 5.7),
    ("degrees", 0.0, 2 * np.pi),
    ("radians", 0.0, 360.0),
    ("deg2rad", 0.0, 360.0),
    ("rad2deg", 0.0, 2 * np.pi),
]


@pytest.mark.parametrize("name, lo, hi", UNARY_FLOAT_FUNCS)
def test_unary_float_matches_numpy(name, lo, hi):
    rng = np.random.default_rng(seed=42)
    src = rng.uniform(lo, hi, size=(50,))
    fr_fn = getattr(ferray, name)
    np_fn = getattr(np, name)
    np.testing.assert_allclose(fr_fn(src), np_fn(src), rtol=1e-10, atol=1e-12)


def test_abs_alias_for_absolute():
    src = np.array([-1.0, 0.0, 1.0, -2.5])
    np.testing.assert_allclose(ferray.abs(src), np.abs(src))


def test_unary_float_dtype_preserved_f32():
    src = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = ferray.sin(src)
    assert out.dtype == np.float32


def test_unary_float_promotes_int_input():
    # numpy promotes integer input to float64 for transcendental ufuncs
    # (generate_umath.py: sin/exp/... register only float loops, int args
    # promote via the inexact common type). The binding now routes int
    # input through the promoted float path instead of rejecting it.
    src = np.array([1, 2, 3], dtype=np.int64)
    expected = np.sin(src)  # live numpy oracle (R-CHAR-3)
    got = ferray.sin(src)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# Predicates: float → bool
# ---------------------------------------------------------------------------


def test_isnan_basic():
    src = np.array([1.0, np.nan, 3.0, np.inf])
    np.testing.assert_array_equal(ferray.isnan(src), np.isnan(src))


def test_isinf_basic():
    src = np.array([1.0, np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(ferray.isinf(src), np.isinf(src))


def test_isfinite_basic():
    src = np.array([1.0, np.nan, np.inf, -np.inf, 0.0])
    np.testing.assert_array_equal(ferray.isfinite(src), np.isfinite(src))


def test_signbit_basic():
    src = np.array([-1.0, 0.0, 1.0, -0.0])
    np.testing.assert_array_equal(ferray.signbit(src), np.signbit(src))


# ---------------------------------------------------------------------------
# Binary numeric (broadcasting)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", ["add", "subtract", "multiply", "divide"])
def test_binary_numeric_matches_numpy(op_name):
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    fr_fn = getattr(ferray, op_name)
    np_fn = getattr(np, op_name)
    np.testing.assert_allclose(fr_fn(a, b), np_fn(a, b))


def test_add_broadcasts_2d_and_1d():
    a = np.arange(12).reshape(3, 4)
    b = np.arange(4)
    np.testing.assert_array_equal(ferray.add(a, b), np.add(a, b))


def test_multiply_int_dtype():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([4, 5, 6], dtype=np.int32)
    out = ferray.multiply(a, b)
    assert out.dtype == np.int32
    np.testing.assert_array_equal(out, np.multiply(a, b))


# ---------------------------------------------------------------------------
# Binary float
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", [
    "power", "maximum", "minimum", "fmax", "fmin", "copysign",
    "hypot", "arctan2", "logaddexp", "logaddexp2", "heaviside",
])
def test_binary_float_matches_numpy(op_name):
    rng = np.random.default_rng(123)
    a = rng.uniform(0.5, 5.0, size=(20,))
    b = rng.uniform(0.5, 5.0, size=(20,))
    fr_fn = getattr(ferray, op_name)
    np_fn = getattr(np, op_name)
    np.testing.assert_allclose(fr_fn(a, b), np_fn(a, b), rtol=1e-10)


def test_power_broadcasts():
    a = np.array([2.0, 3.0])
    b = np.array([[1.0], [2.0], [3.0]])
    np.testing.assert_allclose(ferray.power(a, b), np.power(a, b))


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op_name", [
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
])
def test_comparison_matches_numpy(op_name):
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 2.0, 2.0, 2.0])
    fr_fn = getattr(ferray, op_name)
    np_fn = getattr(np, op_name)
    np.testing.assert_array_equal(fr_fn(a, b), np_fn(a, b))


def test_equal_broadcasts():
    a = np.arange(6).reshape(2, 3)
    b = np.array([0, 1, 2])
    np.testing.assert_array_equal(ferray.equal(a, b), np.equal(a, b))


# ---------------------------------------------------------------------------
# Logical
# ---------------------------------------------------------------------------


def test_logical_and_matches_numpy():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(ferray.logical_and(a, b), np.logical_and(a, b))


def test_logical_or_matches_numpy():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(ferray.logical_or(a, b), np.logical_or(a, b))


def test_logical_xor_matches_numpy():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(ferray.logical_xor(a, b), np.logical_xor(a, b))


def test_logical_not_matches_numpy():
    src = np.array([True, False, True, False])
    np.testing.assert_array_equal(ferray.logical_not(src), np.logical_not(src))


def test_logical_and_on_int_truthy():
    a = np.array([0, 1, 2, 0])
    b = np.array([1, 0, 1, 1])
    np.testing.assert_array_equal(ferray.logical_and(a, b), np.logical_and(a, b))


# ---------------------------------------------------------------------------
# Clip
# ---------------------------------------------------------------------------


def test_clip_matches_numpy():
    src = np.array([-3.0, -1.0, 0.0, 1.0, 5.0])
    np.testing.assert_allclose(ferray.clip(src, -1.0, 2.0), np.clip(src, -1.0, 2.0))


# ---------------------------------------------------------------------------
# diff axis/prepend/append + ediff1d scalar appendage (#980)
# ---------------------------------------------------------------------------
# numpy.diff(a, n, axis=-1, prepend, append) supports N-D + edge values; the
# prior binding accepted only n=. numpy.ediff1d to_end/to_begin accept a SCALAR
# (numpy ravels it); the prior Vec<f64> signature rejected scalars.


def test_diff_axis0():
    np.testing.assert_array_equal(
        ferray.diff([[1, 3], [6, 10]], axis=0), np.diff([[1, 3], [6, 10]], axis=0)
    )


def test_diff_axis1():
    np.testing.assert_array_equal(
        ferray.diff([[1, 3], [6, 10]], axis=1), np.diff([[1, 3], [6, 10]], axis=1)
    )


def test_diff_prepend():
    np.testing.assert_array_equal(ferray.diff([1, 2, 4], prepend=0), np.diff([1, 2, 4], prepend=0))


def test_diff_append():
    np.testing.assert_array_equal(ferray.diff([1, 2, 4], append=7), np.diff([1, 2, 4], append=7))


def test_diff_1d_native_unchanged():
    np.testing.assert_array_equal(ferray.diff([1, 4, 9, 16], n=2), np.diff([1, 4, 9, 16], n=2))


def test_ediff1d_scalar_to_end():
    np.testing.assert_array_equal(
        ferray.ediff1d([1, 2, 4], to_end=99), np.ediff1d([1, 2, 4], to_end=99)
    )


def test_ediff1d_scalar_to_begin():
    np.testing.assert_array_equal(
        ferray.ediff1d([1, 2, 4], to_begin=-1), np.ediff1d([1, 2, 4], to_begin=-1)
    )


def test_ediff1d_both_appendages():
    np.testing.assert_array_equal(
        ferray.ediff1d([1, 2, 4], to_end=[9, 8], to_begin=0),
        np.ediff1d([1, 2, 4], to_end=[9, 8], to_begin=0),
    )


# ---------------------------------------------------------------------------
# unwrap period= / axis= (#983)
# ---------------------------------------------------------------------------
# numpy.unwrap(p, discont, axis=-1, period=2*pi). The prior binding lacked
# period=/axis=. An INTEGER input + INTEGER period keeps an int64 result.


def test_unwrap_period_int_keeps_int():
    r = ferray.unwrap([0, 5, 10], period=6)
    n = np.unwrap([0, 5, 10], period=6)
    assert np.asarray(r).dtype == n.dtype == np.int64
    np.testing.assert_array_equal(r, n)


def test_unwrap_period_float():
    np.testing.assert_allclose(
        ferray.unwrap([0.0, 5, 10], period=6.0), np.unwrap([0.0, 5, 10], period=6.0)
    )


def test_unwrap_axis():
    np.testing.assert_allclose(
        ferray.unwrap([[0.0, 5], [10, 15]], axis=0), np.unwrap([[0.0, 5], [10, 15]], axis=0)
    )


def test_unwrap_default_unchanged():
    np.testing.assert_allclose(ferray.unwrap([0.0, 5, 10]), np.unwrap([0.0, 5, 10]))


# ---------------------------------------------------------------------------
# interp left/right/period (#984)
# ---------------------------------------------------------------------------


def test_interp_left_right_fill():
    np.testing.assert_allclose(
        ferray.interp([0.0, 4], [1, 2, 3], [10, 20, 30], left=-1, right=99),
        np.interp([0.0, 4], [1, 2, 3], [10, 20, 30], left=-1, right=99),
    )


def test_interp_period():
    np.testing.assert_allclose(
        ferray.interp([0.5], [0, 1], [0, 10], period=2), np.interp([0.5], [0, 1], [0, 10], period=2)
    )


def test_interp_basic_unchanged():
    np.testing.assert_allclose(
        ferray.interp([1.5, 2.5], [1, 2, 3], [10, 20, 30]),
        np.interp([1.5, 2.5], [1, 2, 3], [10, 20, 30]),
    )


# ---------------------------------------------------------------------------
# fmax/fmin integer dtype preservation (#987)
# ---------------------------------------------------------------------------
# numpy.fmax/fmin have integer loops and keep the promoted integer dtype (for
# non-NaN ints they equal maximum/minimum). The prior float-promote binding
# upcast int -> float64.


def test_fmax_int_preserves_dtype():
    r = ferray.fmax([1, 5, 3], [4, 2, 6])
    n = np.fmax([1, 5, 3], [4, 2, 6])
    assert np.asarray(r).dtype == n.dtype == np.int64
    np.testing.assert_array_equal(r, n)


def test_fmin_int_preserves_dtype():
    r = ferray.fmin([1, 5, 3], [4, 2, 6])
    n = np.fmin([1, 5, 3], [4, 2, 6])
    assert np.asarray(r).dtype == n.dtype == np.int64
    np.testing.assert_array_equal(r, n)


def test_fmax_uint8_preserved():
    a = np.array([1, 5], np.uint8)
    b = np.array([3, 2], np.uint8)
    assert np.asarray(ferray.fmax(a, b)).dtype == np.uint8


def test_fmax_float_nan_suppress_unchanged():
    np.testing.assert_array_equal(
        ferray.fmax([1.0, np.nan, 3], [np.nan, 2, np.nan]),
        np.fmax([1.0, np.nan, 3], [np.nan, 2, np.nan]),
    )


def test_fmax_mixed_int_float_promotes():
    assert np.asarray(ferray.fmax([1, 5], [4.0, 2])).dtype == np.float64


def test_fmax_complex_preserved():
    r = ferray.fmax([1 + 1j, 2], [3, 1 + 5j])
    assert np.asarray(r).dtype == np.complex128
    np.testing.assert_array_equal(r, np.fmax([1 + 1j, 2], [3, 1 + 5j]))
