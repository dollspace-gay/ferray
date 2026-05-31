"""Adversarial divergence pins for the ferray ufunc surface (acto-critic re-audit).

Each test computes the numpy 2.4.5 oracle value LIVE (R-CHAR-3 — never copied
from ferray) and asserts ferray matches (value + dtype, or matching exception
type). Every test is expected to FAIL against current ferray: these pin real
divergences in `ferray-python/src/ufunc.rs` and the library crates it marshals.

Upstream citations refer to the numpy working tree at /home/doll/numpy-ref.

Root cause for most int-dtype pins: `ufunc.rs` dispatches the algebraic and
"float" binary ufuncs through `match_dtype_float!`, which only knows
float32/float64 and raises `TypeError: unsupported dtype for floating-point op`
for every integer dtype. numpy registers integer loops for all of these
(see generate_umath.py cites per test). Fix lives DOWN in ferray-ufunc's dtype
dispatch (goal.md: semantic bugs fixed in the owning library crate), surfaced
through the ufunc.rs macro choice.
"""

import warnings

import numpy as np
import pytest

import ferray as fr


# ===========================================================================
# GROUP A — integer-dtype support (numpy registers int loops; ferray routes
# through float-only dispatch and raises TypeError).
# ===========================================================================


# ---------------------------------------------------------------------------
# A1: np.divide is true-division — integer inputs promote to float64.
# numpy cite: generate_umath.py:422 "'PyUFunc_TrueDivisionTypeResolver',"
#   :404 "# 'true_divide' : aliased to divide".
# Expected: divide([1,2,3],[2,2,2]) -> [0.5,1.0,1.5] float64.
# Actual (ferray): [0,1,1] int64 (floor-style integer division).
# ---------------------------------------------------------------------------
def test_divide_int_promotes_to_float_true_division():
    a = np.array([1, 2, 3])
    b = np.array([2, 2, 2])
    expected = np.divide(a, b)
    got = fr.divide(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# A2: np.divide of integers by zero yields float +inf/nan + RuntimeWarning,
#   NOT a Rust panic. ferray PANICS across the FFI boundary
#   ("attempt to divide by zero" at ferray-ufunc/src/ops/arithmetic.rs:626),
#   raising pyo3_runtime.PanicException — a release-blocker (R-CODE-2).
# numpy cite: generate_umath.py:422 TrueDivisionTypeResolver — int/int divide
#   is true division (float64), and float div-by-zero is inf/nan not a trap.
# Expected: divide([1,2],[0,0]) -> [inf, inf] float64.
# Actual (ferray): pyo3_runtime.PanicException "attempt to divide by zero".
# ---------------------------------------------------------------------------
def test_divide_int_by_zero_no_panic():
    a = np.array([1, 2])
    b = np.array([0, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = np.divide(a, b)
    got = fr.divide(a, b)  # must NOT panic
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# A3: mixed int + float promotes to float64 (the float operand's fractional
#   part must survive). ferray coerces x2 to x1's int dtype, truncating.
# numpy cite: generate_umath.py:419-429 'divide'/add use common-type
#   promotion; NEP-50 int+float -> float64.
# Expected: add([1,2],[1.5,2.5]) -> [2.5,4.5] float64.
# Actual (ferray): [2,4] int64 (second operand truncated to int).
# ---------------------------------------------------------------------------
def test_add_mixed_int_float_promotes_to_float64():
    a = np.array([1, 2])
    b = np.array([1.5, 2.5])
    expected = np.add(a, b)
    got = fr.add(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# A4: maximum / minimum support integer dtypes.
# numpy cite: generate_umath.py:661 'maximum', :671 'minimum'.
# Expected: maximum([1,5],[3,2]) -> [3,5] int64.
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
# A5: power supports integer dtypes (int**int -> int).
# numpy cite: generate_umath.py:484 "TD(ints)," (first 'power' loop).
# Expected: power([2,3],[2,3]) -> [4,27] int64.
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
# A6: unary algebraic ops keep int dtype; floor/ceil/trunc/rint accept ints.
# numpy cite: generate_umath.py:516 'negative' "TD(ints + flts ...)";
#   :496 'absolute' "TD(bints + flts ...)"; :534 'sign'.
#   floor/ceil/trunc/rint accept ints (no-op, returning the int dtype).
# Expected: negative([1,-2,3]) -> [-1,2,-3] int64; absolute([-1,-2,3]) ->
#   [1,2,3] int64; sign([-5,0,7]) -> [-1,0,1] int64; floor([1,2,3]) ->
#   [1,2,3] int64; same for ceil/trunc/rint; positive([1,-2]) -> [1,-2].
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name, src",
    [
        ("negative", [1, -2, 3]),
        ("positive", [1, -2, 3]),
        ("absolute", [-1, -2, 3]),
        ("sign", [-5, 0, 7]),
        ("floor", [1, 2, 3]),
        ("ceil", [1, 2, 3]),
        ("trunc", [1, 2, 3]),
        ("rint", [1, 2, 3]),
        ("square", [2, 3]),
        ("reciprocal", [2, 4]),
    ],
)
def test_unary_int_supported(name, src):
    a = np.array(src)
    expected = getattr(np, name)(a)
    got = getattr(fr, name)(a)
    assert got.dtype == expected.dtype, (name, got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# A7: fabs / hypot / arctan2 on int inputs promote to float64 (not reject).
# numpy cite: generate_umath.py:1003 'fabs' "TD(flts, f='fabs' ...)" — numpy
#   promotes int args up to float64 via the inexact common type; :1057 'hypot';
#   :1030 'arctan2'.
# Expected: fabs([-1,2]) -> [1.,2.] float64; hypot([3],[4]) -> [5.] float64;
#   arctan2([1],[1]) -> [0.785...] float64.
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
def test_fabs_int_promotes_float64():
    a = np.array([-1, 2])
    expected = np.fabs(a)
    got = fr.fabs(a)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


def test_hypot_int_promotes_float64():
    a = np.array([3])
    b = np.array([4])
    expected = np.hypot(a, b)
    got = fr.hypot(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


def test_arctan2_int_promotes_float64():
    a = np.array([1])
    b = np.array([1])
    expected = np.arctan2(a, b)
    got = fr.arctan2(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# A8: gcd / lcm are INTEGER-ONLY in numpy (and ferray inverts this — it
#   accepts floats and REJECTS ints, exactly backwards).
# numpy cite: generate_umath.py:1156 'gcd' "TD(ints)", :1163 'lcm' "TD(ints)".
# Expected: gcd([12,15],[8,5]) -> [4,5] int64; lcm([4,6],[6,8]) -> [12,24].
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
def test_gcd_int():
    a = np.array([12, 15])
    b = np.array([8, 5])
    expected = np.gcd(a, b)
    got = fr.gcd(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


def test_lcm_int():
    a = np.array([4, 6])
    b = np.array([6, 8])
    expected = np.lcm(a, b)
    got = fr.lcm(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# A9: gcd rejects float inputs in numpy (no float loop) — ferray silently
#   accepts them. Pin the exception-type contract (R-DEV-2).
# numpy cite: generate_umath.py:1156 'gcd' registers only "TD(ints)"; calling
#   on float64 raises numpy.exceptions.UFuncTypeError (a TypeError subclass).
# Expected: np.gcd(float, float) raises TypeError.
# Actual (ferray): returns array([2.]) — no error.
# ---------------------------------------------------------------------------
def test_gcd_float_rejected():
    a = np.array([4.0])
    b = np.array([6.0])
    with pytest.raises(TypeError):
        np.gcd(a, b)
    with pytest.raises(TypeError):
        fr.gcd(a, b)


# ===========================================================================
# GROUP B — clip surface (int support, array bounds, None bound).
# ===========================================================================


# ---------------------------------------------------------------------------
# B1: np.clip supports integer arrays, returning that int dtype.
# numpy cite: fromnumeric.py clip == minimum(a_max, maximum(a, a_min)),
#   and maximum/minimum (generate_umath.py:661/:671) carry int loops.
# Expected: clip([1,5,10],2,8) -> [2,5,8] int64.
# Actual (ferray): TypeError "unsupported dtype for floating-point op: int64".
# ---------------------------------------------------------------------------
def test_clip_int_array():
    src = np.array([1, 5, 10])
    expected = np.clip(src, 2, 8)
    got = fr.clip(src, 2, 8)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# B2: np.clip accepts array-valued a_min / a_max bounds (broadcast).
#   ferray types a_min/a_max as scalar f64, so an array bound fails coercion.
# numpy cite: fromnumeric.py clip(a, a_min, a_max) "array_like or None".
# Expected: clip([1.,5.,10.],[2.,2.,2.],[8.,8.,8.]) -> [2.,5.,8.].
# Actual (ferray): TypeError "argument 'a_min': only 0-dimensional arrays can
#   be converted to Python scalars".
# ---------------------------------------------------------------------------
def test_clip_array_bounds():
    src = np.array([1.0, 5.0, 10.0])
    lo = np.array([2.0, 2.0, 2.0])
    hi = np.array([8.0, 8.0, 8.0])
    expected = np.clip(src, lo, hi)
    got = fr.clip(src, lo, hi)
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# B3: np.clip accepts None for a missing bound (one-sided clip). ferray types
#   a_min as non-optional f64.
# numpy cite: fromnumeric.py clip — a None edge means "do not clip that side".
# Expected: clip([1.,5.,10.], None, 8.0) -> [1.,5.,8.].
# Actual (ferray): TypeError "argument 'a_min': must be real number, not
#   NoneType".
# ---------------------------------------------------------------------------
def test_clip_none_lower_bound():
    src = np.array([1.0, 5.0, 10.0])
    expected = np.clip(src, None, 8.0)
    got = fr.clip(src, None, 8.0)
    np.testing.assert_array_equal(got, expected)


# ===========================================================================
# GROUP C — missing top-level functions.
# ===========================================================================


# ---------------------------------------------------------------------------
# C1: top-level ufunc aliases/functions present in numpy but absent in ferray.
# numpy cite: generate_umath.py:404 "'true_divide' : aliased to divide",
#   :405 'floor_divide', :490 'float_power', :1039 'remainder' (mod is an
#   alias of remainder, see umath.py module exports).
# Expected: fr exposes true_divide / floor_divide / mod / remainder /
#   float_power.
# Actual (ferray): AttributeError — not registered in __init__.py.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name", ["true_divide", "floor_divide", "mod", "remainder", "float_power"]
)
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


def test_float_power_value():
    a = np.array([2, 3])
    b = np.array([2, 3])
    expected = np.float_power(a, b)  # always float64, even for int inputs
    got = fr.float_power(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


def test_mod_value():
    a = np.array([7, 8])
    b = np.array([3, 3])
    expected = np.mod(a, b)
    got = fr.mod(a, b)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)


# ===========================================================================
# GROUP D — output object contract: scalar vs 0-d ndarray.
# ===========================================================================


# ---------------------------------------------------------------------------
# D1: a ufunc on scalar / 0-d inputs returns a numpy scalar, not a 0-d
#   ndarray (numpy's $OUT_SCALAR contract — R-DEV-3).
# numpy cite: ufunc_docstrings $OUT_SCALAR_2; np.add(np.array(1.),np.array(2.))
#   -> np.float64(3.0), type is numpy.float64, not ndarray.
# Expected: type(np.add(np.array(1.0),np.array(2.0))) is np.float64.
# Actual (ferray): returns a 0-d numpy.ndarray.
# ---------------------------------------------------------------------------
def test_binary_scalar_return_type():
    expected = np.add(np.array(1.0), np.array(2.0))
    got = fr.add(np.array(1.0), np.array(2.0))
    assert type(got) is type(expected), (type(got), type(expected))


def test_unary_scalar_return_type():
    expected = np.sin(np.array(0.0))
    got = fr.sin(np.array(0.0))
    assert type(got) is type(expected), (type(got), type(expected))


def test_comparison_scalar_return_type():
    expected = np.equal(np.array(1), np.array(1))
    got = fr.equal(np.array(1), np.array(1))
    assert type(got) is type(expected), (type(got), type(expected))


# ===========================================================================
# GROUP E — ufunc kwarg ABI surface (out=).
# ===========================================================================


# ---------------------------------------------------------------------------
# E1: numpy ufuncs accept an out= keyword writing the result in-place and
#   returning it. ferray's bindings declare no out= parameter.
# numpy cite: ufunc_docstrings $OUT — every binary ufunc takes
#   "out : ndarray, None, or tuple of ndarray and None, optional".
# Expected: add([1.,2.],[3.,4.],out=o) writes [4.,6.] into o and returns it.
# Actual (ferray): TypeError "add() got an unexpected keyword argument 'out'".
# ---------------------------------------------------------------------------
def test_add_out_kwarg():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    expected_out = np.empty(2)
    np.add(a, b, out=expected_out)
    got_out = np.empty(2)
    fr.add(a, b, out=got_out)
    np.testing.assert_array_equal(got_out, expected_out)


# ===========================================================================
# GROUP F — acto-critic re-audit of the directly-edited ufunc.rs helpers
#   (ediff1d scalar/array-like appendage, trapezoid x-dtype promotion).
#   numpy oracle: live 2.4.4 at /home/doll/ferray/.venv numpy.
# ===========================================================================


# ---------------------------------------------------------------------------
# F1: ediff1d to_end / to_begin accept ANY array_like and numpy RAVELS it
#   (numpy/lib/_arraysetops_impl.py:110 `to_end = np.asanyarray(to_end)` then
#   :115 `to_end = to_end.ravel()`). A 2-D appendage is flattened.
# ferray's `ediff1d_appendage` (ufunc.rs:3074) only tries `Vec<f64>` (1-D) then
#   a scalar `f64`, so a 2-D list raises before reaching the kernel.
# Expected (numpy): ediff1d([1,2,4], to_end=[[97,98],[1,2]]) -> [1,2,97,98,1,2].
# Actual (ferray): TypeError "must be real number, not list".
# ---------------------------------------------------------------------------
def test_ediff1d_to_end_2d_ravel():
    ary = [1, 2, 4]
    to_end = [[97, 98], [1, 2]]
    expected = np.ediff1d(ary, to_end=to_end)
    got = fr.ediff1d(ary, to_end=to_end)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)


# ---------------------------------------------------------------------------
# F2: ediff1d enforces the `same_kind` casting rule on the appendage
#   (numpy/lib/_arraysetops_impl.py:111-113 `if not np.can_cast(to_end,
#   dtype_req, casting="same_kind"): raise TypeError`). A float appendage on an
#   integer input is NOT castable same_kind -> numpy raises TypeError.
# ferray funnels every appendage through `Vec<f64>` then casts back to the
#   element type `as T` (ufunc.rs:3143-3144), silently TRUNCATING 0.5 -> 0
#   (R-CODE-4 silent data corruption) instead of raising.
# Expected (numpy): ediff1d(int_array, to_end=0.5) raises TypeError.
# Actual (ferray): returns [1, 2, 0] (0.5 truncated to int 0).
# ---------------------------------------------------------------------------
def test_ediff1d_int_float_appendage_same_kind_error():
    ary = np.array([1, 2, 4], dtype=np.int64)
    # numpy raises; assert ferray raises the same error type.
    raised_np = False
    try:
        np.ediff1d(ary, to_end=0.5)
    except TypeError:
        raised_np = True
    assert raised_np, "oracle precondition: numpy must reject float appendage on int input"
    with pytest.raises(TypeError):
        fr.ediff1d(ary, to_end=0.5)


# ---------------------------------------------------------------------------
# F3: trapezoid result dtype follows result_type(y, x), NOT y alone.
#   numpy computes `d = diff(x)` (numpy/lib/_function_base_impl.py:5035) keeping
#   x's dtype, then `d * (y[1:]+y[:-1]) / 2.0` (:5048) — NEP-50 promotes a
#   float32 y times a float64 d to float64. ferray derives `real_dt` from `y`
#   alone (ufunc.rs:3037-3041) and coerces x DOWN to y's dtype, returning
#   float32 and computing the sum in float32 (R-CODE-4 lossy round-trip).
# Expected (numpy): trapezoid(f32 y, x=f64) -> np.float64.
# Actual (ferray): np.float32 (wrong dtype AND wrong value at large magnitudes).
# ---------------------------------------------------------------------------
def test_trapezoid_f32y_f64x_promotes_to_float64():
    y = np.array([1e8, 2e8, 3e8], dtype=np.float32)
    x = np.array([0.0, 1.0000001, 3.0000003], dtype=np.float64)
    expected = np.trapezoid(y, x=x)
    got = fr.trapezoid(y, x=x)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    np.testing.assert_array_equal(got, expected)
