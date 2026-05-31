"""float16 binary arithmetic + NEP-50 promotion: `import ferray as fr` vs `numpy`.

Pins the float16 arithmetic divergences closed by #953 (REQ-3): `fr.add`/
`subtract`/`multiply`/`divide`/`power`/`floor_divide`/`remainder`/`mod`/
`maximum`/`minimum` over float16 — which the real-only `match_dtype_numeric!` /
`match_dtype_float_or_int!` dispatch previously rejected with
`TypeError: unsupported dtype for numeric op: "float16"`.

The binding now detects a float16 operand and DELEGATES the op to numpy's own
top-level function on the original operands (`ferray-python/src/ufunc.rs`,
`binary_involves_float16` + `f16_binary_delegate`), so numpy owns BOTH the
NEP-50 promotion (`result_type`) AND the float16 compute contract (numpy
computes float16 arithmetic at float32 width and rounds the result back to
float16; `.design/ferray-core-float16.md`). This keeps `half::f16` out of the
Rust boundary entirely — the same detect-and-delegate pattern as the float16
reductions (#954) and creation coercion (#952).

Every expected value is derived LIVE from numpy (installed numpy 2.4.x = the
oracle), never literal-copied from the ferray side (goal.md R-CHAR-3).
"""

import numpy as np
import pytest

import ferray as fr

# Every binary arithmetic op closed by #953, paired with its numpy oracle.
ARITH_OPS = [
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "floor_divide",
    "remainder",
    "mod",
    "maximum",
    "minimum",
]


def _eq(frv, npv):
    """Result dtype AND value must match numpy exactly (float16 in -> float16 out)."""
    fa = np.asarray(frv)
    na = np.asarray(npv)
    assert str(fa.dtype) == str(na.dtype), f"dtype {fa.dtype} != numpy {na.dtype}"
    assert np.array_equal(fa, na, equal_nan=True), f"value {fa!r} != numpy {na!r}"


# ---------------------------------------------------------------------------
# f16 + f16 -> f16: every op computes and narrows like numpy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ARITH_OPS)
def test_f16_f16_returns_float16(op):
    # numpy: np.<op>(f16, f16).dtype == float16 (computed at f32, narrowed).
    a = np.array([1, 2, 3, 7], dtype="float16")
    b = np.array([4, 5, 6, 2], dtype="float16")
    fr_f = getattr(fr, op)
    np_f = getattr(np, op)
    _eq(fr_f(a, b), np_f(a, b))


@pytest.mark.parametrize("op", ARITH_OPS)
def test_f16_f16_2d_preserves_shape(op):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float16")
    b = np.array([[2.0, 2.0], [1.0, 8.0]], dtype="float16")
    fr_f = getattr(fr, op)
    np_f = getattr(np, op)
    r = np.asarray(fr_f(a, b))
    assert r.shape == (2, 2)
    _eq(r, np_f(a, b))


# ---------------------------------------------------------------------------
# NEP-50 promotion: numpy's result_type is the oracle for the result dtype
# ---------------------------------------------------------------------------


def test_f16_plus_f32_promotes_to_float32():
    # numpy NEP-50: f16 + f32 -> float32 (verified live, numpy 2.4).
    a = np.array([1, 2, 3], dtype="float16")
    b = np.array([4, 5, 6], dtype="float32")
    assert str(np.add(a, b).dtype) == "float32"  # oracle
    _eq(fr.add(a, b), np.add(a, b))


def test_f16_plus_f64_promotes_to_float64():
    a = np.array([1, 2, 3], dtype="float16")
    b = np.array([4, 5, 6], dtype="float64")
    assert str(np.add(a, b).dtype) == "float64"  # oracle
    _eq(fr.add(a, b), np.add(a, b))


def test_f16_plus_python_float_stays_float16():
    # NEP-50 weak scalar: f16 + python-float -> float16.
    a = np.array([1, 2, 3], dtype="float16")
    assert str(np.add(a, 1.0).dtype) == "float16"  # oracle
    _eq(fr.add(a, 1.0), np.add(a, 1.0))


def test_f16_plus_python_int_stays_float16():
    a = np.array([1, 2, 3], dtype="float16")
    assert str(np.add(a, 1).dtype) == "float16"  # oracle
    _eq(fr.add(a, 1), np.add(a, 1))


def test_f16_plus_int32_array_promotes_to_float64():
    # NEP-50 array-int promotion: f16 + int32-array -> float64.
    a = np.array([1, 2, 3], dtype="float16")
    b = np.array([1, 2, 3], dtype="int32")
    assert str(np.add(a, b).dtype) == "float64"  # oracle
    _eq(fr.add(a, b), np.add(a, b))


def test_int32_array_plus_f16_reflected_promotes_to_float64():
    # Reflected operand order resolves through the same numpy result_type.
    a = np.array([1, 2, 3], dtype="int32")
    b = np.array([1, 2, 3], dtype="float16")
    assert str(np.add(a, b).dtype) == "float64"  # oracle
    _eq(fr.add(a, b), np.add(a, b))


def test_f32_plus_f16_reflected_promotes_to_float32():
    a = np.array([1, 2, 3], dtype="float32")
    b = np.array([4, 5, 6], dtype="float16")
    assert str(np.add(a, b).dtype) == "float32"  # oracle
    _eq(fr.add(a, b), np.add(a, b))


@pytest.mark.parametrize("op", ARITH_OPS)
def test_f16_op_python_scalar_stays_float16(op):
    # Every op with a weak python scalar keeps float16 (NEP-50).
    a = np.array([1, 2, 3, 8], dtype="float16")
    fr_f = getattr(fr, op)
    np_f = getattr(np, op)
    assert str(np_f(a, 2.0).dtype) == "float16"  # oracle
    _eq(fr_f(a, 2.0), np_f(a, 2.0))


# ---------------------------------------------------------------------------
# Per-op correctness: division-by-zero, power, floor/remainder, min/max NaN
# ---------------------------------------------------------------------------


def test_f16_divide_is_true_division():
    a = np.array([1, 2, 3], dtype="float16")
    b = np.array([2, 4, 6], dtype="float16")
    # numpy: f16 true-division -> 0.5 each, dtype float16.
    _eq(fr.divide(a, b), np.divide(a, b))


def test_f16_divide_by_zero_is_inf_not_panic():
    # numpy returns inf (RuntimeWarning), does NOT raise.
    a = np.array([1.0, -1.0, 0.0], dtype="float16")
    b = np.array([0.0, 0.0, 0.0], dtype="float16")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.divide(a, b)
    _eq(fr.divide(a, b), expected)


def test_f16_power():
    a = np.array([2.0, 3.0, 4.0], dtype="float16")
    b = np.array([2.0, 0.0, 0.5], dtype="float16")
    _eq(fr.power(a, b), np.power(a, b))


def test_f16_floor_divide():
    a = np.array([7.0, 8.0, 9.0], dtype="float16")
    b = np.array([2.0, 3.0, 4.0], dtype="float16")
    _eq(fr.floor_divide(a, b), np.floor_divide(a, b))


def test_f16_remainder_and_mod_agree():
    a = np.array([7.0, 8.0, 9.0], dtype="float16")
    b = np.array([3.0, 3.0, 4.0], dtype="float16")
    _eq(fr.remainder(a, b), np.remainder(a, b))
    _eq(fr.mod(a, b), np.mod(a, b))


def test_f16_maximum_minimum():
    a = np.array([1.0, 5.0, 3.0], dtype="float16")
    b = np.array([4.0, 2.0, 3.0], dtype="float16")
    _eq(fr.maximum(a, b), np.maximum(a, b))
    _eq(fr.minimum(a, b), np.minimum(a, b))


def test_f16_maximum_propagates_nan():
    # numpy maximum/minimum PROPAGATE NaN (unlike fmax/fmin).
    a = np.array([np.nan, 5.0], dtype="float16")
    b = np.array([4.0, 2.0], dtype="float16")
    _eq(fr.maximum(a, b), np.maximum(a, b))
    _eq(fr.minimum(a, b), np.minimum(a, b))


def test_f16_overflow_to_inf():
    # numpy float16 arithmetic overflows to inf (max finite f16 ~= 65504).
    big = np.array([60000.0, 60000.0], dtype="float16")
    with np.errstate(over="ignore"):
        expected = np.add(big, big)
    assert np.all(np.isinf(np.asarray(expected)))  # oracle: overflow -> inf
    _eq(fr.add(big, big), expected)


def test_f16_multiply_overflow_to_inf():
    big = np.array([300.0, 300.0], dtype="float16")
    with np.errstate(over="ignore"):
        expected = np.multiply(big, big)
    _eq(fr.multiply(big, big), expected)


# ---------------------------------------------------------------------------
# Scalar collapse: all-scalar inputs -> numpy scalar (not 0-d ndarray)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ARITH_OPS)
def test_f16_all_scalar_returns_numpy_scalar(op):
    fr_f = getattr(fr, op)
    np_f = getattr(np, op)
    r = fr_f(np.float16(6.0), np.float16(2.0))
    e = np_f(np.float16(6.0), np.float16(2.0))
    assert str(np.asarray(r).dtype) == str(np.asarray(e).dtype) == "float16"
    assert np.asarray(r) == np.asarray(e)


# ---------------------------------------------------------------------------
# Regression guard: the real f32/f64/int/complex paths are UNCHANGED.
# ---------------------------------------------------------------------------


def test_real_float32_add_unchanged():
    a = np.array([1.0, 2.0, 3.0], dtype="float32")
    b = np.array([4.0, 5.0, 6.0], dtype="float32")
    _eq(fr.add(a, b), np.add(a, b))


def test_real_int64_ops_unchanged():
    a = np.array([7, 8, 9], dtype="int64")
    b = np.array([2, 3, 4], dtype="int64")
    _eq(fr.add(a, b), np.add(a, b))
    _eq(fr.maximum(a, b), np.maximum(a, b))
    _eq(fr.floor_divide(a, b), np.floor_divide(a, b))
    # int true-division promotes to float64 (numpy contract) — unchanged.
    _eq(fr.divide(a, b), np.divide(a, b))


def test_real_complex_add_unchanged():
    c = np.array([1 + 2j, 3 + 4j], dtype="complex128")
    _eq(fr.add(c, c), np.add(c, c))
