"""Adversarial divergence tests for ferray's array-creation surface.

Each test pins a CONFIRMED divergence between ``import ferray as fr`` and the
NumPy oracle (numpy 2.4.5). The NumPy oracle value is computed LIVE in every
test, never hard-coded, and the assertion compares value + dtype + shape (or
matching exception TYPE). Each docstring cites the upstream numpy source
(file:line) plus expected-vs-actual.

These tests are EXPECTED TO FAIL on current ferray; they document work the
generator must fix. They are written by the adversarial critic and contain no
fixes.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# arange: float computation method (start + i*step vs accumulation)
# ---------------------------------------------------------------------------


def test_arange_float_uses_start_plus_i_times_step():
    """numpy arange fills each element as ``start + i*step`` (a fresh
    multiplication), NOT a running accumulation.

    numpy _core/multiarray.pyi arange overload + the C ``fill`` loop in
    numpy/_core/src/multiarray/arraytypes.c.src use ``start + i*delta``.
    Observable: np.arange(0.1, 1.0, 0.1)[5] == 0.6000000000000001? No —
    numpy yields exactly 0.6 (== 0.1 + 5*0.1), while ferray accumulates and
    yields 0.6000000000000001.

    Expected (numpy): np.arange(0.1, 1.0, 0.1) bit-exact.
    Actual (ferray): differs at index 5 (accumulation rounding).
    """
    oracle = np.arange(0.1, 1.0, 0.1)
    got = fr.arange(0.1, 1.0, 0.1)
    assert got.dtype == oracle.dtype
    assert got.shape == oracle.shape
    # Bit-exact equality is numpy's contract here (each element independent).
    assert np.array_equal(got, oracle), (
        f"index-wise mismatch: numpy={oracle.tolist()} ferray={got.tolist()}"
    )


def test_arange_float_endpoint_value_exact():
    """np.arange(1, 2, 0.3) last element is ``1 + 3*0.3`` evaluated fresh,
    i.e. 1.9000000000000001, not the accumulated 1.9.

    Same root cause as above: numpy computes start + i*step per element.

    Expected (numpy): last element repr 1.9000000000000001.
    Actual (ferray): 1.9 (accumulation).
    """
    oracle = np.arange(1.0, 2.0, 0.3)
    got = fr.arange(1.0, 2.0, 0.3)
    assert np.array_equal(got, oracle), (
        f"numpy={oracle.tolist()} ferray={got.tolist()}"
    )


# ---------------------------------------------------------------------------
# arange: dtype inference must key off Python *type*, never float .fract()
# ---------------------------------------------------------------------------


def test_arange_float_arg_yields_float_dtype():
    """A float-typed argument forces a float result dtype regardless of
    value. numpy/_core keys arange dtype off the operand *type*.

    numpy/_core/_type_aliases / ``np.arange``: ``np.arange(3.0).dtype`` is
    float64 because 3.0 is a Python float — the value being integral is
    irrelevant.

    Expected (numpy): np.arange(3.0).dtype == float64.
    Actual (ferray): int64 (binding inspects f64.fract()==0 instead of the
    Python type, so 3.0 is misclassified as integer).
    """
    oracle = np.arange(3.0)
    got = fr.arange(3.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


def test_arange_float_start_stop_yields_float_dtype():
    """np.arange(0.0, 3.0) is float64 — both endpoints are Python floats.

    Same contract as above; pins the two-argument form.

    Expected (numpy): float64. Actual (ferray): int64.
    """
    oracle = np.arange(0.0, 3.0)
    got = fr.arange(0.0, 3.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )


# ---------------------------------------------------------------------------
# arange: zero step exception TYPE
# ---------------------------------------------------------------------------


def test_arange_zero_step_raises_zerodivision():
    """numpy raises ``ZeroDivisionError`` (or, in some builds, ValueError —
    but never a ferray-specific message) for a zero step.

    numpy/_core/multiarray ``arange``: zero step triggers a division while
    computing the length, surfacing as ZeroDivisionError in numpy 2.4.5.

    Expected (numpy): ZeroDivisionError.
    Actual (ferray): ValueError('invalid value: step cannot be zero').
    """
    with pytest.raises(ZeroDivisionError):
        np.arange(0, 5, 0)  # establish the oracle TYPE
    with pytest.raises(ZeroDivisionError):
        fr.arange(0, 5, 0)


def test_arange_float_step_int_dtype_floors_not_zero_step():
    """np.arange(0, 5, 0.5, dtype='int64') computes the float sequence then
    casts — it does NOT cast the step to int first.

    numpy/_core function: arange with explicit integer dtype still uses the
    float step length, producing 10 elements (all floored to 0), never an
    error about a zero step.

    Expected (numpy): array of length 10, all 0, dtype int64.
    Actual (ferray): ValueError('step cannot be zero') because the binding
    casts step 0.5 -> 0 before computing.
    """
    oracle = np.arange(0, 5, 0.5, dtype="int64")
    got = fr.arange(0, 5, 0.5, dtype="int64")
    assert got.dtype == oracle.dtype
    assert got.shape == oracle.shape
    assert np.array_equal(got, oracle), (
        f"numpy={oracle.tolist()} ferray={got.tolist()}"
    )


# ---------------------------------------------------------------------------
# full / full default dtype: keyed off the fill-value Python type, not value
# ---------------------------------------------------------------------------


def test_full_float_fill_yields_float_dtype():
    """np.full((2,), 1.0) is float64 — the fill value is a Python float.

    numpy/_core/numeric.py ``full``:
        ``if dtype is None: dtype = np.array(fill_value).dtype`` — so the
    *type* of fill_value decides the dtype. 1.0 -> float64 even though the
    value is integral.

    Expected (numpy): float64. Actual (ferray): int64 (binding checks
    fill_value.fract()==0.0 instead of the source Python type).
    """
    oracle = np.full((2,), 1.0)
    got = fr.full((2,), 1.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


def test_full_bool_fill_yields_bool_dtype():
    """np.full((3,), True) is bool — ``np.array(True).dtype`` is bool.

    numpy/_core/numeric.py ``full`` infers dtype from ``np.array(fill_value)``.

    Expected (numpy): dtype bool, values [True, True, True].
    Actual (ferray): int64 [1, 1, 1] (binding coerces fill_value to f64,
    losing the bool type entirely).
    """
    oracle = np.full((3,), True)
    got = fr.full((3,), True)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


# ---------------------------------------------------------------------------
# Negative dimension: exception TYPE (ValueError, not TypeError/OverflowError)
# ---------------------------------------------------------------------------


def test_zeros_negative_dim_raises_valueerror():
    """numpy raises ``ValueError('negative dimensions are not allowed')``.

    numpy/_core/multiarray ``zeros``/``empty`` reject negative dimensions
    with ValueError.

    Expected (numpy): ValueError. Actual (ferray): TypeError ('shape must be
    an int or a sequence of ints').
    """
    with pytest.raises(ValueError):
        np.zeros(-1)  # oracle TYPE
    with pytest.raises(ValueError):
        fr.zeros(-1)


def test_eye_negative_n_raises_valueerror():
    """np.eye(-1) raises ValueError('negative dimensions are not allowed').

    numpy/_core/twodim_base.py ``eye``: builds zeros((N, M)) which rejects
    negative N with ValueError.

    Expected (numpy): ValueError. Actual (ferray): OverflowError ("can't
    convert negative int to unsigned").
    """
    with pytest.raises(ValueError):
        np.eye(-1)  # oracle TYPE
    with pytest.raises(ValueError):
        fr.eye(-1)


def test_identity_negative_n_raises_valueerror():
    """np.identity(-2) raises ValueError, not OverflowError.

    numpy/_core/numeric.py ``identity`` -> eye(n) -> zeros((n,n)) rejects
    negative n with ValueError.

    Expected (numpy): ValueError. Actual (ferray): OverflowError.
    """
    with pytest.raises(ValueError):
        np.identity(-2)  # oracle TYPE
    with pytest.raises(ValueError):
        fr.identity(-2)


# ---------------------------------------------------------------------------
# linspace: negative num exception TYPE + missing retstep kwarg
# ---------------------------------------------------------------------------


def test_linspace_negative_num_raises_valueerror():
    """numpy/_core/function_base.py:124-127 linspace:
        ``if num < 0: raise ValueError(f"Number of samples, {num}, must be
        non-negative.")``

    Expected (numpy): ValueError. Actual (ferray): OverflowError ("can't
    convert negative int to unsigned") — num is bound as ``usize`` so the
    PyO3 conversion fails before ferray's own check runs.
    """
    with pytest.raises(ValueError):
        np.linspace(0, 1, -1)  # oracle TYPE (function_base.py:125)
    with pytest.raises(ValueError):
        fr.linspace(0, 1, -1)


def test_linspace_retstep_returns_tuple():
    """numpy/_core/function_base.py:28-29 + :185-188 linspace signature
    includes ``retstep=False``; when True it returns ``(samples, step)``.

        ``if retstep: return y, step``  (function_base.py:185-186)

    Expected (numpy): linspace(0,1,5,retstep=True) -> (ndarray, 0.25).
    Actual (ferray): TypeError — ``retstep`` is not an accepted kwarg.
    """
    samples_oracle, step_oracle = np.linspace(0.0, 1.0, 5, retstep=True)
    result = fr.linspace(0.0, 1.0, 5, retstep=True)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"expected (samples, step) tuple, got {type(result)}"
    )
    samples, step = result
    assert np.array_equal(samples, samples_oracle)
    assert step == step_oracle


# ---------------------------------------------------------------------------
# *_like: dtype support gap (float16 must be preserved)
# ---------------------------------------------------------------------------


def test_zeros_like_preserves_float16_dtype():
    """np.zeros_like must preserve the source array's dtype, including
    float16.

    numpy/_core/numeric.py ``zeros_like``: ``res = empty_like(a, dtype=dtype,
    ...)`` with dtype=None inheriting ``a.dtype``.

    Expected (numpy): float16 zeros. Actual (ferray): TypeError ('unsupported
    dtype: "float16"') — the binding's dtype dispatch omits float16.
    """
    src = np.zeros(2, dtype=np.float16)
    oracle = np.zeros_like(src)
    got = fr.zeros_like(src)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


# ---------------------------------------------------------------------------
# geomspace: negative endpoints must be exact (out_sign rotation)
# ---------------------------------------------------------------------------


def test_geomspace_negative_endpoints_exact():
    """numpy/_core/function_base.py:432-448 geomspace rotates ``start`` to
    positive real (``out_sign = sign(start); start /= out_sign``), computes
    in log space, then forces the endpoints back exactly:

        ``result[0] = start`` ... ``result[-1] = stop``  (lines 444-446)

    so np.geomspace(-1, -1000, 4) yields EXACT [-1, -10, -100, -1000].

    Expected (numpy): exact integers (e.g. element 1 == -10.0).
    Actual (ferray): -9.999999999999998 (no endpoint pinning / sign
    rotation), drifting from numpy by > 1e-15.
    """
    oracle = np.geomspace(-1.0, -1000.0, 4)
    got = fr.geomspace(-1.0, -1000.0, 4)
    assert got.dtype == oracle.dtype
    assert got.shape == oracle.shape
    # numpy guarantees the endpoints are bit-exact; assert that contract.
    assert got[0] == oracle[0], f"start: numpy={oracle[0]!r} ferray={got[0]!r}"
    assert got[-1] == oracle[-1], (
        f"stop: numpy={oracle[-1]!r} ferray={got[-1]!r}"
    )
    # The interior value -10.0 is exact under numpy's construction.
    assert got[1] == oracle[1], (
        f"interior: numpy={oracle[1]!r} ferray={got[1]!r}"
    )
