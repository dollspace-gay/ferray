"""Adversarial divergence tests for ferray's array-creation surface.

Each test pins a CONFIRMED divergence between ``import ferray as fr`` and the
NumPy oracle (numpy 2.4.5). The oracle value/type is computed LIVE in every
test (R-CHAR-3 — never literal-copied from the ferray side); the assertion
compares value + dtype + shape, or the raised exception TYPE. Every docstring
cites the upstream numpy source (file:line) plus expected-vs-actual.

These tests are EXPECTED TO FAIL on the current build; they document work the
generator must fix. Written by the adversarial critic; contain no fixes.

Layer tags:
  [binding]  — fix lives in ferray-python/src/creation.rs (marshalling / dtype
               inference / exception mapping / kwarg surface / dtype dispatch).
  [library]  — fix lives down in ferray-core (numeric construction).
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# arange: float element computation (start + i*step, evaluated per element)
# [library] ferray-core arange numeric construction
# ---------------------------------------------------------------------------


def test_arange_float_index5_matches_numpy():
    """numpy fills arange float output as ``start + i*step`` per element
    (numpy/_core/src/multiarray/arraytypes.c.src @@fill@@ loop), so
    ``np.arange(0.1, 1.0, 0.1)[5]`` is exactly ``0.6``.

    Expected (numpy): index 5 == 0.6 and the whole array is bit-equal.
    Actual (ferray): index 5 == 0.6000000000000001 (construction diverges).
    """
    oracle = np.arange(0.1, 1.0, 0.1)
    got = fr.arange(0.1, 1.0, 0.1)
    assert got.dtype == oracle.dtype
    assert got.shape == oracle.shape
    assert np.array_equal(got, oracle), (
        f"index-wise mismatch: numpy={oracle.tolist()} ferray={got.tolist()}"
    )


def test_arange_float_last_element_matches_numpy():
    """``np.arange(1.0, 2.0, 0.3)`` last element == ``1 + 3*0.3`` evaluated
    fresh == 1.9000000000000001.

    Expected (numpy): last element repr 1.9000000000000001.
    Actual (ferray): 1.9 (construction diverges from start + i*step).
    """
    oracle = np.arange(1.0, 2.0, 0.3)
    got = fr.arange(1.0, 2.0, 0.3)
    assert np.array_equal(got, oracle), (
        f"numpy={oracle.tolist()} ferray={got.tolist()}"
    )


# ---------------------------------------------------------------------------
# arange: dtype inference keys off operand *type*, not value .fract()
# [binding] creation.rs `arange` inspects f64.fract()==0 instead of the
#           Python type of the argument
# ---------------------------------------------------------------------------


def test_arange_single_float_arg_yields_float_dtype():
    """``np.arange(3.0).dtype`` is float64 because 3.0 is a Python float;
    the integral value is irrelevant (function_base docstring + numpy
    arange C dispatch on operand type).

    Expected (numpy): float64.  Actual (ferray): int64.
    """
    oracle = np.arange(3.0)
    got = fr.arange(3.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


def test_arange_float_start_stop_yields_float_dtype():
    """``np.arange(0.0, 3.0).dtype`` is float64 — both endpoints are
    Python floats.

    Expected (numpy): float64.  Actual (ferray): int64.
    """
    oracle = np.arange(0.0, 3.0)
    got = fr.arange(0.0, 3.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )


def test_arange_float_step_int_endpoints_yields_float_dtype():
    """A Python-float *step* alone forces float64 even when start/stop are
    integers: ``np.arange(0, 3, 1.0).dtype`` is float64.

    Expected (numpy): float64.  Actual (ferray): int64 — the binding's
    ``[s, e, st].fract()==0`` heuristic misclassifies an all-integral-valued
    float step as integer. (Distinct input from the single/double-arg pins.)
    """
    oracle = np.arange(0, 3, 1.0)
    got = fr.arange(0, 3, 1.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


# ---------------------------------------------------------------------------
# arange: zero step exception TYPE
# [binding] exception mapping in creation.rs / ferray-core
# ---------------------------------------------------------------------------


def test_arange_zero_step_raises_zerodivision():
    """numpy 2.4.5 raises ``ZeroDivisionError`` for a zero step (length is
    computed by dividing by step).

    Expected (numpy): ZeroDivisionError.
    Actual (ferray): ValueError('invalid value: step cannot be zero').
    """
    with pytest.raises(ZeroDivisionError):
        np.arange(0, 5, 0)  # establish oracle TYPE
    with pytest.raises(ZeroDivisionError):
        fr.arange(0, 5, 0)


def test_arange_float_step_with_int_dtype_does_not_zero_step_error():
    """``np.arange(0, 5, 0.5, dtype='int64')`` computes the float sequence
    (10 elements) then casts — it does NOT truncate the step to int first.

    Expected (numpy): length-10 int64 array of all zeros.
    Actual (ferray): ValueError('step cannot be zero') — the binding casts
    step 0.5 -> 0 before constructing.
    """
    oracle = np.arange(0, 5, 0.5, dtype="int64")
    got = fr.arange(0, 5, 0.5, dtype="int64")
    assert got.dtype == oracle.dtype
    assert got.shape == oracle.shape
    assert np.array_equal(got, oracle), (
        f"numpy={oracle.tolist()} ferray={got.tolist()}"
    )


# ---------------------------------------------------------------------------
# full: default dtype keys off the fill-value Python type, not its value
# [binding] full/full_impl coerces fill_value to f64 and inspects .fract()
# ---------------------------------------------------------------------------


def test_full_float_fill_yields_float_dtype():
    """numpy/_core/numeric.py full: dtype defaults to
    ``np.array(fill_value).dtype``, so ``np.full((2,), 1.0)`` is float64.

    Expected (numpy): float64.  Actual (ferray): int64 (fract()==0 heuristic).
    """
    oracle = np.full((2,), 1.0)
    got = fr.full((2,), 1.0)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


def test_full_bool_fill_yields_bool_dtype():
    """``np.full((3,), True)`` is bool — ``np.array(True).dtype`` is bool.

    Expected (numpy): dtype bool, [True, True, True].
    Actual (ferray): int64 [1, 1, 1] — the binding binds fill_value as f64,
    erasing the bool type at the boundary.
    """
    oracle = np.full((3,), True)
    got = fr.full((3,), True)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


# ---------------------------------------------------------------------------
# Negative dimension: exception TYPE (ValueError everywhere in numpy)
# [binding] shape/n bound as usize -> PyO3 raises before ferray's own check
# ---------------------------------------------------------------------------


def test_zeros_negative_dim_raises_valueerror():
    """numpy zeros/empty reject negative dimensions with
    ``ValueError('negative dimensions are not allowed')``.

    Expected (numpy): ValueError.  Actual (ferray): TypeError ('shape must
    be an int or a sequence of ints').
    """
    with pytest.raises(ValueError):
        np.zeros(-1)  # oracle TYPE
    with pytest.raises(ValueError):
        fr.zeros(-1)


def test_eye_negative_n_raises_valueerror():
    """``np.eye(-1)`` raises ValueError('negative dimensions are not
    allowed') (numpy/_core/twodim_base.py eye -> zeros((N, M))).

    Expected (numpy): ValueError.  Actual (ferray): OverflowError
    ("can't convert negative int to unsigned") — n is bound as ``usize``.
    """
    with pytest.raises(ValueError):
        np.eye(-1)  # oracle TYPE
    with pytest.raises(ValueError):
        fr.eye(-1)


def test_identity_negative_n_raises_valueerror():
    """``np.identity(-2)`` raises ValueError, not OverflowError
    (numpy/_core/numeric.py identity -> eye -> zeros((n, n))).

    Expected (numpy): ValueError.  Actual (ferray): OverflowError.
    """
    with pytest.raises(ValueError):
        np.identity(-2)  # oracle TYPE
    with pytest.raises(ValueError):
        fr.identity(-2)


# ---------------------------------------------------------------------------
# linspace: negative num exception TYPE + retstep kwarg + integer dtype
# ---------------------------------------------------------------------------


def test_linspace_negative_num_raises_valueerror():
    """numpy/_core/function_base.py:124-126 linspace:
        ``if num < 0: raise ValueError("Number of samples, {num}, must be
        non-negative.")``

    Expected (numpy): ValueError.  Actual (ferray): OverflowError
    ("can't convert negative int to unsigned") — num is bound as ``usize``,
    so PyO3 conversion fails before ferray's own check.
    [binding] num should be bound as a signed int and validated.
    """
    with pytest.raises(ValueError):
        np.linspace(0, 1, -1)  # oracle TYPE (function_base.py:125-126)
    with pytest.raises(ValueError):
        fr.linspace(0, 1, -1)


def test_linspace_retstep_returns_tuple():
    """numpy/_core/function_base.py:28 + :185-188 linspace signature includes
    ``retstep=False``; when True it returns ``(samples, step)``:
        ``if retstep: return y, step``  (function_base.py:185-186)

    Expected (numpy): linspace(0,1,5,retstep=True) -> (ndarray, 0.25).
    Actual (ferray): TypeError — ``retstep`` is not an accepted kwarg.
    [binding] missing kwarg on the linspace signature.
    """
    samples_oracle, step_oracle = np.linspace(0.0, 1.0, 5, retstep=True)
    result = fr.linspace(0.0, 1.0, 5, retstep=True)
    assert isinstance(result, tuple) and len(result) == 2, (
        f"expected (samples, step) tuple, got {type(result)}"
    )
    samples, step = result
    assert np.array_equal(samples, samples_oracle)
    assert step == step_oracle


def test_linspace_integer_dtype_floors_toward_neg_inf():
    """numpy/_core/function_base.py:181-182 linspace: an integer ``dtype`` is
    accepted and the result is floored toward -inf:
        ``if integer_dtype: _nx.floor(y, out=y)``

    So ``np.linspace(-0.5, -5.5, 6, dtype='int64')`` is
    ``[-1, -2, -3, -4, -5, -6]`` (int64).

    Expected (numpy): int64 array, floored toward -inf.
    Actual (ferray): TypeError ('unsupported dtype for floating-point op:
    "int64"') — linspace dispatches over float dtypes only.
    [binding] linspace uses match_dtype_float!; numpy accepts any dtype.
    """
    oracle = np.linspace(-0.5, -5.5, 6, dtype="int64")
    got = fr.linspace(-0.5, -5.5, 6, dtype="int64")
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle), (
        f"numpy={oracle.tolist()} ferray={got.tolist()}"
    )


def test_logspace_integer_dtype_supported():
    """numpy logspace accepts an integer ``dtype`` (astype at the end):
    ``np.logspace(0, 2, 3, dtype='int64')`` is ``[1, 10, 100]`` int64.

    Expected (numpy): int64 [1, 10, 100].
    Actual (ferray): TypeError — logspace dispatches over float dtypes only.
    [binding] logspace uses match_dtype_float!; numpy accepts any dtype.
    """
    oracle = np.logspace(0, 2, 3, dtype="int64")
    got = fr.logspace(0, 2, 3, dtype="int64")
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


def test_geomspace_integer_dtype_supported():
    """numpy/_core/function_base.py:453 geomspace ends with
    ``return result.astype(dtype, copy=False)``, so an integer ``dtype`` is
    accepted: ``np.geomspace(1, 100, 3, dtype='int64')`` is ``[1, 10, 100]``.

    Expected (numpy): int64 [1, 10, 100].
    Actual (ferray): TypeError — geomspace dispatches over float dtypes only.
    [binding] geomspace uses match_dtype_float!; numpy accepts any dtype.
    """
    oracle = np.geomspace(1, 100, 3, dtype="int64")
    got = fr.geomspace(1, 100, 3, dtype="int64")
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


# ---------------------------------------------------------------------------
# *_like: float16 dtype must be preserved
# [binding] creation_dispatch omits float16 from the dtype match
# ---------------------------------------------------------------------------


def test_zeros_like_preserves_float16_dtype():
    """``np.zeros_like`` inherits the source dtype, including float16.

    Expected (numpy): float16 zeros.
    Actual (ferray): TypeError ('unsupported dtype: "float16"') — the
    binding's dtype dispatch omits float16.
    """
    src = np.zeros(2, dtype=np.float16)
    oracle = np.zeros_like(src)
    got = fr.zeros_like(src)
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


def test_zeros_float16_dtype_supported():
    """``np.zeros(2, dtype='float16')`` produces a float16 array; numpy
    supports float16 across the creation surface.

    Expected (numpy): float16.
    Actual (ferray): TypeError ('unsupported dtype: "float16"').
    [binding] creation_dispatch omits float16.
    """
    oracle = np.zeros(2, dtype="float16")
    got = fr.zeros(2, dtype="float16")
    assert got.dtype == oracle.dtype, (
        f"numpy dtype={oracle.dtype} ferray dtype={got.dtype}"
    )
    assert np.array_equal(got, oracle)


# ---------------------------------------------------------------------------
# geomspace: negative endpoints must be bit-exact (sign rotation + pinning)
# [library] ferray-core geomspace lacks out_sign rotation + endpoint pinning
# ---------------------------------------------------------------------------


def test_geomspace_negative_endpoints_exact():
    """numpy/_core/function_base.py:432-448 geomspace rotates ``start`` to a
    positive real (``out_sign = sign(start); start /= out_sign``), computes in
    log space, then forces the endpoints back exactly:
        ``result[0] = start``    (function_base.py:444)
        ``result[-1] = stop``    (function_base.py:446)

    So ``np.geomspace(-1, -1000, 4)`` is EXACT ``[-1, -10, -100, -1000]``.

    Expected (numpy): exact (endpoints bit-equal; interior -10.0/-100.0 exact).
    Actual (ferray): [-1, -9.999999999999998, -99.99999999999996,
    -999.9999999999998] — no sign rotation / endpoint pinning.
    """
    oracle = np.geomspace(-1.0, -1000.0, 4)
    got = fr.geomspace(-1.0, -1000.0, 4)
    assert got.dtype == oracle.dtype
    assert got.shape == oracle.shape
    assert got[0] == oracle[0], f"start: numpy={oracle[0]!r} ferray={got[0]!r}"
    assert got[-1] == oracle[-1], (
        f"stop: numpy={oracle[-1]!r} ferray={got[-1]!r}"
    )
    assert got[1] == oracle[1], (
        f"interior: numpy={oracle[1]!r} ferray={got[1]!r}"
    )
