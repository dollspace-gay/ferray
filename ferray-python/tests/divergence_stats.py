"""Adversarial divergence pins for ferray-python/src/stats.rs vs NumPy 2.4.

Authored by the acto-critic re-audit of `ferray-python/src/stats.rs`. Each
test computes the NumPy oracle LIVE (R-CHAR-3: no expected value is copied
from the ferray side) and asserts that ferray matches value, dtype, or
exception type. Every test here is EXPECTED TO FAIL on the current ferray
build; that failure pins a real NumPy divergence for the fixer.

Oracle: ``import numpy as np`` (installed numpy 2.4.x = source of truth).
Target: ``import ferray as fr``.

NOTE ON #764 (empty reductions): a prior fix made ``mean/var/std/median``
of an empty array return ``nan`` (matching numpy's warn-and-return, not
raise). That fix is VERIFIED CORRECT by this re-audit and is therefore NOT
re-pinned here (re-pinning already-fixed behavior would be a tautology).
``sum([]) -> 0.0`` and ``prod([]) -> 1.0`` were also verified correct.

NumPy emits RuntimeWarnings for degenerate reductions: that is numpy
RETURNING a value (nan / inf), NOT raising. ferray must return the value too.
"""

import warnings

import numpy as np
import pytest

import ferray as fr


def _np_value(fn):
    """Evaluate a numpy reduction, swallowing the RuntimeWarning numpy emits
    while still capturing the value it returns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn()


# ---------------------------------------------------------------------------
# ddof >= n: numpy clamps `rcount - ddof` to 0 and divides -> +inf (+ warning);
# it does NOT raise. numpy/_core/_methods.py:204
#   `rcount = um.maximum(rcount - ddof, 0)`  then  true_divide -> inf.
# (the warning is numpy/_core/_methods.py:154-156). ferray raises ValueError.
# ---------------------------------------------------------------------------


def test_var_ddof_ge_n_returns_inf_not_valueerror():
    """numpy/_core/_methods.py:204 clamps dof to 0 then divides -> inf when
    ddof >= n. Expected: inf. Actual: ferray raises ValueError
    ('ddof >= number of elements')."""
    expected = _np_value(lambda: np.var(np.array([1.0, 2.0, 3.0]), ddof=3))
    assert np.isinf(expected)
    got = fr.var(fr.array([1.0, 2.0, 3.0]), ddof=3)
    assert np.isinf(np.asarray(got))


def test_std_ddof_ge_n_returns_inf_not_valueerror():
    """numpy/_core/_methods.py:217-227 (_std = sqrt of _var) -> inf for ddof>=n.
    Expected: inf. Actual: ferray raises ValueError."""
    expected = _np_value(lambda: np.std(np.array([1.0, 2.0, 3.0]), ddof=5))
    assert np.isinf(expected)
    got = fr.std(fr.array([1.0, 2.0, 3.0]), ddof=5)
    assert np.isinf(np.asarray(got))


# ---------------------------------------------------------------------------
# Integer accumulator promotion. numpy promotes a sub-platform-int input to
# the default platform integer (int64 signed / uint64 unsigned) for the
# accumulator. numpy/_core/fromnumeric.py:2325-2327 (sum):
#   "if `a` is signed then the platform integer is used while if `a` is
#    unsigned then an unsigned integer of the same precision ... is used".
#
# ferray keeps the narrow input dtype. Two observable consequences:
#   (a) WRONG RESULT DTYPE even when no overflow occurs, and
#   (b) on overflow ferray PANICS (debug overflow check) where numpy returns
#       the correct widened value.
# ---------------------------------------------------------------------------


def test_sum_int8_result_dtype_is_int64():
    """numpy/_core/fromnumeric.py:2325 — int8 sum accumulates in the platform
    integer (int64). No overflow here (1+2+3=6); only the dtype diverges.
    Expected dtype int64. ferray returns int8."""
    expected = np.sum(np.array([1, 2, 3], dtype=np.int8))
    assert expected.dtype == np.int64
    got = np.asarray(fr.sum(fr.array([1, 2, 3], dtype="int8")))
    assert got.dtype == expected.dtype
    assert int(got) == int(expected)


def test_sum_uint8_result_dtype_is_uint64():
    """numpy/_core/fromnumeric.py:2326-2327 — unsigned sub-platform int sum
    accumulates in uint64. Expected dtype uint64. ferray returns uint8."""
    expected = np.sum(np.array([1, 2, 3], dtype=np.uint8))
    assert expected.dtype == np.uint64
    got = np.asarray(fr.sum(fr.array([1, 2, 3], dtype="uint8")))
    assert got.dtype == expected.dtype
    assert int(got) == int(expected)


def test_sum_int8_promotes_and_does_not_overflow():
    """numpy/_core/fromnumeric.py:2325 — int8 sum [100,100] accumulates in
    int64 -> 200. Expected 200 (int64). ferray PANICS (attempt to add with
    overflow, ferray-stats/src/parallel.rs) because it keeps int8."""
    expected = np.sum(np.array([100, 100], dtype=np.int8))
    assert int(expected) == 200
    assert expected.dtype == np.int64
    got = np.asarray(fr.sum(fr.array([100, 100], dtype="int8")))
    assert int(got) == 200
    assert got.dtype == np.int64


def test_prod_int8_result_dtype_is_int64():
    """numpy/_core/fromnumeric.py:2692-2693 — prod of a sub-platform int
    promotes the accumulator to the platform int. Expected dtype int64.
    ferray keeps int8 (value 24 is correct but dtype is wrong)."""
    expected = np.prod(np.array([2, 3, 4], dtype=np.int8))
    assert expected.dtype == np.int64
    got = np.asarray(fr.prod(fr.array([2, 3, 4], dtype="int8")))
    assert got.dtype == expected.dtype
    assert int(got) == int(expected)


def test_cumsum_int8_promotes_and_does_not_overflow():
    """numpy/_core/fromnumeric.py:2854-2855 — cumsum of a sub-platform int
    promotes to the platform int. Expected [100, 200] int64. ferray keeps
    int8 and PANICS (attempt to add with overflow,
    ferray-ufunc/src/ops/arithmetic.rs)."""
    expected = np.cumsum(np.array([100, 100], dtype=np.int8))
    assert expected.dtype == np.int64
    np.testing.assert_array_equal(np.asarray(expected), [100, 200])
    got = np.asarray(fr.cumsum(fr.array([100, 100], dtype="int8")))
    assert got.dtype == np.int64
    np.testing.assert_array_equal(got, [100, 200])


# ---------------------------------------------------------------------------
# bool reductions: numpy sums/products bools in the platform integer.
# numpy/_core/fromnumeric.py:2325-2327 (the same promotion rule covers bool,
# treated as an integer of less precision than the platform integer).
# ferray raises TypeError ('unsupported dtype for numeric op: "bool"').
# ---------------------------------------------------------------------------


def test_sum_bool_returns_int64():
    """numpy/_core/fromnumeric.py:2325 — bool sum accumulates in the platform
    integer. Expected int64 value 3. ferray raises TypeError (bool unsupported)."""
    expected = np.sum(np.array([True, True, False, True]))
    assert int(expected) == 3
    assert expected.dtype == np.int64
    got = np.asarray(fr.sum(fr.array([True, True, False, True])))
    assert int(got) == 3
    assert got.dtype == np.int64


def test_prod_bool_returns_int64():
    """numpy/_core/fromnumeric.py:2692-2693 — bool prod accumulates in the
    platform integer. Expected int64 value 1. ferray raises TypeError."""
    expected = np.prod(np.array([True, True]))
    assert int(expected) == 1
    assert expected.dtype == np.int64
    got = np.asarray(fr.prod(fr.array([True, True])))
    assert int(got) == 1
    assert got.dtype == np.int64


# ---------------------------------------------------------------------------
# percentile / quantile accept array-like q and return an array of results.
# numpy/lib/_function_base_impl.py:4083 (percentile) and :4284 (quantile):
#   "q : array_like of float".
# ferray's binding signature only accepts a scalar f64 -> TypeError on list q.
# ---------------------------------------------------------------------------


def test_percentile_accepts_sequence_q():
    """numpy/lib/_function_base_impl.py:4083 'q : array_like of float'. A list q
    returns an array of percentiles. Expected [1.75, 2.5, 3.25]. ferray raises
    TypeError ('argument q: must be real number, not list')."""
    expected = np.percentile(np.array([1.0, 2.0, 3.0, 4.0]), [25, 50, 75])
    got = np.asarray(fr.percentile(fr.array([1.0, 2.0, 3.0, 4.0]), [25, 50, 75]))
    np.testing.assert_allclose(got, expected)


def test_quantile_accepts_sequence_q():
    """numpy/lib/_function_base_impl.py:4284 'q : array_like of float' mirrors
    percentile. A list q returns an array. Expected [1.75, 2.5, 3.25]. ferray
    raises TypeError on a list q."""
    expected = np.quantile(np.array([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75])
    got = np.asarray(fr.quantile(fr.array([1.0, 2.0, 3.0, 4.0]), [0.25, 0.5, 0.75]))
    np.testing.assert_allclose(got, expected)


# ---------------------------------------------------------------------------
# Axis out of bounds: numpy raises numpy.exceptions.AxisError, which is a
# subclass of BOTH ValueError and IndexError.
# numpy/exceptions.py:108 — `class AxisError(ValueError, IndexError)`.
# ferray raises a plain builtins.ValueError, so a caller doing
# `except IndexError` or `except np.exceptions.AxisError` would NOT catch it.
# ---------------------------------------------------------------------------


def test_sum_axis_out_of_bounds_raises_axiserror():
    """numpy/exceptions.py:108 — out-of-bounds axis raises AxisError
    (subclass of ValueError AND IndexError). ferray raises plain ValueError,
    which fails `isinstance(err, IndexError)`."""
    src = np.array([1, 2, 3])
    with pytest.raises(np.exceptions.AxisError):
        np.sum(src, axis=5)
    with pytest.raises(np.exceptions.AxisError):
        fr.sum(fr.array([1, 2, 3]), axis=5)


# ---------------------------------------------------------------------------
# Divergence: ferray-python `isin` is missing the `invert=` / `kind=` kwargs.
#
# numpy: `isin(element, test_elements, assume_unique=False, invert=False, *,
# kind=None)`  (numpy/lib/_arraysetops_impl.py:959). With `invert=True` numpy
# returns the LOGICAL NEGATION of the membership mask, still shape-preserving
# (numpy/lib/_arraysetops_impl.py:1076 `in1d(..., invert=invert,
# kind=kind).reshape(element.shape)`).
#
# ferray `pub fn isin` signature is `(element, test_elements,
# assume_unique=False)` (ferray-python/src/stats.rs:3654-3661) — neither
# `invert=` nor `kind=` exists, so the call raises
# `TypeError: isin() got an unexpected keyword argument 'invert'` where numpy
# computes the inverted bool array. Oracle live (numpy 2.4.4):
#   np.isin([1,2,3,4],[2,4],invert=True) == [True,False,True,False]
#   np.isin([[1,2],[3,4]],[2,4],invert=True) == [[True,False],[True,False]]
# Tracking: #<crosslink>
# ---------------------------------------------------------------------------
def test_divergence_isin_invert_1d():
    element = np.array([1, 2, 3, 4])
    test = np.array([2, 4])
    expected = np.isin(element, test, invert=True)  # live oracle (R-CHAR-3)
    got = fr.isin(element, test, invert=True)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(got, expected)


def test_divergence_isin_invert_2d_shape():
    element = np.array([[1, 2], [3, 4]])
    test = np.array([2, 4])
    expected = np.isin(element, test, invert=True)  # live oracle (R-CHAR-3)
    got = fr.isin(element, test, invert=True)
    assert got.shape == expected.shape == (2, 2)
    np.testing.assert_array_equal(got, expected)


def test_divergence_isin_kind_kwarg():
    # numpy accepts `kind=` ('sort'/'table'/None); ferray rejects it. The
    # oracle result is identical to the default-kind membership; ferray raises
    # TypeError on the kwarg, which is the divergence being pinned.
    element = np.array([1, 2, 3, 4])
    test = np.array([2, 4])
    expected = np.isin(element, test, kind="sort")  # live oracle (R-CHAR-3)
    got = fr.isin(element, test, kind="sort")
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# REQ-21 (#992): a FULL reduction (axis=None / inherently scalar) must return a
# numpy SCALAR (numpy.int64 / float64 / bool_ / complex128), NOT a 0-d
# numpy.ndarray — numpy's `$OUT_SCALAR` contract
# (numpy/_core/code_generators/ufunc_docstrings.py: `'OUT_SCALAR_1'`). numpy
# realises this by returning the bare `ufunc.reduce(...)` / method-reduce result
# (numpy/_core/fromnumeric.py:83 for sum/prod/min/max/ptp; :2539/:2634 for
# any/all; numpy/_core/_methods.py for mean/var/std/amin/amax;
# numpy/lib/_function_base_impl.py for median/average/percentile/quantile), and a
# 0-d reduce auto-boxes to a numpy scalar. An AXIS reduction that leaves >= 1
# dimension stays an ndarray. The oracle type/value are computed LIVE (R-CHAR-3).
# ---------------------------------------------------------------------------

# (name, ferray fn, numpy fn, input) for the representative scalar-collapse set.
# Each input is chosen so the full reduction is well-defined and the numpy oracle
# returns a genuine scalar.
_REQ21_FULL_REDUCTIONS = [
    ("sum", fr.sum, np.sum, [1, 2, 3]),
    ("prod", fr.prod, np.prod, [1, 2, 3, 4]),
    ("mean", fr.mean, np.mean, [1.0, 2.0, 3.0]),
    ("max", fr.max, np.max, [3, 1, 2]),
    ("min", fr.min, np.min, [3, 1, 2]),
    ("argmax", fr.argmax, np.argmax, [3, 1, 2]),
    ("argmin", fr.argmin, np.argmin, [3, 1, 2]),
    ("all", fr.all, np.all, [True, True, False]),
    ("any", fr.any, np.any, [False, True, False]),
    ("ptp", fr.ptp, np.ptp, [3, 1, 7, 2]),
    ("var", fr.var, np.var, [1.0, 2.0, 3.0, 4.0]),
    ("std", fr.std, np.std, [1.0, 2.0, 3.0, 4.0]),
    ("median", fr.median, np.median, [1.0, 2.0, 3.0, 4.0]),
    ("average", fr.average, np.average, [1.0, 2.0, 3.0]),
    ("count_nonzero", fr.count_nonzero, np.count_nonzero, [0, 1, 2, 0, 3]),
    ("nansum", fr.nansum, np.nansum, [1.0, np.nan, 3.0]),
    ("nanmean", fr.nanmean, np.nanmean, [1.0, np.nan, 3.0]),
    ("nanmax", fr.nanmax, np.nanmax, [1.0, np.nan, 3.0]),
    ("nanargmax", fr.nanargmax, np.nanargmax, [1.0, np.nan, 3.0]),
]


@pytest.mark.parametrize(
    "name,frfn,npfn,x",
    _REQ21_FULL_REDUCTIONS,
    ids=[t[0] for t in _REQ21_FULL_REDUCTIONS],
)
def test_req21_full_reduction_returns_numpy_scalar(name, frfn, npfn, x):
    expected = _np_value(lambda: npfn(x))  # live oracle (R-CHAR-3)
    got = _np_value(lambda: frfn(x))
    # 1. NOT a 0-d ndarray — the $OUT_SCALAR contract.
    assert not isinstance(got, np.ndarray), f"{name}: got a {type(got)}, want a numpy scalar"
    # 2. Exact numpy scalar TYPE parity with the oracle.
    assert type(got) is type(expected), f"{name}: {type(got)} != {type(expected)}"
    # 3. Value parity (NaN-aware) — the collapse must not perturb the value.
    np.testing.assert_array_equal(got, expected)


def test_req21_full_complex_reduction_is_complex128_scalar():
    # A full complex reduction collapses to a numpy.complex128 scalar, not a 0-d
    # complex ndarray. numpy: `type(np.sum([1+2j,3+4j])) is np.complex128`.
    x = [1 + 2j, 3 + 4j]
    expected = np.sum(x)  # live oracle (R-CHAR-3)
    got = fr.sum(x)
    assert not isinstance(got, np.ndarray)
    assert type(got) is type(expected) is np.complex128
    assert got == expected


def test_req21_full_complex_mean_is_complex128_scalar():
    x = [1 + 2j, 3 - 1j]
    expected = np.mean(x)  # live oracle (R-CHAR-3)
    got = fr.mean(x)
    assert not isinstance(got, np.ndarray)
    assert type(got) is type(expected) is np.complex128
    np.testing.assert_array_equal(got, expected)


def test_req21_axis_reduction_stays_ndarray():
    # An AXIS reduction that leaves >= 1 dimension must REMAIN an ndarray —
    # `scalarize` must be a no-op for ndim != 0. numpy:
    # `fr.sum([[1,2],[3,4]], axis=0) -> array([4, 6])`.
    x = [[1, 2], [3, 4]]
    expected = np.sum(x, axis=0)  # live oracle (R-CHAR-3)
    got = fr.sum(x, axis=0)
    assert isinstance(got, np.ndarray)
    assert got.shape == expected.shape == (2,)
    np.testing.assert_array_equal(got, expected)


def test_req21_percentile_scalar_q_is_scalar_axis_q_stays_ndarray():
    # A scalar-`q` full percentile collapses to a numpy float64 scalar; an axis
    # percentile (still scalar `q`) leaves >= 1 dim and stays an ndarray.
    x = [[1.0, 2.0], [3.0, 4.0]]
    exp_full = np.percentile(x, 50)  # live oracle (R-CHAR-3)
    got_full = fr.percentile(x, 50)
    assert not isinstance(got_full, np.ndarray)
    assert type(got_full) is type(exp_full) is np.float64
    np.testing.assert_array_equal(got_full, exp_full)

    exp_axis = np.percentile(x, 50, axis=0)  # live oracle (R-CHAR-3)
    got_axis = fr.percentile(x, 50, axis=0)
    assert isinstance(got_axis, np.ndarray)
    assert got_axis.shape == exp_axis.shape == (2,)
    np.testing.assert_array_equal(got_axis, exp_axis)


# ---------------------------------------------------------------------------
# Divergence (#1006): the ferray-python nan-family reductions reject the
# `keepdims=` kwarg entirely, while numpy accepts it on ALL of them.
#
# numpy: every nan-reduction takes `keepdims` (numpy/_nanfunctions_impl.py —
# e.g. `nansum(a, axis=None, dtype=None, out=None, keepdims=<no value>, ...)`
# at the nansum def; same for nanprod/nanmean/nanmin/nanmax/nanvar/nanstd/
# nanmedian/nanargmin/nanargmax/nanpercentile/nanquantile). With
# `keepdims=True` a FULL reduction returns an ndarray whose reduced dims are
# all 1 (NOT a scalar) — this is exactly the REQ-21 edge case #2 (keepdims
# must NOT collapse to a scalar), but ferray cannot even reach the path:
#
# ferray `bind_nan_reduction!` macro signature is `(a, axis = None)`
# (ferray-python/src/stats.rs:2073, :2104, :2152) — NO `keepdims` parameter —
# so `fr.nansum(x, keepdims=True)` raises
# `TypeError: nansum() got an unexpected keyword argument 'keepdims'`.
#
# Pre-existing ABI gap, INDEPENDENT of REQ-21 (#992 only wrapped egresses in
# conv::scalarize); the REQ-21 scalar-return behaviour itself is correct.
# Oracle live (numpy 2.4.4):
#   np.nansum([[1.,2.],[3.,nan]], keepdims=True) == array([[6.]]), shape (1,1)
#   np.nansum([1.,nan,3.], keepdims=True)        == array([4.]),  shape (1,)
# Tracking: #1006
# ---------------------------------------------------------------------------

_NAN_FAMILY_KEEPDIMS = [
    ("nansum", lambda x, **k: fr.nansum(x, **k), lambda x, **k: np.nansum(x, **k)),
    ("nanprod", lambda x, **k: fr.nanprod(x, **k), lambda x, **k: np.nanprod(x, **k)),
    ("nanmean", lambda x, **k: fr.nanmean(x, **k), lambda x, **k: np.nanmean(x, **k)),
    ("nanmin", lambda x, **k: fr.nanmin(x, **k), lambda x, **k: np.nanmin(x, **k)),
    ("nanmax", lambda x, **k: fr.nanmax(x, **k), lambda x, **k: np.nanmax(x, **k)),
    ("nanvar", lambda x, **k: fr.nanvar(x, **k), lambda x, **k: np.nanvar(x, **k)),
    ("nanstd", lambda x, **k: fr.nanstd(x, **k), lambda x, **k: np.nanstd(x, **k)),
    ("nanmedian", lambda x, **k: fr.nanmedian(x, **k), lambda x, **k: np.nanmedian(x, **k)),
    ("nanargmin", lambda x, **k: fr.nanargmin(x, **k), lambda x, **k: np.nanargmin(x, **k)),
    ("nanargmax", lambda x, **k: fr.nanargmax(x, **k), lambda x, **k: np.nanargmax(x, **k)),
    ("nanpercentile",
     lambda x, **k: fr.nanpercentile(x, 50, **k),
     lambda x, **k: np.nanpercentile(x, 50, **k)),
    ("nanquantile",
     lambda x, **k: fr.nanquantile(x, 0.5, **k),
     lambda x, **k: np.nanquantile(x, 0.5, **k)),
]


@pytest.mark.parametrize(
    "name,frfn,npfn",
    _NAN_FAMILY_KEEPDIMS,
    ids=[t[0] for t in _NAN_FAMILY_KEEPDIMS],
)
def test_divergence_nan_family_rejects_keepdims_kwarg(name, frfn, npfn):
    # 2-D input so a full keepdims reduction yields a (1, 1) ndarray oracle.
    x = [[1.0, 2.0], [3.0, np.nan]]
    expected = _np_value(lambda: npfn(np.array(x), keepdims=True))  # live (R-CHAR-3)
    # numpy keeps the reduced dims as 1s -> an ndarray, never a scalar.
    assert isinstance(expected, np.ndarray)
    assert expected.shape == (1, 1)
    # ferray must accept keepdims= and likewise return a (1, 1) ndarray.
    got = _np_value(lambda: frfn(fr.array(x), keepdims=True))
    assert isinstance(got, np.ndarray), f"{name}: keepdims result must be an ndarray"
    assert got.shape == expected.shape == (1, 1)
    np.testing.assert_array_equal(got, expected)


def test_divergence_nansum_keepdims_1d_is_shape_one_ndarray():
    # REQ-21 edge #2 at 1-D: a full keepdims reduction stays an ndarray shape
    # (1,), NOT a scalar. numpy: np.nansum([1,nan,3], keepdims=True) -> [4.].
    x = [1.0, np.nan, 3.0]
    expected = _np_value(lambda: np.nansum(np.array(x), keepdims=True))  # live
    assert isinstance(expected, np.ndarray)
    assert expected.shape == (1,)
    got = _np_value(lambda: fr.nansum(fr.array(x), keepdims=True))
    assert isinstance(got, np.ndarray)
    assert got.shape == expected.shape == (1,)
    np.testing.assert_array_equal(got, expected)
