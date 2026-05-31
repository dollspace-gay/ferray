"""Complex REDUCTION expansion — sum/prod/mean/var/std/cumsum/cumprod/average
(#925, REQ-6 of .design/ferray-core-complex.md).

The prior ferray-python binding made every reduction real-only, with two
failure classes on complex input (both verified live, numpy 2.4.4):

  * ``sum``/``prod``/``cumsum``/``cumprod`` dispatched ``match_dtype_numeric!``
    (no complex arm) and so raised ``TypeError`` — numpy COMPUTES these as
    complex scalars/arrays.
  * ``mean``/``var``/``std``/``average`` coerced the input to ``float64`` ahead
    of dispatch, and numpy's complex->real cast DROPS the imaginary part with a
    ``ComplexWarning`` — an active R-CODE-4 boundary corruption
    (``fr.mean([1+2j,3-1j])`` returned ``1.0``; numpy returns ``(2+0.5j)``).

The complex ``ReduceAcc`` (``ferray-core/src/array/reductions.rs``, ``Acc =
Self``) already folds complex; ``mean`` is composed as complex-sum/count via a
reciprocal-multiply (ma #873), and ``var``/``std`` return the REAL
``mean(|x - mean|^2)`` (numpy's ``absolute(danom)**2``).

Every EXPECTED value here is taken from a LIVE ``numpy`` call (R-CHAR-3); none
is literal-copied from the ferray side. The whole point is that ``fr.*`` must
equal ``np.*`` AND must not emit a ``ComplexWarning`` for the corrupting ops.
"""

import warnings

import numpy as np
import pytest

import ferray as fr

# A few non-trivial complex vectors (distinct real/imag, mixed signs) so a
# real-only collapse, a dropped-imag, or a swapped component all show up.
C128 = np.array([1 + 2j, 3 + 4j, 5 - 1j, 2 + 0j], dtype="complex128")
C64 = C128.astype("complex64")


def _no_complex_warning(fn):
    """Call `fn` with ComplexWarning promoted to an error, return its result.

    numpy's complex->real cast raises ``ComplexWarning`` ("Casting complex
    values to real discards the imaginary part"); the corruption guard asserts
    the reduction does NOT take that path.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", np.exceptions.ComplexWarning)
        return fn()


# ---------------------------------------------------------------------------
# sum / prod — complex fold via ReduceAcc, width preserved.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_sum_complex_matches_numpy_keeps_width(dtype):
    z = C128.astype(dtype)
    got = fr.sum(z)
    want = np.sum(z)
    assert np.asarray(got).dtype == want.dtype  # c64->c64, c128->c128
    assert np.allclose(np.asarray(got), want)


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_prod_complex_matches_numpy_keeps_width(dtype):
    z = C128.astype(dtype)
    got = fr.prod(z)
    want = np.prod(z)
    assert np.asarray(got).dtype == want.dtype
    assert np.allclose(np.asarray(got), want)


def test_sum_complex_exact_value():
    # Derived live: np.sum([1+2j,3+4j]) == (4+6j).
    got = fr.sum(fr.array([1 + 2j, 3 + 4j]))
    assert complex(np.asarray(got)) == complex(np.sum([1 + 2j, 3 + 4j]))


def test_prod_complex_exact_value():
    # Derived live: np.prod([1+2j,3+4j]) == (-5+10j).
    got = fr.prod(fr.array([1 + 2j, 3 + 4j]))
    assert complex(np.asarray(got)) == complex(np.prod([1 + 2j, 3 + 4j]))


# ---------------------------------------------------------------------------
# mean — the headline R-CODE-4 corruption: imag must NOT be discarded.
# ---------------------------------------------------------------------------


def test_mean_complex_no_imag_discard_matches_numpy():
    z = fr.array([1 + 2j, 3 - 1j])
    got = _no_complex_warning(lambda: fr.mean(z))
    want = np.mean([1 + 2j, 3 - 1j])  # (2+0.5j), live
    assert complex(np.asarray(got)) == complex(want)
    # The corruption was returning the real-only 1.0; assert imag survived.
    assert np.asarray(got).imag != 0.0


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_mean_complex_keeps_width(dtype):
    z = C128.astype(dtype)
    got = _no_complex_warning(lambda: fr.mean(z))
    want = np.mean(z)
    # numpy MAIN-array mean preserves the complex width (unlike numpy.ma).
    assert np.asarray(got).dtype == want.dtype
    assert np.allclose(np.asarray(got), want)


def test_mean_complex_bit_exact_reciprocal_multiply():
    # numpy's `dsum / cnt` for a real divisor is a reciprocal-MULTIPLY; the
    # binding must match the exact bits (ma #873). Derive the bits live.
    data = [1 + 2j, 3 - 1j, 2 + 0j, 4 + 4j]
    got = complex(np.asarray(_no_complex_warning(lambda: fr.mean(fr.array(data)))))
    want = complex(np.mean(data))
    assert got.real.hex() == want.real.hex()
    assert got.imag.hex() == want.imag.hex()


# ---------------------------------------------------------------------------
# var / std — REAL result, width-following (c64->f32, c128->f64).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_var_complex_real_result_matches_numpy(dtype):
    z = C128.astype(dtype)
    got = _no_complex_warning(lambda: fr.var(z))
    want = np.var(z)
    assert np.asarray(got).dtype == want.dtype  # c64->float32, c128->float64
    assert np.allclose(np.asarray(got), want)


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_std_complex_real_result_matches_numpy(dtype):
    z = C128.astype(dtype)
    got = _no_complex_warning(lambda: fr.std(z))
    want = np.std(z)
    assert np.asarray(got).dtype == want.dtype
    assert np.allclose(np.asarray(got), want)


def test_var_complex_ddof_matches_numpy():
    z = np.array([1.5 + 2.5j, 3.5 - 1.5j, 0.5 + 0.5j, 4.5 + 4.5j], dtype="complex128")
    got = _no_complex_warning(lambda: fr.var(z, ddof=1))
    assert np.allclose(np.asarray(got), np.var(z, ddof=1))


def test_var_complex_value_is_real_not_collapsed():
    # The corruption returned var of the real-only collapse; assert the value
    # equals numpy's complex variance (which uses both components).
    z = fr.array([1 + 2j, 3 + 4j])
    got = _no_complex_warning(lambda: fr.var(z))
    assert float(np.asarray(got)) == float(np.var([1 + 2j, 3 + 4j]))


# ---------------------------------------------------------------------------
# cumsum / cumprod — complex fold, width + shape preserved.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_cumsum_complex_matches_numpy(dtype):
    z = C128.astype(dtype)
    got = fr.cumsum(z)
    want = np.cumsum(z)
    assert np.asarray(got).dtype == want.dtype
    assert np.asarray(got).shape == want.shape
    assert np.allclose(np.asarray(got), want)


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_cumprod_complex_matches_numpy(dtype):
    z = C128.astype(dtype)
    got = fr.cumprod(z)
    want = np.cumprod(z)
    assert np.asarray(got).dtype == want.dtype
    assert np.allclose(np.asarray(got), want)


# ---------------------------------------------------------------------------
# average — no-weights == mean; weighted promotes to complex128.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_average_complex_no_weights_matches_numpy(dtype):
    z = C128.astype(dtype)
    got = _no_complex_warning(lambda: fr.average(z))
    want = np.average(z)
    assert np.asarray(got).dtype == want.dtype
    assert np.allclose(np.asarray(got), want)


def test_average_complex_weighted_matches_numpy():
    z = np.array([1 + 2j, 3 + 4j, 5 - 1j], dtype="complex128")
    w = np.array([1.0, 2.0, 3.0])
    got = _no_complex_warning(lambda: fr.average(z, weights=w))
    want = np.average(z, weights=w)
    assert np.allclose(np.asarray(got), want)
    assert np.asarray(got).imag != 0.0  # imag survived the weighting


def test_average_complex64_weighted_promotes_to_complex128():
    # numpy promotes a weighted complex64 average to complex128 (live, 2.4.4).
    z = np.array([1 + 2j, 3 + 4j, 5 - 1j], dtype="complex64")
    w = np.array([1.0, 2.0, 3.0])
    got = _no_complex_warning(lambda: fr.average(z, weights=w))
    want = np.average(z, weights=w)
    assert np.asarray(got).dtype == want.dtype == np.dtype("complex128")
    assert np.allclose(np.asarray(got), want)


# ---------------------------------------------------------------------------
# axis= on 2-D complex.
# ---------------------------------------------------------------------------


Z2 = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype="complex128")


def test_sum_complex_axis0_matches_numpy():
    got = fr.sum(Z2, axis=0)
    want = np.sum(Z2, axis=0)
    assert np.asarray(got).shape == want.shape
    assert np.allclose(np.asarray(got), want)


def test_mean_complex_axis1_matches_numpy():
    got = _no_complex_warning(lambda: fr.mean(Z2, axis=1))
    want = np.mean(Z2, axis=1)  # [2+3j, 6+7j], live
    assert np.asarray(got).shape == want.shape
    assert np.allclose(np.asarray(got), want)


def test_var_complex_axis0_matches_numpy():
    got = _no_complex_warning(lambda: fr.var(Z2, axis=0))
    want = np.var(Z2, axis=0)
    assert np.asarray(got).shape == want.shape
    assert np.allclose(np.asarray(got), want)


def test_prod_complex_axis1_matches_numpy():
    got = fr.prod(Z2, axis=1)
    want = np.prod(Z2, axis=1)
    assert np.allclose(np.asarray(got), want)


# ---------------------------------------------------------------------------
# Real reductions UNCHANGED (no regression from the complex arms).
# ---------------------------------------------------------------------------

REAL_F64 = np.array([1.0, 2.0, 3.0, 4.0])
REAL_I64 = np.array([1, 2, 3, 4], dtype="int64")


def test_real_sum_unregressed():
    assert float(np.asarray(fr.sum(REAL_F64))) == float(np.sum(REAL_F64))
    assert int(np.asarray(fr.sum(REAL_I64))) == int(np.sum(REAL_I64))


def test_real_prod_unregressed():
    assert float(np.asarray(fr.prod(REAL_F64))) == float(np.prod(REAL_F64))


def test_real_mean_unregressed():
    assert float(np.asarray(fr.mean(REAL_F64))) == float(np.mean(REAL_F64))
    # int mean promotes to float64.
    got = fr.mean(REAL_I64)
    assert np.asarray(got).dtype == np.dtype("float64")
    assert float(np.asarray(got)) == float(np.mean(REAL_I64))


def test_real_var_std_unregressed():
    assert np.allclose(np.asarray(fr.var(REAL_F64)), np.var(REAL_F64))
    assert np.allclose(np.asarray(fr.std(REAL_F64)), np.std(REAL_F64))


def test_real_cumsum_cumprod_unregressed():
    assert np.allclose(np.asarray(fr.cumsum(REAL_F64)), np.cumsum(REAL_F64))
    assert np.allclose(np.asarray(fr.cumprod(REAL_F64)), np.cumprod(REAL_F64))


def test_real_average_unregressed():
    assert float(np.asarray(fr.average(REAL_F64))) == float(np.average(REAL_F64))
    w = np.array([1.0, 1.0, 1.0, 2.0])
    assert np.allclose(np.asarray(fr.average(REAL_F64, weights=w)), np.average(REAL_F64, weights=w))
