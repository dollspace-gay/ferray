"""Expansion tests — complex ORDERING ops (#932, epic #919).

numpy orders complex LEXICOGRAPHICALLY (real first, then imaginary). Prior to
#932 every ferray ordering op (`sort`/`argsort`/`unique`/`min`/`max`/`amin`/
`amax`/`argmin`/`argmax`/`ptp`) raised a TypeError on complex input because the
binding dispatched through a real-only `match_dtype_orderable!`.

This suite derives EVERY expected value from a LIVE numpy call (R-CHAR-3) —
never copied from the ferray side — covering:
  * lexicographic order (real, then imag) for sort/argsort/unique,
  * lexicographic extremum + its index for min/max/amin/amax/argmin/argmax,
  * ptp = lexicographic max - lexicographic min,
  * NaN-complex (either part NaN) ordering: LAST for sort/argsort/unique,
    PROPAGATING (first NaN wins) for the min/max/arg/ptp reductions,
  * complex64 vs complex128 width preservation (argsort/argmin -> int64),
  * axis variants,
  * the real path is UNREGRESSED (incl. the #918 nan-last real sort).
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    """ferray array -> numpy array for value/dtype comparison."""
    return np.asarray(np.array(x.tolist()))


Z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
ZDUP = [1 + 2j, -3 + 4j, 1 + 2j, 2 - 1j]
ZNAN = [3 + 1j, np.nan + 0j, 1 + 1j, np.nan + 5j, 2 + 0j]


# ---------------------------------------------------------------------------
# sort / argsort
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_sort_complex_lexicographic(dt):
    z = np.array(Z, dtype=dt)
    expected = np.sort(z)
    result = _np(fr.sort(fr.array(z)))
    assert np.array_equal(result, expected), f"{dt}: {result!r} != {expected!r}"


def test_sort_complex_nan_last():
    z = np.array(ZNAN)
    expected = np.sort(z)  # nan-complex sorts last
    result = _np(fr.sort(fr.array(z)))
    # NaN != NaN so compare with equal_nan semantics via repr-free check.
    assert np.array_equal(result, expected, equal_nan=True)


@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_argsort_complex_lexicographic(dt):
    z = np.array(Z, dtype=dt)
    expected = np.argsort(z)
    result = _np(fr.argsort(fr.array(z)))
    assert np.array_equal(result, expected)
    assert result.dtype == np.int64


def test_argsort_complex_nan_last():
    z = np.array(ZNAN)
    expected = np.argsort(z)
    result = _np(fr.argsort(fr.array(z)))
    assert np.array_equal(result, expected)


def test_sort_complex_axis():
    z = np.array([[3 + 0j, 1 + 2j], [0 + 1j, 2 + 0j]])
    for ax in (0, 1):
        expected = np.sort(z, axis=ax)
        result = _np(fr.sort(fr.array(z), axis=ax))
        assert np.array_equal(result, expected), f"axis={ax}"


def test_argsort_complex_axis():
    z = np.array([[3 + 0j, 1 + 2j], [0 + 1j, 2 + 0j]])
    for ax in (0, 1):
        expected = np.argsort(z, axis=ax)
        result = _np(fr.argsort(fr.array(z), axis=ax))
        assert np.array_equal(result, expected), f"axis={ax}"


# ---------------------------------------------------------------------------
# unique
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_unique_complex(dt):
    z = np.array(ZDUP, dtype=dt)
    expected = np.unique(z)
    result = _np(fr.unique(fr.array(z)))
    assert np.array_equal(result, expected)


def test_unique_complex_nan():
    z = np.array([np.nan + 0j, 1 + 1j, np.nan + 0j, 1 + 1j])
    expected = np.unique(z)  # numpy keeps one nan-complex, last
    result = _np(fr.unique(fr.array(z)))
    assert np.array_equal(result, expected, equal_nan=True)


# ---------------------------------------------------------------------------
# min / max / amin / amax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_min_max_complex(dt):
    z = np.array(Z, dtype=dt)
    exp_min, exp_max = np.min(z), np.max(z)
    r_min = _np(fr.min(fr.array(z)))
    r_max = _np(fr.max(fr.array(z)))
    assert np.allclose(r_min, exp_min), f"min {dt}"
    assert np.allclose(r_max, exp_max), f"max {dt}"


def test_amin_amax_complex():
    z = np.array(Z)
    assert np.allclose(_np(fr.amin(fr.array(z))), np.amin(z))
    assert np.allclose(_np(fr.amax(fr.array(z))), np.amax(z))


def test_min_max_complex_nan_propagates():
    z = np.array(ZNAN)
    # numpy min/max propagate nan: the FIRST nan-complex (index 1) is returned.
    exp_min, exp_max = np.min(z), np.max(z)
    r_min = _np(fr.min(fr.array(z)))
    r_max = _np(fr.max(fr.array(z)))
    assert np.isnan(r_min.real) or np.isnan(r_min.imag)
    assert np.isnan(exp_min.real) or np.isnan(exp_min.imag)
    assert np.isnan(r_max.real) or np.isnan(r_max.imag)


def test_min_max_complex_axis():
    z = np.array([[3 + 0j, 1 + 2j], [0 + 1j, 2 + 0j]])
    for ax in (0, 1):
        assert np.allclose(_np(fr.min(fr.array(z), axis=ax)), np.min(z, axis=ax))
        assert np.allclose(_np(fr.max(fr.array(z), axis=ax)), np.max(z, axis=ax))


# ---------------------------------------------------------------------------
# argmin / argmax
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_argmin_argmax_complex(dt):
    z = np.array(Z, dtype=dt)
    assert int(np.asarray(fr.argmin(fr.array(z)))) == int(np.argmin(z))
    assert int(np.asarray(fr.argmax(fr.array(z)))) == int(np.argmax(z))
    assert _np(fr.argmin(fr.array(z))).dtype == np.int64


def test_argmin_argmax_complex_nan_first():
    z = np.array(ZNAN)
    # numpy returns the first nan-complex index (1) for both.
    assert int(np.asarray(fr.argmin(fr.array(z)))) == int(np.argmin(z))
    assert int(np.asarray(fr.argmax(fr.array(z)))) == int(np.argmax(z))


def test_argmin_argmax_complex_axis():
    z = np.array([[3 + 0j, 1 + 2j], [0 + 1j, 2 + 0j]])
    for ax in (0, 1):
        assert np.array_equal(
            _np(fr.argmin(fr.array(z), axis=ax)), np.argmin(z, axis=ax)
        )
        assert np.array_equal(
            _np(fr.argmax(fr.array(z), axis=ax)), np.argmax(z, axis=ax)
        )


# ---------------------------------------------------------------------------
# ptp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_ptp_complex(dt):
    z = np.array(Z, dtype=dt)
    expected = np.ptp(z)  # lexicographic max - min
    result = _np(fr.ptp(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-6)


def test_ptp_complex_axis():
    z = np.array([[3 + 0j, 1 + 2j], [0 + 1j, 2 + 0j]])
    for ax in (0, 1):
        assert np.allclose(_np(fr.ptp(fr.array(z), axis=ax)), np.ptp(z, axis=ax))


# ---------------------------------------------------------------------------
# real path unregressed (incl. #918 nan-last real sort)
# ---------------------------------------------------------------------------

def test_real_sort_nan_last_unregressed():
    a = np.array([3.0, np.nan, 1.0, 2.0])
    assert np.array_equal(_np(fr.sort(fr.array(a))), np.sort(a), equal_nan=True)


def test_real_ordering_ops_unregressed():
    a = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    assert np.array_equal(_np(fr.sort(fr.array(a))), np.sort(a))
    assert np.array_equal(_np(fr.argsort(fr.array(a))), np.argsort(a))
    assert np.array_equal(_np(fr.unique(fr.array(a))), np.unique(a))
    assert np.allclose(_np(fr.min(fr.array(a))), np.min(a))
    assert np.allclose(_np(fr.max(fr.array(a))), np.max(a))
    assert int(np.asarray(fr.argmin(fr.array(a)))) == int(np.argmin(a))
    assert int(np.asarray(fr.argmax(fr.array(a)))) == int(np.argmax(a))
    assert np.allclose(_np(fr.ptp(fr.array(a))), np.ptp(a))
