"""Complex masked-array reductions (#873, R-6) vs the live numpy.ma oracle.

Pins ferray.ma's complex64/complex128 full reductions — `sum`, `prod`, `mean`,
`min`, `max`, `var`, `std`, and `np.ma.median` — against numpy 2.4.5. Every
expected value is DERIVED from a live numpy.ma call (R-CHAR-3), never
literal-copied from the ferray side.

numpy.ma complex-reduction contract captured here (verified live, numpy 2.4.5):
  - sum/prod  -> complex scalar, INPUT width preserved (c64->c64, c128->c128).
  - mean      -> complex scalar, ALWAYS complex128 (c64 PROMOTES to c128).
  - min/max   -> complex scalar, LEXICOGRAPHIC (real, then imag), width preserved.
  - var/std   -> REAL float64 ALWAYS (var = mean(|x-mean|**2); std = sqrt(var)),
                 regardless of input width (c64 var/std are STILL float64).
  - median    -> complex scalar, lexicographic sort, middle (odd) / average of
                 the two middles (even), width preserved.
  - all-masked -> the `masked` singleton for every reduction.
"""

import numpy as np
import pytest

import ferray as fr

# Mixed-sign, distinct-magnitude complex data + a mask so the reduction runs
# over a strict subset of the elements (exercises mask-awareness).
_DATA = [1 + 2j, 3 - 1j, 2 + 0j, 4 + 4j]
_MASK = [0, 1, 0, 0]  # element index 1 masked -> reduce over {1+2j, 2+0j, 4+4j}

_DTYPES = [np.complex64, np.complex128]


def _np(data, mask, dt):
    return np.ma.array(data, mask=mask, dtype=dt)


def _fr(data, mask, dt):
    # ferray.ma.array dtype name matches numpy's ("complex64"/"complex128").
    return fr.ma.array(data, mask=mask, dtype=str(np.dtype(dt)))


# --- sum / prod (complex, width-preserved) ---------------------------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_sum(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert complex(fr.ma.sum(f)) == complex(n.sum())
    assert str(np.asarray(fr.ma.sum(f)).dtype) == str(n.sum().dtype)


@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_prod(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert complex(fr.ma.prod(f)) == complex(n.prod())
    assert str(np.asarray(fr.ma.prod(f)).dtype) == str(n.prod().dtype)


# --- mean (complex, ALWAYS complex128) -------------------------------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_mean_value(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert complex(f.mean()) == pytest.approx(complex(n.mean()))


@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_mean_dtype_is_complex128(dt):
    # numpy promotes a complex64 masked-array mean to complex128.
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert str(np.asarray(f.mean()).dtype) == str(n.mean().dtype) == "complex128"


# --- min / max (lexicographic, width-preserved) ----------------------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_min(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert complex(f.min()) == complex(n.min())
    assert str(np.asarray(f.min()).dtype) == str(n.min().dtype)


@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_max(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert complex(f.max()) == complex(n.max())
    assert str(np.asarray(f.max()).dtype) == str(n.max().dtype)


def test_complex_min_max_lexicographic_tie_on_imag():
    # Equal real parts -> the order is decided by the imaginary part.
    data = [2 + 5j, 2 + 1j, 2 + 3j]
    mask = [0, 0, 0]
    n = _np(data, mask, np.complex128)
    f = _fr(data, mask, np.complex128)
    assert complex(f.min()) == complex(n.min())  # 2+1j
    assert complex(f.max()) == complex(n.max())  # 2+5j


# --- var / std (REAL float64 ALWAYS) ---------------------------------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_var_value(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert float(f.var()) == pytest.approx(float(n.var()))


@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_std_value(dt):
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert float(f.std()) == pytest.approx(float(n.std()))


@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_var_std_dtype_is_real_float64(dt):
    # numpy returns float64 var/std for BOTH complex64 and complex128 inputs.
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert str(np.asarray(f.var()).dtype) == str(n.var().dtype) == "float64"
    assert str(np.asarray(f.std()).dtype) == str(n.std().dtype) == "float64"


# --- median (lexicographic, width-preserved, odd & even) -------------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_median_odd_count(dt):
    # _MASK leaves 3 unmasked elements -> odd -> the middle element.
    n = _np(_DATA, _MASK, dt)
    f = _fr(_DATA, _MASK, dt)
    assert complex(fr.ma.median(f)) == complex(np.ma.median(n))
    assert str(np.asarray(fr.ma.median(f)).dtype) == str(np.ma.median(n).dtype)


@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_median_even_count_averages_two_middles(dt):
    data = [1 + 2j, 2 + 0j, 4 + 4j, 3 + 3j]
    mask = [0, 0, 0, 0]  # 4 unmasked -> even -> average the two middle (sorted)
    n = _np(data, mask, dt)
    f = _fr(data, mask, dt)
    assert complex(fr.ma.median(f)) == pytest.approx(complex(np.ma.median(n)))
    assert str(np.asarray(fr.ma.median(f)).dtype) == str(np.ma.median(n).dtype)


# --- single unmasked element -----------------------------------------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_single_unmasked_element(dt):
    data = [5 + 6j, 1 + 1j]
    mask = [0, 1]  # only 5+6j survives
    n = _np(data, mask, dt)
    f = _fr(data, mask, dt)
    assert complex(f.mean()) == pytest.approx(complex(n.mean()))
    assert complex(f.min()) == complex(n.min())
    assert complex(f.max()) == complex(n.max())
    assert complex(fr.ma.median(f)) == complex(np.ma.median(n))
    # variance of a single element is 0.0 (numpy).
    assert float(f.var()) == pytest.approx(float(n.var()))
    assert float(f.std()) == pytest.approx(float(n.std()))


# --- all-masked -> the `masked` singleton for EVERY reduction --------------

@pytest.mark.parametrize("dt", _DTYPES)
def test_complex_all_masked_returns_masked(dt):
    data = [1 + 2j, 3 - 1j]
    mask = [1, 1]
    n = _np(data, mask, dt)
    f = _fr(data, mask, dt)
    assert f.mean() is np.ma.masked
    assert f.min() is np.ma.masked
    assert f.max() is np.ma.masked
    assert f.var() is np.ma.masked
    assert f.std() is np.ma.masked
    assert fr.ma.median(f) is np.ma.masked
    assert fr.ma.sum(f) is np.ma.masked
    assert fr.ma.prod(f) is np.ma.masked
    # sanity: numpy agrees these are masked.
    assert n.mean() is np.ma.masked
    assert np.ma.median(n) is np.ma.masked
