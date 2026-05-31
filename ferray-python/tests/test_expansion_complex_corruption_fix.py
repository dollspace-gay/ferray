"""R-CODE-4 corruption-fix regression: complex nansum/nanmean/nanprod/median/
correlate/round must COMPUTE complex (no silent imaginary-part discard).

Blocker #930 (refs #919). Before the fix these six funnelled complex input
through a `coerce_dtype(arr, "float64")` (or were a no-op for `round`), which
numpy casts complex->float64 DROPPING the imaginary part with a `ComplexWarning`
— an active boundary data-corruption. numpy COMPUTES every one of them as
complex.

Every `expected` is derived from a LIVE numpy call (R-CHAR-3) — never copied
from the ferray side. Covered: with + without NaN, axis reductions, odd/even
median, all correlate modes, round decimals (0 and >0), c64 + c128 width, and
the real path staying byte-identical.
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    """ferray array -> numpy array for value/dtype comparison."""
    return np.asarray(np.array(x.tolist()))


def _match(fv, expected):
    got = _np(fv)
    assert np.iscomplexobj(got) == np.iscomplexobj(expected), (
        f"complexness mismatch: ferray {got!r} vs numpy {expected!r}"
    )
    assert np.allclose(got, expected, rtol=1e-6, equal_nan=True), (
        f"value mismatch: ferray {got!r} vs numpy {expected!r}"
    )


# ---------------------------------------------------------------------------
# nansum / nanprod / nanmean — complex, NaN->identity, width, axis
# ---------------------------------------------------------------------------

Z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
ZP = [1 + 2j, -3 + 4j, 2 - 1j, 1 + 1j]


def test_nansum_complex_no_nan():
    _match(fr.nansum(fr.array(Z)), np.nansum(np.array(Z)))


def test_nanprod_complex_no_nan():
    _match(fr.nanprod(fr.array(ZP)), np.nanprod(np.array(ZP)))


def test_nanmean_complex_no_nan():
    _match(fr.nanmean(fr.array(Z)), np.nanmean(np.array(Z)))


def test_nansum_complex_with_nan():
    z = [1 + 2j, complex(np.nan, 0), 3 + 0j, complex(0, np.nan)]
    _match(fr.nansum(fr.array(z)), np.nansum(np.array(z)))


def test_nanprod_complex_with_nan():
    z = [1 + 2j, complex(np.nan, np.nan), 2 - 1j, 1 + 1j]
    _match(fr.nanprod(fr.array(z)), np.nanprod(np.array(z)))


def test_nanmean_complex_with_nan():
    z = [1 + 2j, complex(np.nan, 0), 3 + 0j]
    _match(fr.nanmean(fr.array(z)), np.nanmean(np.array(z)))


@pytest.mark.parametrize("npdt", [np.complex64, np.complex128])
def test_nan_reductions_complex_width_preserved(npdt):
    z = np.array([1 + 2j, -3 + 4j, 2 - 1j, 1 + 1j], dtype=npdt)
    _match(fr.nansum(fr.array(z)), np.nansum(z))
    _match(fr.nanprod(fr.array(z)), np.nanprod(z))
    _match(fr.nanmean(fr.array(z)), np.nanmean(z))


def test_nan_reductions_complex_axis():
    m = np.array([[1 + 2j, 3 + 1j], [5 + 0j, 2 - 2j]])
    _match(fr.nansum(fr.array(m), 0), np.nansum(m, 0))
    _match(fr.nansum(fr.array(m), 1), np.nansum(m, 1))
    _match(fr.nanmean(fr.array(m), 0), np.nanmean(m, 0))
    _match(fr.nanprod(fr.array(m), 1), np.nanprod(m, 1))


# ---------------------------------------------------------------------------
# median — lexicographic sort, odd/even, axis, width
# ---------------------------------------------------------------------------

def test_median_complex_even():
    _match(fr.median(fr.array(Z)), np.median(np.array(Z)))


def test_median_complex_odd():
    z = [1 + 2j, -3 + 4j, 2 - 1j]
    _match(fr.median(fr.array(z)), np.median(np.array(z)))


def test_median_complex_axis():
    m = np.array([[1 + 2j, 3 + 1j], [5 + 0j, 2 - 2j]])
    _match(fr.median(fr.array(m), 0), np.median(m, 0))
    _match(fr.median(fr.array(m), 1), np.median(m, 1))


@pytest.mark.parametrize("npdt", [np.complex64, np.complex128])
def test_median_complex_width_preserved(npdt):
    z = np.array([1 + 2j, -3 + 4j, 0j, 2 - 1j], dtype=npdt)
    _match(fr.median(fr.array(z)), np.median(z))


# ---------------------------------------------------------------------------
# correlate — numpy conjugates v; all modes, width
# ---------------------------------------------------------------------------

CA = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
CB = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]


@pytest.mark.parametrize("mode", ["valid", "same", "full"])
def test_correlate_complex_modes_conjugate(mode):
    _match(
        fr.correlate(fr.array(CA), fr.array(CB), mode),
        np.correlate(np.array(CA), np.array(CB), mode),
    )


def test_correlate_complex_c64_width():
    a = np.array(CA, dtype=np.complex64)
    b = np.array(CB, dtype=np.complex64)
    _match(fr.correlate(fr.array(a), fr.array(b)), np.correlate(a, b))


def test_correlate_complex_asymmetric_lengths():
    a = [1 + 1j, 2 + 0j, 3 - 1j]
    b = [0 + 0j, 1 + 0j, 0.5j]
    for mode in ("valid", "same", "full"):
        _match(
            fr.correlate(fr.array(a), fr.array(b), mode),
            np.correlate(np.array(a), np.array(b), mode),
        )


# ---------------------------------------------------------------------------
# round / around — per-component half-to-even, decimals, width
# ---------------------------------------------------------------------------

def test_round_complex_decimals0():
    z = [1.4 + 2.6j, 3.5 - 1.5j]
    _match(fr.round(fr.array(z)), np.round(np.array(z)))


def test_around_complex_decimals_positive():
    z = [1.45 + 2.65j, 3.123 - 1.987j]
    _match(fr.around(fr.array(z), 1), np.round(np.array(z), 1))
    _match(fr.around(fr.array(z), 2), np.round(np.array(z), 2))


def test_round_complex_half_even():
    # ties to even on both parts: 2.5->2, 3.5->4, -1.5->-2, 0.5->0
    z = [2.5 + 3.5j, -1.5 + 0.5j]
    _match(fr.round(fr.array(z)), np.round(np.array(z)))


@pytest.mark.parametrize("npdt", [np.complex64, np.complex128])
def test_around_complex_width_preserved(npdt):
    z = np.array([1.4 + 2.6j, 3.5 - 1.5j], dtype=npdt)
    _match(fr.around(fr.array(z)), np.round(z))


# ---------------------------------------------------------------------------
# Real path UNCHANGED (no regression from the complex arms)
# ---------------------------------------------------------------------------

def test_real_path_unchanged():
    _match(fr.nansum(fr.array([1.0, np.nan, 3.0])), np.nansum([1.0, np.nan, 3.0]))
    _match(fr.nanprod(fr.array([1.0, np.nan, 3.0])), np.nanprod([1.0, np.nan, 3.0]))
    _match(fr.nanmean(fr.array([1.0, np.nan, 3.0])), np.nanmean([1.0, np.nan, 3.0]))
    _match(fr.median(fr.array([1.0, 2.0, 3.0, 4.0])), np.median([1.0, 2.0, 3.0, 4.0]))
    _match(fr.round(fr.array([1.4, 2.5, 3.5])), np.round([1.4, 2.5, 3.5]))
    _match(
        fr.correlate(fr.array([1.0, 2.0, 3.0]), fr.array([0.0, 1.0, 0.5])),
        np.correlate([1.0, 2.0, 3.0], [0.0, 1.0, 0.5]),
    )


def test_corruption_eliminated_oracle():
    """The exact one-liner from the dispatch: nanmean complex matches numpy."""
    got = np.asarray(fr.nanmean(fr.array([1 + 2j, np.nan + 0j, 3 + 0j])))
    assert np.allclose(got, np.nanmean([1 + 2j, np.nan, 3]))
