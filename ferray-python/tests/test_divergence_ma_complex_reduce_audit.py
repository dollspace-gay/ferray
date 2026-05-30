"""ACToR critic re-audit pins for #873 complex masked reductions.

Audited builder commit 58f4985e (ferray-python/src/ma.rs: ComplexParts +
ma_complex_mean / ma_complex_var / ma_complex_extremum / ma_complex_median).

Every expected value below is DERIVED LIVE from numpy.ma at test time
(R-CHAR-3) — never literal-copied from the ferray side. Each test that
diverges from numpy is left UNMARKED (release-blocker: failing test IS the
block). Probe-matrix tests that PASS are kept as regression guards.

Verification model: pytest `import ferray as fr` vs `import numpy as np`
(numpy 2.4.5 oracle). Rebuild with `maturin develop` before running.
"""

import numpy as np
import ferray as fr
import pytest


C64 = np.complex64
C128 = np.complex128


def _frma(data, dtype, mask):
    return fr.ma.array(np.asarray(data, dtype=dtype), mask=mask)


def _npma(data, dtype, mask):
    return np.ma.array(np.asarray(data, dtype=dtype), mask=mask)


# ---------------------------------------------------------------------------
# DIVERGENCE 1 — ma_complex_mean: per-component real division vs numpy's
# complex-by-scalar division (Smith's algorithm).
#
# numpy/ma/core.py:5417  `_mean = sum / count` where `sum` is a complex
# scalar and the division `complex128 / int` routes through numpy's complex
# division umath loop (numpy/_core/src/umath/loops.c.src), NOT a per-component
# real divide. For sum=(7+6j), count=3, numpy yields real part
# 2.333333333333333; ferray's `acc.re / n` (ma.rs:967) yields 7.0/3.0 =
# 2.3333333333333335 (1 ULP high).
# ---------------------------------------------------------------------------
def test_divergence_complex_mean_smith_division_c128():
    data = [1 + 2j, 3 - 1j, 2 + 0j, 4 + 4j]
    mask = [0, 1, 0, 0]
    expected = complex(_npma(data, C128, mask).mean())  # live numpy oracle
    got = complex(_frma(data, C128, mask).mean())
    assert got.real.hex() == expected.real.hex(), (
        f"complex mean real part: numpy={expected.real.hex()} "
        f"ferray={got.real.hex()}"
    )
    assert got.imag == expected.imag


def test_divergence_complex_mean_smith_division_c64():
    data = [1 + 2j, 3 - 1j, 2 + 0j, 4 + 4j]
    mask = [0, 1, 0, 0]
    expected = complex(_npma(data, C64, mask).mean())  # live numpy oracle
    got = complex(_frma(data, C64, mask).mean())
    assert got.real.hex() == expected.real.hex()
    assert got.imag == expected.imag


# ---------------------------------------------------------------------------
# DIVERGENCE 2 — ma_complex_var: `re*re + im*im` vs numpy's
# `absolute(x - mean) ** 2`.
#
# numpy/ma/core.py MaskedArray.var: `danom = self - self.mean(...)`,
# `danom = umath.absolute(danom) ** 2`, `dvar = danom.sum() / cnt`.
# numpy computes |z|^2 as (hypot(re, im)) ** 2, which rounds to a different
# last bit than ferray's `dr*dr + di*di` (ma.rs:989). Robust 1-ULP
# divergence; identical for c64 and c128 inputs (numpy mean intermediate is
# complex128 in both).
# ---------------------------------------------------------------------------
def test_divergence_complex_var_abs_squared_c128():
    data = [1.5 + 2.5j, 3.5 - 1.5j, 0.5 + 0.5j, 4.5 + 4.5j]
    mask = [0, 1, 0, 0]
    expected = float(_npma(data, C128, mask).var())  # live numpy oracle
    got = float(_frma(data, C128, mask).var())
    assert got.hex() == expected.hex(), (
        f"complex var: numpy={expected.hex()} ferray={got.hex()}"
    )


def test_divergence_complex_var_abs_squared_c64():
    data = [1.5 + 2.5j, 3.5 - 1.5j, 0.5 + 0.5j, 4.5 + 4.5j]
    mask = [0, 1, 0, 0]
    expected = float(_npma(data, C64, mask).var())  # live numpy oracle
    got = float(_frma(data, C64, mask).var())
    assert got.hex() == expected.hex()


def test_divergence_complex_std_abs_squared_c128():
    # std = sqrt(var); inherits the var divergence above.
    data = [1.5 + 2.5j, 3.5 - 1.5j, 0.5 + 0.5j, 4.5 + 4.5j]
    mask = [0, 1, 0, 0]
    expected = float(_npma(data, C128, mask).std())  # live numpy oracle
    got = float(_frma(data, C128, mask).std())
    assert got.hex() == expected.hex()


# ---------------------------------------------------------------------------
# DIVERGENCE 3 — complex var/std return a Python `float`, not a numpy
# `float64` scalar (R-DEV-3 output-object contract). numpy's complex var/std
# return np.float64; ferray returns a bare Python float (loses the numpy
# dtype contract across the PyO3 boundary, R-CODE-4). NOTE: this also affects
# the REAL-dtype var/std path, i.e. it is a pre-existing binding convention
# surfaced (not newly introduced by #873), but it remains a live divergence.
# ---------------------------------------------------------------------------
def test_divergence_complex_var_returns_numpy_float64():
    data = [1 + 2j, 3 - 1j, 2 + 0j, 4 + 4j]
    mask = [0, 1, 0, 0]
    np_type = type(_npma(data, C128, mask).var())  # numpy.float64 (live)
    fr_val = _frma(data, C128, mask).var()
    assert np_type is np.float64
    assert isinstance(fr_val, np.floating), (
        f"complex var type: numpy returns {np_type}, ferray returns "
        f"{type(fr_val)}"
    )


# ===========================================================================
# REGRESSION GUARDS (these PASS — pin numpy-faithful behavior the builder got
# right, so a future change can't silently break them).
# ===========================================================================
@pytest.mark.parametrize("dtype", [C64, C128])
@pytest.mark.parametrize("op", ["min", "max"])
def test_guard_complex_extremum_lexicographic_dtype_width(dtype, op):
    data = [2 + 5j, 2 - 3j, 2 + 1j, -1 + 9j, -1 - 9j]
    npa = np.ma.array(np.asarray(data, dtype=dtype))
    fra = fr.ma.array(np.asarray(data, dtype=dtype))
    rn = getattr(npa, op)()
    rf = getattr(fra, op)()
    assert complex(rf) == complex(rn)
    assert str(rf.dtype) == str(rn.dtype)  # width preserved (no promotion)


@pytest.mark.parametrize("dtype", [C64, C128])
def test_guard_complex_median_even_count_average_width(dtype):
    data = [1 + 1j, 4 + 4j, 2 + 0j, 3 + 9j]  # sorted middles -> avg
    npr = np.ma.median(np.ma.array(np.asarray(data, dtype=dtype)))
    frr = fr.ma.median(fr.ma.array(np.asarray(data, dtype=dtype)))
    assert complex(frr) == complex(npr)
    assert str(frr.dtype) == str(npr.dtype)


@pytest.mark.parametrize("dtype", [C64, C128])
def test_guard_complex_median_odd_count(dtype):
    data = [1 + 2j, 2 + 0j, 4 + 4j]
    npr = np.ma.median(np.ma.array(np.asarray(data, dtype=dtype)))
    frr = fr.ma.median(fr.ma.array(np.asarray(data, dtype=dtype)))
    assert complex(frr) == complex(npr)
    assert str(frr.dtype) == str(npr.dtype)


@pytest.mark.parametrize("dtype", [C64, C128])
@pytest.mark.parametrize("op", ["mean", "min", "max", "var", "std"])
def test_guard_all_masked_returns_masked_singleton(dtype, op):
    data = [1 + 2j, 3 - 1j]
    mask = [1, 1]
    rn = getattr(_npma(data, dtype, mask), op)()
    rf = getattr(_frma(data, dtype, mask), op)()
    assert (rn is np.ma.masked) == (rf is np.ma.masked)
    assert rf is np.ma.masked


@pytest.mark.parametrize("dtype", [C64, C128])
def test_guard_all_masked_sum_prod_median_masked(dtype):
    data = [1 + 2j, 3 - 1j]
    mask = [1, 1]
    npa = _npma(data, dtype, mask)
    fra = _frma(data, dtype, mask)
    assert (np.ma.sum(npa) is np.ma.masked) and (fr.ma.sum(fra) is np.ma.masked)
    assert (np.ma.prod(npa) is np.ma.masked) and (fr.ma.prod(fra) is np.ma.masked)
    assert (np.ma.median(npa) is np.ma.masked) and (
        fr.ma.median(fra) is np.ma.masked
    )


@pytest.mark.parametrize("dtype", [C64, C128])
def test_guard_sum_prod_width_preserved(dtype):
    data = [1 + 2j, 3 - 1j, 2 + 0j, 4 + 4j]
    mask = [0, 1, 0, 0]
    npa = _npma(data, dtype, mask)
    fra = _frma(data, dtype, mask)
    rn_s, rf_s = np.ma.sum(npa), fr.ma.sum(fra)
    rn_p, rf_p = np.ma.prod(npa), fr.ma.prod(fra)
    assert complex(rf_s) == complex(rn_s) and str(rf_s.dtype) == str(rn_s.dtype)
    assert complex(rf_p) == complex(rn_p) and str(rf_p.dtype) == str(rn_p.dtype)


def test_guard_single_unmasked_var_zero():
    fra = fr.ma.array(np.asarray([5 + 5j], dtype=C128))
    npa = np.ma.array(np.asarray([5 + 5j], dtype=C128))
    assert float(fra.var()) == float(npa.var()) == 0.0
