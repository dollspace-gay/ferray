"""Divergence pins — COMPREHENSIVE complex-dtype convergence sweep after epic #919.

Audited HEAD: d20cdf1f (blockers #920-#929 landed). The named blockers covered
input coercion, unary transcendentals, abs/negative/sinc/rint, binary
arithmetic, comparison, the sum/prod/mean/var/std/cumsum/average reduction
family, branch-cut signs, and maximum/minimum/fmax/fmin. This sweep hunts the
**un-named-blocker** ufunc/function surface for remaining complex divergence,
with priority on R-CODE-4 silent complex->real casts (data corruption).

Every `expected` is derived from a LIVE numpy call (R-CHAR-3) — never copied
from the ferray side. All tests are EXPECTED TO FAIL against HEAD d20cdf1f.

Classification:
  C = ferray SILENTLY DISCARDS imaginary part / wrong value (R-CODE-4 HIGH)
  D = ferray RAISES where numpy COMPUTES (missing — MEDIUM)

Tracking: crosslink blockers filed alongside this file.
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    """ferray array -> numpy array for value/dtype comparison."""
    return np.asarray(np.array(x.tolist()))


# ===========================================================================
# GROUP C — SILENT complex->real CAST / WRONG VALUE (R-CODE-4, DATA CORRUPTION)
# These are the highest-severity: ferray returns a real result, silently
# dropping the imaginary part. numpy's value is complex.
# ===========================================================================

def test_divergence_nansum_complex_silent_real_cast():
    """nansum(complex) — ferray returns real 0.0, dropping imag (R-CODE-4).

    numpy/lib/_nanfunctions_impl.py: nansum replaces NaN with 0 then sums,
    preserving complex dtype. np.nansum([1+2j,-3+4j,0j,2-1j]) -> 5j.
    ferray returns the real scalar 0.0 — full imaginary loss.
    """
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.nansum(np.array(z))  # live oracle
    assert np.iscomplexobj(expected)
    result = _np(fr.nansum(fr.array(z)))
    assert np.iscomplexobj(result), (
        f"nansum complex: numpy {expected!r} (complex), ferray {result!r} (real) "
        "— SILENT IMAG DISCARD"
    )
    assert np.allclose(result, expected, rtol=1e-12)


def test_divergence_nanmean_complex_silent_real_cast():
    """nanmean(complex) — ferray returns real 0.0, numpy 1.25j (R-CODE-4)."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.nanmean(np.array(z))  # live oracle
    assert np.iscomplexobj(expected)
    result = _np(fr.nanmean(fr.array(z)))
    assert np.iscomplexobj(result), (
        f"nanmean complex: numpy {expected!r}, ferray {result!r} — SILENT IMAG DISCARD"
    )
    assert np.allclose(result, expected, rtol=1e-12)


def test_divergence_nanprod_complex_silent_real_cast():
    """nanprod(complex) — ferray returns real, numpy complex (R-CODE-4)."""
    z = [1 + 2j, -3 + 4j, 2 - 1j, 1 + 1j]
    expected = np.nanprod(np.array(z))  # live oracle
    assert np.iscomplexobj(expected)
    result = _np(fr.nanprod(fr.array(z)))
    assert np.iscomplexobj(result), (
        f"nanprod complex: numpy {expected!r}, ferray {result!r} — SILENT IMAG DISCARD"
    )
    assert np.allclose(result, expected, rtol=1e-12)


def test_divergence_median_complex_silent_real_cast():
    """median(complex) — ferray drops imag (returns 0.5), numpy 0.5+1j (R-CODE-4).

    numpy sorts complex lexicographically and averages the two middle values,
    preserving complex dtype. np.median([1+2j,-3+4j,0j,2-1j]) -> (0.5+1j).
    ferray returns the real scalar 0.5.
    """
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.median(np.array(z))  # live oracle
    assert np.iscomplexobj(expected)
    result = _np(fr.median(fr.array(z)))
    assert np.iscomplexobj(result), (
        f"median complex: numpy {expected!r}, ferray {result!r} — SILENT IMAG DISCARD"
    )
    assert np.allclose(result, expected, rtol=1e-12)


def test_divergence_correlate_complex_silent_real_cast():
    """correlate(complex) — ferray returns real [1.0], numpy [-3+4j] (R-CODE-4).

    numpy conjugates the second arg for complex correlate
    (numpy/_core/numeric.py correlate). np.correlate([1+2j,..],[2+0j,..])
    -> [-3.+4.j]. ferray returns a real wrong value [1.0].
    """
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    expected = np.correlate(np.array(a), np.array(b))  # live oracle
    assert np.iscomplexobj(expected)
    result = _np(fr.correlate(fr.array(a), fr.array(b)))
    assert np.iscomplexobj(result), (
        f"correlate complex: numpy {expected!r}, ferray {result!r} — SILENT IMAG DISCARD"
    )
    assert np.allclose(result, expected, rtol=1e-12)


def test_divergence_round_complex_is_noop_wrong_value():
    """round(complex) — ferray returns the input UNROUNDED (no-op), wrong value.

    numpy rounds real and imaginary parts independently (banker's rounding).
    np.round([1.4+2.6j, 3.5-1.5j]) -> [1.+3.j, 4.-2.j].
    ferray returns the input unchanged [1.4+2.6j, 3.5-1.5j].
    """
    z = [1.4 + 2.6j, 3.5 - 1.5j]
    expected = np.round(np.array(z))  # live oracle
    result = _np(fr.round(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"round complex: numpy {expected!r}, ferray {result!r} — round is a no-op"
    )


# ===========================================================================
# GROUP D — ferray RAISES where numpy COMPUTES a complex result (MEDIUM/HIGH)
# These are missing complex support. dot/matmul/linalg/sort are HIGH-impact
# common ops; square/sign/reciprocal/isnan are common ufuncs.
# ===========================================================================

# --- HIGH: linear-algebra-core common ops ---

def test_divergence_dot_complex_raises():
    """dot(complex) — numpy computes (5+10j); ferray raises TypeError."""
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    expected = np.dot(np.array(a), np.array(b))  # live oracle
    result = _np(fr.dot(fr.array(a), fr.array(b)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"dot complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_vdot_complex_raises():
    """vdot(complex) — numpy conjugates first arg (-3-4j); ferray raises."""
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    expected = np.vdot(np.array(a), np.array(b))  # live oracle
    result = _np(fr.vdot(fr.array(a), fr.array(b)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"vdot complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_matmul_complex_raises():
    """matmul(complex) — numpy computes; ferray raises TypeError."""
    m = [[1 + 1j, 2 + 0j], [0 + 1j, 1 - 1j]]
    expected = np.matmul(np.array(m), np.array(m))  # live oracle
    result = _np(fr.matmul(fr.array(m), fr.array(m)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"matmul complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_linalg_norm_complex_raises():
    """linalg.norm(complex) — numpy 5.916..., ferray raises TypeError."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.linalg.norm(np.array(z))  # live oracle
    result = float(np.asarray(fr.linalg.norm(fr.array(z))))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"linalg.norm complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_linalg_solve_complex_raises():
    """linalg.solve(complex) — numpy computes; ferray raises TypeError."""
    A = [[2 + 0j, 1 + 1j], [1 - 1j, 3 + 0j]]
    b = [1 + 0j, 2 + 0j]
    expected = np.linalg.solve(np.array(A), np.array(b))  # live oracle
    result = _np(fr.linalg.solve(fr.array(A), fr.array(b)))
    assert np.allclose(result, expected, rtol=1e-10), (
        f"linalg.solve complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_linalg_inv_complex_raises():
    """linalg.inv(complex) — numpy computes; ferray raises TypeError."""
    A = [[2 + 0j, 1 + 1j], [1 - 1j, 3 + 0j]]
    expected = np.linalg.inv(np.array(A))  # live oracle
    result = _np(fr.linalg.inv(fr.array(A)))
    assert np.allclose(result, expected, rtol=1e-10), (
        f"linalg.inv complex: numpy {expected!r}, ferray {result!r}"
    )


# --- HIGH: ordering-based manipulation (lexicographic for complex) ---

def test_divergence_sort_complex_raises():
    """sort(complex) — numpy sorts lexicographically; ferray raises."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.sort(np.array(z))  # live oracle
    result = _np(fr.sort(fr.array(z)))
    assert np.array_equal(result, expected), (
        f"sort complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_argsort_complex_raises():
    """argsort(complex) — numpy lexicographic indices; ferray raises."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.argsort(np.array(z))  # live oracle
    result = _np(fr.argsort(fr.array(z)))
    assert np.array_equal(result, expected), (
        f"argsort complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_argmin_argmax_complex_raises():
    """argmin/argmax(complex) — numpy lexicographic; ferray raises."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    exp_min = int(np.argmin(np.array(z)))  # live oracle
    exp_max = int(np.argmax(np.array(z)))
    r_min = int(np.asarray(fr.argmin(fr.array(z))))
    r_max = int(np.asarray(fr.argmax(fr.array(z))))
    assert r_min == exp_min, f"argmin complex: numpy {exp_min}, ferray {r_min}"
    assert r_max == exp_max, f"argmax complex: numpy {exp_max}, ferray {r_max}"


def test_divergence_min_max_complex_raises():
    """min/max(complex) — numpy lexicographic; ferray raises ordering-op error."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    exp_min = np.min(np.array(z))  # live oracle
    exp_max = np.max(np.array(z))
    r_min = np.asarray(np.array(fr.min(fr.array(z)).tolist()))
    r_max = np.asarray(np.array(fr.max(fr.array(z)).tolist()))
    assert np.allclose(r_min, exp_min), f"min complex: numpy {exp_min}, ferray {r_min}"
    assert np.allclose(r_max, exp_max), f"max complex: numpy {exp_max}, ferray {r_max}"


def test_divergence_unique_complex_raises():
    """unique(complex) — numpy returns sorted unique; ferray raises."""
    z = [1 + 2j, -3 + 4j, 1 + 2j, 2 - 1j]
    expected = np.unique(np.array(z))  # live oracle
    result = _np(fr.unique(fr.array(z)))
    assert np.array_equal(result, expected), (
        f"unique complex: numpy {expected!r}, ferray {result!r}"
    )


# --- MEDIUM/HIGH: dtype-preserving manipulation that should never raise ---

def test_divergence_concatenate_complex_raises():
    """concatenate(complex) — pure data move; ferray raises TypeError."""
    a = [1 + 2j, -3 + 4j]
    b = [0 + 0j, 2 - 1j]
    expected = np.concatenate([np.array(a), np.array(b)])  # live oracle
    result = _np(fr.concatenate([fr.array(a), fr.array(b)]))
    assert np.array_equal(result, expected), (
        f"concatenate complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_reshape_complex_raises():
    """reshape(complex) — pure view; ferray raises TypeError."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.reshape(np.array(z), (2, 2))  # live oracle
    result = _np(fr.reshape(fr.array(z), (2, 2)))
    assert np.array_equal(result, expected), (
        f"reshape complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_transpose_complex_raises():
    """transpose(complex) — pure view; ferray raises TypeError."""
    z = [[1 + 2j, -3 + 4j], [0 + 0j, 2 - 1j]]
    expected = np.transpose(np.array(z))  # live oracle
    result = _np(fr.transpose(fr.array(z)))
    assert np.array_equal(result, expected), (
        f"transpose complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_where_complex_raises():
    """where(complex) — pure select; ferray raises TypeError."""
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    mask = [True, False, True, False]
    expected = np.where(np.array(mask), np.array(a), np.array(b))  # live oracle
    result = _np(fr.where(fr.array(mask), fr.array(a), fr.array(b)))
    assert np.array_equal(result, expected), (
        f"where complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_flip_roll_repeat_tile_complex_raise():
    """flip/roll/repeat/tile(complex) — pure data moves; ferray raises."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    nz = np.array(z)
    for name, npf, frf in [
        ("flip", lambda: np.flip(nz), lambda: fr.flip(fr.array(z))),
        ("roll", lambda: np.roll(nz, 1), lambda: fr.roll(fr.array(z), 1)),
        ("repeat", lambda: np.repeat(nz, 2), lambda: fr.repeat(fr.array(z), 2)),
        ("tile", lambda: np.tile(nz, 2), lambda: fr.tile(fr.array(z), 2)),
    ]:
        expected = npf()  # live oracle
        result = _np(frf())
        assert np.array_equal(result, expected), (
            f"{name} complex: numpy {expected!r}, ferray {result!r}"
        )


# --- MEDIUM: common ufuncs that numpy registers complex loops for ---

def test_divergence_square_complex_raises():
    """square(complex) — numpy z*z; ferray raises TypeError."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.square(np.array(z))  # live oracle
    result = _np(fr.square(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"square complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_reciprocal_complex_raises():
    """reciprocal(complex) — numpy 1/z; ferray raises TypeError."""
    z = [1 + 2j, -3 + 4j, 2 - 1j, 1 + 1j]
    expected = np.reciprocal(np.array(z))  # live oracle
    result = _np(fr.reciprocal(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"reciprocal complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_sign_complex_raises():
    """sign(complex) — numpy sign(z)=z/|z| (NEP convention); ferray raises.

    np.sign for complex returns z/abs(z) (unit phasor), 0 for 0.
    np.sign([1+2j,...]) -> [0.447+0.894j, ...]. ferray raises TypeError.
    """
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.sign(np.array(z))  # live oracle
    result = _np(fr.sign(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"sign complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_isnan_isinf_isfinite_complex_raise():
    """isnan/isinf/isfinite(complex) — numpy tests either part; ferray raises.

    numpy registers complex loops: isnan true if either part is NaN, etc.
    """
    z = [1 + 2j, complex(np.nan, 0), complex(0, np.inf), 2 - 1j]
    nz = np.array(z)
    for name, npf, frf in [
        ("isnan", np.isnan, fr.isnan),
        ("isinf", np.isinf, fr.isinf),
        ("isfinite", np.isfinite, fr.isfinite),
    ]:
        expected = npf(nz)  # live oracle
        result = _np(frf(fr.array(z)))
        assert np.array_equal(result, expected), (
            f"{name} complex: numpy {expected!r}, ferray {result!r}"
        )


# --- MEDIUM: arithmetic-ish reductions/scans on complex ---

def test_divergence_ptp_complex_raises():
    """ptp(complex) — numpy max-min lexicographic (5-5j); ferray raises."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.ptp(np.array(z))  # live oracle
    result = _np(fr.ptp(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"ptp complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_diff_complex_raises():
    """diff(complex) — numpy successive differences; ferray raises."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.diff(np.array(z))  # live oracle
    result = _np(fr.diff(fr.array(z)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"diff complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_convolve_complex_raises():
    """convolve(complex) — numpy computes; ferray raises TypeError."""
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    expected = np.convolve(np.array(a), np.array(b))  # live oracle
    result = _np(fr.convolve(fr.array(a), fr.array(b)))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"convolve complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_inner_outer_complex_raise():
    """inner/outer(complex) — numpy computes; ferray raises TypeError."""
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    exp_in = np.inner(np.array(a), np.array(b))  # live oracle
    exp_out = np.outer(np.array(a), np.array(b))
    r_in = _np(fr.inner(fr.array(a), fr.array(b)))
    r_out = _np(fr.outer(fr.array(a), fr.array(b)))
    assert np.allclose(r_in, exp_in, rtol=1e-12), (
        f"inner complex: numpy {exp_in!r}, ferray {r_in!r}"
    )
    assert np.allclose(r_out, exp_out, rtol=1e-12), (
        f"outer complex: numpy {exp_out!r}, ferray {r_out!r}"
    )


# --- MEDIUM: logic / equality on complex ---

def test_divergence_logical_and_not_complex_raise():
    """logical_and/logical_not(complex) — numpy via nonzero; ferray raises."""
    a = [1 + 2j, 0 + 0j, 3 + 3j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 0 + 0j, 1 + 0j]
    exp_and = np.logical_and(np.array(a), np.array(b))  # live oracle
    exp_not = np.logical_not(np.array(a))
    r_and = _np(fr.logical_and(fr.array(a), fr.array(b)))
    r_not = _np(fr.logical_not(fr.array(a)))
    assert np.array_equal(r_and, exp_and), (
        f"logical_and complex: numpy {exp_and!r}, ferray {r_and!r}"
    )
    assert np.array_equal(r_not, exp_not), (
        f"logical_not complex: numpy {exp_not!r}, ferray {r_not!r}"
    )


def test_divergence_array_equal_complex_raises():
    """array_equal(complex) — numpy True; ferray raises TypeError."""
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.array_equal(np.array(z), np.array(z))  # live oracle
    result = bool(np.asarray(fr.array_equal(fr.array(z), fr.array(z))))
    assert result == expected, (
        f"array_equal complex: numpy {expected!r}, ferray {result!r}"
    )


# --- MEDIUM: complex-dtype creation ---

def test_divergence_linspace_complex_endpoints_raises():
    """linspace(complex) — numpy interpolates complex; ferray raises TypeError."""
    expected = np.linspace(0 + 0j, 1 + 1j, 5)  # live oracle
    result = _np(fr.linspace(0 + 0j, 1 + 1j, 5))
    assert np.allclose(result, expected, rtol=1e-12), (
        f"linspace complex: numpy {expected!r}, ferray {result!r}"
    )


def test_divergence_arange_complex_dtype_raises():
    """arange(dtype=complex128) — numpy complex ramp; ferray raises TypeError."""
    expected = np.arange(0, 3, 1, dtype="complex128")  # live oracle
    result = _np(fr.arange(0, 3, 1, dtype="complex128"))
    assert np.array_equal(result, expected), (
        f"arange complex dtype: numpy {expected!r}, ferray {result!r}"
    )
