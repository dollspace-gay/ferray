"""Expansion tests (#935) — complex-dtype arms for diff / convolve /
logical_and/or/xor/not / array_equal / linspace / arange.

These functions previously raised `TypeError` on complex input (the
`match_dtype_*` dispatch had no complex arm) while numpy COMPUTES a complex (or
boolean) result. The builder added a complex arm to each binding, composing from
the existing complex marshalling seam (`complex_pyarray_to_ferray` /
`complex_ferray_to_pyarray`) and the `Complex`-generic library ops
(`ferray_ufunc::diff`/`convolve`/`logical_*`/`array_equal`).

Every `expected` is derived from a LIVE numpy call (R-CHAR-3) — never copied from
the ferray side. The matching #935 divergence pins in
`test_divergence_complex_sweep_audit.py` flip GREEN alongside these.
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    """ferray array -> numpy array for value/dtype comparison."""
    return np.asarray(np.array(x.tolist()))


# ---------------------------------------------------------------------------
# diff — successive differences, dtype-preserving (pure subtraction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dt", ["complex128", "complex64"])
def test_diff_complex_matches_numpy(dt):
    z = np.array([1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j], dtype=dt)
    expected = np.diff(z)  # live oracle
    result = _np(fr.diff(fr.array(z.tolist())))
    assert np.iscomplexobj(result), f"diff {dt} dropped imag: {result!r}"
    assert np.allclose(result, expected, rtol=1e-6)


def test_diff_complex_n2_matches_numpy():
    z = np.array([1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j])
    expected = np.diff(z, n=2)  # live oracle
    result = _np(fr.diff(fr.array(z.tolist()), 2))
    assert np.allclose(result, expected, rtol=1e-12)


def test_diff_real_unchanged():
    a = [1, 3, 6, 10]
    expected = np.diff(np.array(a))  # live oracle: [2, 3, 4]
    result = _np(fr.diff(fr.array(a)))
    assert not np.iscomplexobj(result)
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# convolve — sum of products (NO conjugation), modes full/same/valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["full", "same", "valid"])
def test_convolve_complex_modes_match_numpy(mode):
    a = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    v = [2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j]
    expected = np.convolve(np.array(a), np.array(v), mode)  # live oracle
    result = _np(fr.convolve(fr.array(a), fr.array(v), mode))
    assert np.iscomplexobj(result), f"convolve {mode} dropped imag: {result!r}"
    assert np.allclose(result, expected, rtol=1e-12), (
        f"convolve {mode}: numpy {expected!r}, ferray {result!r}"
    )


def test_convolve_complex_no_conjugation():
    # If ferray (wrongly) conjugated v like correlate does, the result would
    # differ. Pin against numpy's convolve (no conj) AND against the known
    # difference from correlate (which DOES conj).
    a = [1 + 1j, 2 + 0j]
    v = [0 + 1j, 1 + 0j]
    expected = np.convolve(np.array(a), np.array(v))  # live oracle
    result = _np(fr.convolve(fr.array(a), fr.array(v)))
    assert np.allclose(result, expected, rtol=1e-12)


def test_convolve_complex64_width_preserved():
    a = np.array([1 + 2j, 3 - 1j], dtype="complex64")
    v = np.array([2 + 0j, 0 + 1j], dtype="complex64")
    expected = np.convolve(a, v)  # live oracle
    result = _np(fr.convolve(fr.array(a.tolist()), fr.array(v.tolist())))
    assert np.allclose(result, expected, rtol=1e-6)


def test_convolve_real_unchanged():
    expected = np.convolve([1, 2, 3], [0, 1, 0.5])  # live oracle
    result = _np(fr.convolve(fr.array([1.0, 2.0, 3.0]), fr.array([0.0, 1.0, 0.5])))
    assert not np.iscomplexobj(result)
    assert np.allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# logical_and / or / xor / not — bool via nonzero-truthiness
# ---------------------------------------------------------------------------


def test_logical_and_or_xor_complex_match_numpy():
    a = [1 + 2j, 0 + 0j, 3 + 3j, 2 - 1j]
    b = [2 + 0j, 1 - 1j, 0 + 0j, 1 + 0j]
    na, nb = np.array(a), np.array(b)
    for name, npf, frf in [
        ("and", np.logical_and, fr.logical_and),
        ("or", np.logical_or, fr.logical_or),
        ("xor", np.logical_xor, fr.logical_xor),
    ]:
        expected = npf(na, nb)  # live oracle
        result = _np(frf(fr.array(a), fr.array(b)))
        assert result.dtype == np.bool_, f"logical_{name} not bool: {result.dtype}"
        assert np.array_equal(result, expected), (
            f"logical_{name} complex: numpy {expected!r}, ferray {result!r}"
        )


def test_logical_not_complex_matches_numpy():
    z = [0 + 0j, 1 + 0j, 0 + 2j, 3 - 4j]
    expected = np.logical_not(np.array(z))  # live oracle: [True, False, False, False]
    result = _np(fr.logical_not(fr.array(z)))
    assert result.dtype == np.bool_
    assert np.array_equal(result, expected)


def test_logical_not_complex64_truthiness():
    z = np.array([0 + 0j, 0 + 1j, 1 + 0j], dtype="complex64")
    expected = np.logical_not(z)  # live oracle
    result = _np(fr.logical_not(fr.array(z.tolist())))
    assert np.array_equal(result, expected)


def test_logical_real_unchanged():
    expected = np.logical_and([1, 0, 1], [1, 1, 0])  # live oracle
    result = _np(fr.logical_and(fr.array([1, 0, 1]), fr.array([1, 1, 0])))
    assert np.array_equal(result, expected)


# ---------------------------------------------------------------------------
# array_equal — same shape + all elements equal
# ---------------------------------------------------------------------------


def test_array_equal_complex_true():
    z = [1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j]
    expected = np.array_equal(np.array(z), np.array(z))  # live oracle: True
    result = bool(np.asarray(fr.array_equal(fr.array(z), fr.array(z))))
    assert result == expected


def test_array_equal_complex_false():
    a = [1 + 2j, -3 + 4j]
    b = [1 + 2j, -3 + 5j]
    expected = np.array_equal(np.array(a), np.array(b))  # live oracle: False
    result = bool(np.asarray(fr.array_equal(fr.array(a), fr.array(b))))
    assert result == expected


def test_array_equal_complex_diff_shape_false():
    a = [1 + 2j, -3 + 4j]
    b = [1 + 2j]
    expected = np.array_equal(np.array(a), np.array(b))  # live oracle: False
    result = bool(np.asarray(fr.array_equal(fr.array(a), fr.array(b))))
    assert result == expected


# ---------------------------------------------------------------------------
# linspace — complex endpoints
# ---------------------------------------------------------------------------


def test_linspace_complex_endpoints_matches_numpy():
    expected = np.linspace(0 + 0j, 1 + 1j, 5)  # live oracle
    result = _np(fr.linspace(0 + 0j, 1 + 1j, 5))
    assert np.iscomplexobj(result)
    assert np.allclose(result, expected, rtol=1e-12)


def test_linspace_complex_asymmetric_endpoints():
    expected = np.linspace(1 + 2j, 3 + 4j, 3)  # live oracle: [1+2j,2+3j,3+4j]
    result = _np(fr.linspace(1 + 2j, 3 + 4j, 3))
    assert np.allclose(result, expected, rtol=1e-12)


def test_linspace_complex_endpoint_pinned_exact():
    # numpy pins the last sample to `stop` exactly when endpoint=True.
    expected = np.linspace(0j, 2 + 6j, 4)  # live oracle
    result = _np(fr.linspace(0j, 2 + 6j, 4))
    assert np.allclose(result, expected, rtol=1e-12)
    assert result[-1] == expected[-1]


def test_linspace_complex_no_endpoint():
    expected = np.linspace(0j, 1 + 1j, 4, endpoint=False)  # live oracle
    result = _np(fr.linspace(0j, 1 + 1j, 4, endpoint=False))
    assert np.allclose(result, expected, rtol=1e-12)


def test_linspace_complex_dtype_real_endpoints():
    expected = np.linspace(0.0, 1.0, 3, dtype="complex128")  # live oracle
    result = _np(fr.linspace(0.0, 1.0, 3, dtype="complex128"))
    assert np.iscomplexobj(result)
    assert np.allclose(result, expected, rtol=1e-12)


def test_linspace_real_unchanged():
    expected = np.linspace(0.0, 1.0, 5)  # live oracle
    result = _np(fr.linspace(0.0, 1.0, 5))
    assert not np.iscomplexobj(result)
    assert np.allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# arange — complex dtype with real args
# ---------------------------------------------------------------------------


def test_arange_complex128_dtype_matches_numpy():
    expected = np.arange(0, 3, 1, dtype="complex128")  # live oracle
    result = _np(fr.arange(0, 3, 1, dtype="complex128"))
    assert np.iscomplexobj(result)
    assert np.array_equal(result, expected)


def test_arange_complex64_dtype_matches_numpy():
    expected = np.arange(0, 6, 2, dtype="complex64")  # live oracle
    result = _np(fr.arange(0, 6, 2, dtype="complex64"))
    assert np.array_equal(result, expected)


def test_arange_complex_single_arg():
    expected = np.arange(3, dtype="complex128")  # live oracle: [0j,1+0j,2+0j]
    result = _np(fr.arange(3, dtype="complex128"))
    assert np.array_equal(result, expected)


def test_arange_real_unchanged():
    expected = np.arange(0, 3, 1)  # live oracle: int64 [0,1,2]
    result = _np(fr.arange(0, 3, 1))
    assert not np.iscomplexobj(result)
    assert np.array_equal(result, expected)
