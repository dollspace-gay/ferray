"""Expansion tests for the #970 R-CODE-4 complex-corruption fix.

`fr.trapezoid` / `fr.interp` / `fr.polyval` / `fr.polymul` / `fr.polyadd` /
`fr.polyder` / `fr.polyint` previously cast complex operands to float64 at the
PyO3 boundary, silently dropping the imaginary part. Each function now detects
a complex operand (`is_complex_dtype`) and delegates the computation to numpy,
which owns the genuinely-complex result.

Every expected value is derived LIVE from numpy in the assertion (R-CHAR-3) —
never literal-copied from the ferray side. Real / float / int paths must remain
unchanged, so each function also has a real-path regression assertion.
"""

import numpy as np
import pytest

import ferray as fr


def _np(x):
    return np.asarray(x)


CZ = [1 + 2j, 3 - 4j, -5 + 0.5j, 2 + 0j]


# ---------------------------------------------------------------------------
# trapezoid
# ---------------------------------------------------------------------------
def test_trapezoid_complex_value_and_dtype():
    expected = np.trapezoid(np.array(CZ))
    got = _np(fr.trapezoid(fr.array(CZ)))
    assert np.iscomplexobj(got)
    assert got.dtype == np.asarray(expected).dtype
    assert np.allclose(got, np.asarray(expected))


def test_trapezoid_complex_with_x():
    x = [0.0, 1.0, 3.0, 4.0]
    expected = np.trapezoid(np.array(CZ), x=np.array(x))
    got = _np(fr.trapezoid(fr.array(CZ), fr.array(x)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


def test_trapezoid_complex_with_dx():
    expected = np.trapezoid(np.array(CZ), dx=0.5)
    got = _np(fr.trapezoid(fr.array(CZ), None, 0.5))
    assert np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


def test_trapezoid_real_path_unchanged():
    y = [1.0, 2.0, 4.0, 7.0]
    expected = np.trapezoid(np.array(y))
    got = _np(fr.trapezoid(fr.array(y)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


# ---------------------------------------------------------------------------
# interp
# ---------------------------------------------------------------------------
def test_interp_complex_fp_value_and_dtype():
    x = [0.5, 1.5, 2.5]
    xp = [1.0, 2.0, 3.0]
    fpv = CZ[:3]
    expected = np.interp(np.array(x), np.array(xp), np.array(fpv))
    got = _np(fr.interp(fr.array(x), fr.array(xp), fr.array(fpv)))
    assert np.iscomplexobj(got)
    assert got.dtype == np.asarray(expected).dtype
    assert np.allclose(got, np.asarray(expected))


def test_interp_real_path_unchanged():
    x = [1.5]
    xp = [1.0, 2.0, 3.0]
    fpv = [10.0, 20.0, 30.0]
    expected = np.interp(np.array(x), np.array(xp), np.array(fpv))
    got = _np(fr.interp(fr.array(x), fr.array(xp), fr.array(fpv)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


# ---------------------------------------------------------------------------
# polyval (complex coeffs and/or complex x)
# ---------------------------------------------------------------------------
def test_polyval_complex_coeffs_real_x():
    expected = np.polyval(np.array(CZ), 2.0)
    got = _np(fr.polyval(fr.array(CZ), 2.0))
    assert np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


def test_polyval_real_coeffs_complex_x():
    coeffs = [1.0, 0.0, -2.0, 1.0]
    expected = np.polyval(np.array(coeffs), 1 + 1j)
    got = _np(fr.polyval(fr.array(coeffs), 1 + 1j))
    assert np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


def test_polyval_complex_array_x():
    xs = [1 + 1j, 2 - 1j]
    expected = np.polyval(np.array(CZ), np.array(xs))
    got = _np(fr.polyval(fr.array(CZ), fr.array(xs)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


def test_polyval_real_path_unchanged():
    coeffs = [1.0, -3.0, 2.0]
    expected = np.polyval(np.array(coeffs), 5.0)
    got = _np(fr.polyval(fr.array(coeffs), 5.0))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


# ---------------------------------------------------------------------------
# polymul / polyadd
# ---------------------------------------------------------------------------
def test_polymul_complex():
    expected = np.polymul(np.array(CZ), np.array(CZ))
    got = _np(fr.polymul(fr.array(CZ), fr.array(CZ)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyadd_complex():
    other = [2 + 1j, -1 + 0j]
    expected = np.polyadd(np.array(CZ), np.array(other))
    got = _np(fr.polyadd(fr.array(CZ), fr.array(other)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polymul_mixed_real_complex():
    real = [1.0, 2.0, 3.0]
    expected = np.polymul(np.array(real), np.array(CZ))
    got = _np(fr.polymul(fr.array(real), fr.array(CZ)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyadd_real_path_unchanged():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0]
    expected = np.polyadd(np.array(a), np.array(b))
    got = _np(fr.polyadd(fr.array(a), fr.array(b)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polymul_real_path_unchanged():
    a = [1.0, 2.0]
    b = [1.0, -1.0]
    expected = np.polymul(np.array(a), np.array(b))
    got = _np(fr.polymul(fr.array(a), fr.array(b)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# polyder / polyint
# ---------------------------------------------------------------------------
def test_polyder_complex():
    expected = np.polyder(np.array(CZ))
    got = _np(fr.polyder(fr.array(CZ)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyder_complex_order2():
    expected = np.polyder(np.array(CZ), 2)
    got = _np(fr.polyder(fr.array(CZ), 2))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyder_real_path_unchanged():
    coeffs = [1.0, 0.0, -2.0, 1.0]
    expected = np.polyder(np.array(coeffs))
    got = _np(fr.polyder(fr.array(coeffs)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyint_complex():
    expected = np.polyint(np.array(CZ))
    got = _np(fr.polyint(fr.array(CZ)))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyint_complex_with_m_and_k():
    expected = np.polyint(np.array(CZ), m=2, k=[1 + 1j, 2 + 0j])
    got = _np(fr.polyint(fr.array(CZ), 2, fr.array([1 + 1j, 2 + 0j])))
    assert np.iscomplexobj(got)
    assert np.allclose(got, expected)


def test_polyint_real_path_unchanged():
    coeffs = [3.0, 0.0, 1.0]
    expected = np.polyint(np.array(coeffs))
    got = _np(fr.polyint(fr.array(coeffs)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# float16 where applicable (real path preserved, not corrupted)
# ---------------------------------------------------------------------------
def test_trapezoid_float16_real():
    y = np.array([1.0, 2.0, 4.0], dtype=np.float16)
    expected = np.trapezoid(y)
    got = _np(fr.trapezoid(fr.array(y)))
    assert not np.iscomplexobj(got)
    assert np.allclose(got, np.asarray(expected))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
