"""R-CODE-4 #968: cov/corrcoef/nan_to_num must compute on complex input
(no silent imaginary-discard), matching numpy. Real/float/int paths unchanged.

Oracle: live numpy 2.4.x (R-CHAR-3 — expected values come from numpy, never
literal-copied from ferray).
"""

import numpy as np
import pytest

import ferray as fr


# --------------------------------------------------------------------------- #
# cov — complex covariance uses the conjugate transpose (X @ conj(X).T)
# --------------------------------------------------------------------------- #
def test_cov_complex_matches_numpy():
    m = np.array([[1 + 1j, 2], [3, 4j]])
    got = np.asarray(fr.cov(fr.array(m)))
    exp = np.cov(m)
    assert np.iscomplexobj(got), "cov(complex) must stay complex, not drop imag"
    assert np.allclose(got, exp), f"{got} != {exp}"


def test_cov_complex64_matches_numpy():
    m = np.array([[1 + 1j, 2 - 1j, 3], [4j, 5, 6 + 2j]], dtype=np.complex64)
    got = np.asarray(fr.cov(fr.array(m)))
    exp = np.cov(m)
    assert np.iscomplexobj(got)
    assert np.allclose(got, exp)


def test_cov_complex_ddof_rowvar():
    m = np.array([[1 + 1j, 2, 3 - 2j], [4, 5j, 6]])
    got = np.asarray(fr.cov(fr.array(m), rowvar=False, ddof=0))
    exp = np.cov(m, rowvar=False, ddof=0)
    assert np.allclose(got, exp)


def test_cov_real_unchanged():
    m = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = np.asarray(fr.cov(fr.array(m)))
    exp = np.cov(m)
    assert not np.iscomplexobj(got)
    assert np.allclose(got, exp)


# --------------------------------------------------------------------------- #
# corrcoef — complex correlation is also complex
# --------------------------------------------------------------------------- #
def test_corrcoef_complex_matches_numpy():
    m = np.array([[1 + 1j, 2], [3, 4j]])
    got = np.asarray(fr.corrcoef(fr.array(m)))
    exp = np.corrcoef(m)
    assert np.iscomplexobj(got), "corrcoef(complex) must stay complex"
    assert np.allclose(got, exp), f"{got} != {exp}"


def test_corrcoef_real_unchanged():
    m = np.array([[1.0, 2.0, 4.0], [4.0, 5.0, 6.0]])
    got = np.asarray(fr.corrcoef(fr.array(m)))
    exp = np.corrcoef(m)
    assert not np.iscomplexobj(got)
    assert np.allclose(got, exp)


# --------------------------------------------------------------------------- #
# nan_to_num — replaces re AND im nan/inf independently, stays complex
# --------------------------------------------------------------------------- #
def test_nan_to_num_complex_matches_numpy():
    x = np.array([1 + np.nan * 1j])
    got = np.asarray(fr.nan_to_num(fr.array(x)))
    exp = np.nan_to_num(x)
    assert np.iscomplexobj(got), "nan_to_num(complex) must stay complex"
    assert np.allclose(got, exp), f"{got} != {exp}"


def test_nan_to_num_complex_inf_and_finite():
    x = np.array([np.inf + 2j, 3 + np.nan * 1j, -np.inf - np.inf * 1j, 1 + 1j])
    got = np.asarray(fr.nan_to_num(fr.array(x)))
    exp = np.nan_to_num(x)
    assert np.iscomplexobj(got)
    assert np.allclose(got, exp)


def test_nan_to_num_complex_custom_kwargs():
    x = np.array([np.nan + np.nan * 1j, np.inf + 0j, -np.inf + 0j])
    got = np.asarray(fr.nan_to_num(fr.array(x), nan=5.0, posinf=100.0, neginf=-100.0))
    exp = np.nan_to_num(x, nan=5.0, posinf=100.0, neginf=-100.0)
    assert np.allclose(got, exp)


def test_nan_to_num_complex64_matches_numpy():
    x = np.array([np.nan + 1j, 2 + np.inf * 1j], dtype=np.complex64)
    got = np.asarray(fr.nan_to_num(fr.array(x)))
    exp = np.nan_to_num(x)
    assert np.allclose(got, exp)


@pytest.mark.parametrize(
    "x",
    [
        np.array([1.0, np.nan, np.inf, -np.inf]),
        np.array([1.0, np.nan], dtype=np.float32),
        np.array([1, 2, 3], dtype=np.int64),
    ],
)
def test_nan_to_num_real_int_unchanged(x):
    got = np.asarray(fr.nan_to_num(fr.array(x)))
    exp = np.nan_to_num(x)
    assert got.dtype == exp.dtype
    assert np.array_equal(got, exp, equal_nan=False)
