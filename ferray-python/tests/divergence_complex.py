"""Adversarial divergence tests for ferray-python/src/complex.rs vs numpy 2.4.

Oracle is the live `numpy` package. Each test pins a CONFIRMED divergence
between `import ferray as fr` and `import numpy as np` for the complex-array
helpers (real, imag, conj/conjugate, angle, real_if_close, iscomplex,
isreal). Upstream cites refer to numpy/lib/_type_check_impl.py.

These tests are EXPECTED TO FAIL against current ferray; they assert the
numpy-oracle contract that ferray fails to reproduce.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. real_if_close is missing entirely.
#    numpy/lib/_type_check_impl.py:496 — `def real_if_close(a, tol=100):`
#    "If input is complex with all imaginary parts close to zero, return
#     real parts."  ferray does not expose this public symbol at all.
# ---------------------------------------------------------------------------
def test_real_if_close_exists_and_collapses_near_real():
    """numpy _type_check_impl.py:536-537 example:
    np.real_if_close([2.1 + 4e-14j, 5.2 + 3e-15j], tol=1000) -> array([2.1, 5.2])
    """
    inp = [2.1 + 4e-14j, 5.2 + 3e-15j]
    expected = np.real_if_close(inp, tol=1000)
    assert hasattr(fr, "real_if_close"), "ferray is missing public symbol real_if_close"
    got = fr.real_if_close(inp, tol=1000)
    assert got.dtype == expected.dtype  # float64, imaginary dropped
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# 2. real() / imag() of a REAL (integer) array must preserve the input dtype.
#    numpy/lib/_type_check_impl.py:98-100 (real): "If `val` is real, the type
#    of `val` is used for the output."  numpy returns int32; ferray upcasts to
#    float64 because coerce_to_complex() always routes ints through complex128.
# ---------------------------------------------------------------------------
def test_real_of_int_array_preserves_dtype():
    """numpy _type_check_impl.py:98 'If val is real, the type of val is used'."""
    a = np.array([1, 2, 3], dtype=np.int32)
    expected = np.real(a)
    got = fr.real(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


def test_imag_of_int_array_preserves_dtype():
    """numpy _type_check_impl.py:145 'If val is real, the type of val is used'."""
    a = np.array([1, 2, 3], dtype=np.int32)
    expected = np.imag(a)
    got = fr.imag(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


# ---------------------------------------------------------------------------
# 3. conj() / conjugate() of a REAL array must preserve the real dtype.
#    np.conj on a float64/int64 array returns float64/int64 unchanged; ferray
#    coerces every non-complex input to complex128 (complex.rs coerce_to_complex)
#    and so returns complex128.
# ---------------------------------------------------------------------------
def test_conj_of_real_float_preserves_dtype():
    """np.conj on a real array leaves the dtype real (float64)."""
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = np.conj(a)
    got = fr.conj(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


def test_conjugate_of_int_preserves_dtype():
    """np.conjugate on an int64 array returns int64 unchanged."""
    a = np.array([1, 2, 3], dtype=np.int64)
    expected = np.conjugate(a)
    got = fr.conjugate(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


# ---------------------------------------------------------------------------
# 4. angle() must accept the `deg` keyword.
#    np.angle signature is (z, deg=False) — _type_check_impl.py historically /
#    numpy public API. With deg=True the result is in degrees. ferray's
#    #[pyfunction] angle takes only x, so deg=True raises TypeError.
# ---------------------------------------------------------------------------
def test_angle_deg_true():
    """numpy np.angle signature is (z, deg=False); deg=True -> degrees."""
    a = np.array([1j, 1 + 1j])
    expected = np.angle(a, deg=True)  # [90., 45.]
    got = fr.angle(a, deg=True)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# 5. iscomplex / isreal on scalar (or 0-d) input return a SCALAR bool, not a
#    0-d ndarray.  numpy/lib/_type_check_impl.py:210-211:
#       res = zeros(ax.shape, bool)
#       return res[()]   # convert to scalar if needed
#    ferray returns a 0-d ndarray instead.
# ---------------------------------------------------------------------------
def test_iscomplex_scalar_returns_scalar_bool():
    """numpy _type_check_impl.py:211 'return res[()]  # convert to scalar'."""
    expected = np.iscomplex(5)
    got = fr.iscomplex(5)
    assert np.isscalar(got) or isinstance(got, (bool, np.bool_)), (
        f"expected scalar bool, got {type(got).__name__}"
    )
    assert bool(got) == bool(expected)


def test_isreal_scalar_returns_scalar_bool():
    """numpy isreal(scalar) returns a scalar bool (via imag(x) == 0 broadcast)."""
    expected = np.isreal(5)
    got = fr.isreal(5)
    assert np.isscalar(got) or isinstance(got, (bool, np.bool_)), (
        f"expected scalar bool, got {type(got).__name__}"
    )
    assert bool(got) == bool(expected)
