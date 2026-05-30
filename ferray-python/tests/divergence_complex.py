"""Adversarial divergence tests for ferray-python/src/complex.rs vs numpy 2.4.5.

Oracle is the LIVE `numpy` package (R-CHAR-3): every expected value is produced
by calling numpy in-process, never literal-copied from the ferray side. Each
test pins a CONFIRMED divergence between `import ferray as fr` and
`import numpy as np` for the complex-array helpers (real, imag, conj/conjugate,
angle, real_if_close, iscomplex, isreal).

Upstream cites:
  real/imag/iscomplex/isreal/real_if_close — numpy/lib/_type_check_impl.py
  angle                                     — numpy/lib/_function_base_impl.py:1694

These tests are EXPECTED TO FAIL against the current ferray binding; they assert
the numpy-oracle contract that ferray fails to reproduce. The root cause for the
dtype-upcast group is `coerce_to_complex()` in complex.rs, which routes every
non-complex input through complex128 (or complex64) before the op, dropping the
input's real dtype. The scalar group is the binding always returning an ndarray
where numpy collapses a scalar/0-d result via `res[()]`.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# Group 1 — real() / imag() of a REAL array must PRESERVE the input dtype.
#   numpy/lib/_type_check_impl.py:86-125 real(): docstring lines 98-100
#     "If `val` is real, the type of `val` is used for the output."
#   real()/imag() of a real array return `val.real`/`val.imag`, which for a
#   real ndarray is the array itself (real) or zeros of the SAME dtype (imag).
#   ferray coerce_to_complex() upcasts int32 -> complex128 -> float64 result.
# ---------------------------------------------------------------------------
def test_real_of_int32_preserves_dtype():
    a = np.array([1, 2, 3], dtype=np.int32)
    expected = np.real(a)  # int32
    got = fr.real(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


def test_imag_of_int32_preserves_dtype():
    a = np.array([1, 2, 3], dtype=np.int32)
    expected = np.imag(a)  # int32 zeros
    got = fr.imag(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


# ---------------------------------------------------------------------------
# Group 2 — conj() / conjugate() of a REAL array must PRESERVE the real dtype.
#   np.conj/np.conjugate are the ufuncs numpy.conjugate; on a real input the
#   conjugate is the input unchanged and the dtype stays real. ferray
#   coerce_to_complex() forces complex128, so a float64/int64 input comes back
#   complex128 with a spurious -0.j.
# ---------------------------------------------------------------------------
def test_conj_of_float64_preserves_dtype():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    expected = np.conj(a)  # float64
    got = fr.conj(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


def test_conjugate_of_int64_preserves_dtype():
    a = np.array([1, 2, 3], dtype=np.int64)
    expected = np.conjugate(a)  # int64
    got = fr.conjugate(a)
    assert got.dtype == expected.dtype, f"expected {expected.dtype}, got {got.dtype}"
    assert got.tolist() == expected.tolist()


# ---------------------------------------------------------------------------
# Group 3 — angle() must accept the `deg` keyword.
#   numpy/lib/_function_base_impl.py:1694 `def angle(z, deg=False):`
#   lines 1742-1743: `if deg: a *= 180 / pi`. ferray's #[pyfunction] angle
#   takes only `x`, so deg=True raises TypeError at the binding.
# ---------------------------------------------------------------------------
def test_angle_deg_kwarg():
    a = np.array([1j, 1 + 1j])
    expected = np.angle(a, deg=True)  # [90., 45.]
    got = fr.angle(a, deg=True)
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Group 4 — real_if_close is MISSING entirely.
#   numpy/lib/_type_check_impl.py:496 `def real_if_close(a, tol=100):`
#   in numpy's `__all__` (line 8). ferray exposes no such public symbol.
# ---------------------------------------------------------------------------
def test_real_if_close_exists():
    assert hasattr(fr, "real_if_close"), "ferray is missing public symbol real_if_close"


def test_real_if_close_collapses_near_real():
    inp = [2.1 + 4e-14j, 5.2 + 3e-15j]
    expected = np.real_if_close(inp, tol=1000)  # array([2.1, 5.2]) float64
    got = fr.real_if_close(inp, tol=1000)
    assert got.dtype == expected.dtype
    assert np.allclose(got, expected)


def test_real_if_close_does_not_collapse_when_above_tol():
    inp = [2.1 + 4e-13j, 5.2 + 3e-15j]
    expected = np.real_if_close(inp, tol=1000)  # stays complex128
    got = fr.real_if_close(inp, tol=1000)
    assert got.dtype == expected.dtype
    assert np.allclose(got, expected)


# ---------------------------------------------------------------------------
# Group 5 — iscomplex / isreal on a SCALAR input return a scalar bool, not a
#   0-d ndarray.
#   numpy/lib/_type_check_impl.py:210-211:
#       res = zeros(ax.shape, bool)
#       return res[()]   # convert to scalar if needed
#   isreal -> imag(x) == 0 broadcasts to the same scalar. ferray returns a
#   0-d ndarray (np.zeros((), 'bool')) instead of a scalar.
# ---------------------------------------------------------------------------
def test_iscomplex_scalar_returns_scalar_bool():
    expected = np.iscomplex(5)  # np.False_ (numpy bool scalar, 0-dim)
    got = fr.iscomplex(5)
    assert np.ndim(got) == np.ndim(expected) == 0
    assert not isinstance(got, np.ndarray), f"expected scalar bool, got {type(got).__name__}"
    assert bool(got) == bool(expected)


def test_isreal_scalar_returns_scalar_bool():
    expected = np.isreal(5)  # python/numpy bool scalar
    got = fr.isreal(5)
    assert np.ndim(got) == np.ndim(expected) == 0
    assert not isinstance(got, np.ndarray), f"expected scalar bool, got {type(got).__name__}"
    assert bool(got) == bool(expected)


# ---------------------------------------------------------------------------
# Group 6 — real / imag / angle on a SCALAR input return a numpy scalar, not a
#   0-d ndarray.
#   real()/imag() return `val.real`/`val.imag` (numpy/lib/_type_check_impl.py
#   :122-125, :166-169); for a python complex scalar that is a python float.
#   angle() returns `arctan2(...)` (numpy/lib/_function_base_impl.py:1741) which
#   for a scalar input is a numpy float64 scalar (ndim 0, not an ndarray).
# ---------------------------------------------------------------------------
def test_real_scalar_returns_scalar():
    expected = np.real(1 + 2j)  # 1.0 (python float)
    got = fr.real(1 + 2j)
    assert np.ndim(got) == 0
    assert not isinstance(got, np.ndarray), f"expected scalar, got {type(got).__name__}"
    assert float(got) == float(expected)


def test_imag_scalar_returns_scalar():
    expected = np.imag(1 + 2j)  # 2.0
    got = fr.imag(1 + 2j)
    assert np.ndim(got) == 0
    assert not isinstance(got, np.ndarray), f"expected scalar, got {type(got).__name__}"
    assert float(got) == float(expected)


def test_angle_scalar_returns_scalar():
    expected = np.angle(1 + 1j)  # np.float64(0.7853981633974483), ndim 0
    got = fr.angle(1 + 1j)
    assert np.ndim(got) == 0
    assert not isinstance(got, np.ndarray), f"expected scalar, got {type(got).__name__}"
    assert np.allclose(float(got), float(expected))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
