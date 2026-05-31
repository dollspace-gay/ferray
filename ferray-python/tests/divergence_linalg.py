"""Adversarial divergence tests: ferray.linalg vs the live numpy oracle.

Each test pins a CONTRACT divergence (not a ULP difference) where
``import ferray as fr`` fails to match ``numpy`` 2.4.x. The oracle is the
live installed numpy (R-CHAR-3 — every expected value is produced by a
live ``np.*`` call, never literal-copied from ferray). Values use
``np.allclose``/``np.isclose``; shapes, dtypes and exception TYPES are exact.

Upstream citations reference /home/doll/numpy-ref/numpy/linalg/_linalg.py.

These tests are EXPECTED TO FAIL on current ferray (they pin bugs). The fix
lives DOWN in the owning crate (ferray-linalg) or in the marshalling shim
(ferray-python/src/linalg.rs + lib.rs registration), per goal.md.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. Top-level vector/matrix product API surface.
#
# `np.dot`, `np.vdot`, `np.matmul`, `np.inner`, `np.outer` are TOP-LEVEL
# numpy functions (numpy/_core/multiarray + numpy/__init__.pyi), not under
# numpy.linalg. A drop-in replacement (`import ferray as np`) must expose
# them at the package root. ferray only registers them under fr.linalg.*.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dot", "vdot", "matmul", "inner", "outer"])
def test_top_level_product_symbol_exists(name):
    """`np.<name>` is top-level in numpy; `fr.<name>` must mirror it."""
    assert hasattr(np, name)  # oracle: numpy exposes it at top level
    assert hasattr(fr, name), f"ferray is missing top-level fr.{name}"


def test_top_level_dot_matches_numpy():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    expected = float(np.dot(a, b))
    assert np.isclose(float(fr.dot(a, b)), expected)


def test_top_level_matmul_matches_numpy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = np.matmul(a, b)
    np.testing.assert_allclose(np.asarray(fr.matmul(a, b)), expected)


def test_top_level_vdot_matches_numpy():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    expected = float(np.vdot(a, b))
    assert np.isclose(float(fr.vdot(a, b)), expected)


# ---------------------------------------------------------------------------
# 2. numpy.linalg.LinAlgError must be exposed AND raised.
#
# _linalg.py:115:  class LinAlgError(ValueError):
# _linalg.py:145:  raise LinAlgError("Singular matrix")
#
# LinAlgError IS a subclass of ValueError, so `pytest.raises(LinAlgError)`
# will NOT catch a bare ValueError — that is exactly the divergence: ferray
# raises bare builtins.ValueError, not numpy.linalg.LinAlgError.
# ---------------------------------------------------------------------------


def test_linalg_error_symbol_exists():
    """numpy.linalg.LinAlgError is public (_linalg.py:115); ferray must expose it."""
    assert hasattr(np.linalg, "LinAlgError")  # oracle
    assert hasattr(fr.linalg, "LinAlgError"), "fr.linalg.LinAlgError is missing"


def test_inv_singular_raises_linalgerror():
    """inv of a singular matrix raises LinAlgError, per _linalg.py:145."""
    sing = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank 1, singular
    with pytest.raises(np.linalg.LinAlgError):  # oracle: numpy raises this
        np.linalg.inv(sing)
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.inv(sing)


def test_solve_singular_raises_linalgerror():
    """solve with a singular A raises LinAlgError, per _linalg.py:145."""
    sing = np.array([[1.0, 2.0], [2.0, 4.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(sing, b)
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.solve(sing, b)


def test_cholesky_non_pd_raises_linalgerror():
    """cholesky of a non-positive-definite matrix raises LinAlgError."""
    npd = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, indefinite
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(npd)
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.cholesky(npd)


def test_inv_nonsquare_raises_linalgerror():
    """inv of a non-square matrix raises LinAlgError, not bare ValueError."""
    ns = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(ns)
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.inv(ns)


def test_det_nonsquare_raises_linalgerror():
    """det of a non-square matrix raises LinAlgError, not bare ValueError."""
    ns = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.det(ns)
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.det(ns)


def test_solve_nonsquare_raises_linalgerror():
    """solve with a non-square A raises LinAlgError, not bare ValueError."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = np.array([1.0, 2.0])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(a, b)
    with pytest.raises(np.linalg.LinAlgError):
        fr.linalg.solve(a, b)


# ---------------------------------------------------------------------------
# 3. Integer-input dtype promotion (_commonType -> float64).
#
# _linalg.py _commonType promotes integer/bool inputs to float64 before any
# LAPACK call. det/inv/norm/solve of an int array must NOT raise — they
# return float64. ferray's match_dtype_float! macro rejects int dtypes with
# a TypeError instead of promoting.
# ---------------------------------------------------------------------------


def test_det_integer_input_promotes_to_float64():
    ai = np.array([[1, 2], [3, 4]])
    expected = np.linalg.det(ai)  # oracle, float64
    got = fr.linalg.det(ai)
    assert np.asarray(got).dtype == np.float64
    assert np.isclose(float(got), float(expected))


def test_inv_integer_input_promotes_to_float64():
    ai = np.array([[1, 2], [3, 5]])  # det = -1, invertible
    expected = np.linalg.inv(ai)  # oracle, float64
    got = np.asarray(fr.linalg.inv(ai))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_norm_integer_input_promotes_to_float64():
    ai = np.array([3, 4])
    expected = np.linalg.norm(ai)  # oracle: 5.0, float64
    got = fr.linalg.norm(ai)
    assert np.asarray(got).dtype == np.float64
    assert np.isclose(float(got), float(expected))


def test_solve_integer_input_promotes_to_float64():
    a = np.array([[2, 1], [1, 3]])
    b = np.array([5, 10])
    expected = np.linalg.solve(a, b)  # oracle, float64
    got = np.asarray(fr.linalg.solve(a, b))
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. qr mode='r' returns a SINGLE R ndarray (not a (Q, R) tuple).
#
# _linalg.py: if mode == 'r': ... return wrap(r)  -> bare ndarray.
# ferray maps 'r' onto Reduced and still returns (Q, R).
# ---------------------------------------------------------------------------


def test_qr_mode_r_returns_single_r_matrix():
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected_r = np.linalg.qr(a, mode="r")  # oracle: bare ndarray, shape (2, 2)
    assert not isinstance(expected_r, tuple)  # confirm oracle shape of contract
    got = fr.linalg.qr(a, mode="r")
    assert not isinstance(got, tuple), (
        "qr(mode='r') must return a single R ndarray, not a (Q, R) tuple"
    )
    got_arr = np.asarray(got)
    assert got_arr.shape == expected_r.shape
    # R is unique only up to row signs; compare magnitudes.
    np.testing.assert_allclose(np.abs(got_arr), np.abs(expected_r), atol=1e-10)


# ---------------------------------------------------------------------------
# 5. svd(compute_uv=False) returns ONLY the 1-D singular values.
#
# _linalg.py: def svd(a, full_matrices=True, compute_uv=True, hermitian=False)
# ferray's svd has no compute_uv kwarg at all (TypeError on the keyword).
# ---------------------------------------------------------------------------


def test_svd_compute_uv_false_returns_singular_values_only():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected_s = np.linalg.svd(a, compute_uv=False)  # oracle: 1-D
    got = fr.linalg.svd(a, compute_uv=False)
    got_arr = np.asarray(got)
    assert got_arr.ndim == 1, "compute_uv=False must yield a 1-D singular-value array"
    np.testing.assert_allclose(got_arr, expected_s, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Matrix norm with negative integer ord.
#
# _linalg.py norm(): for a 2-D matrix, ord=-1 is the MIN column abs-sum and
# ord=-2 is the smallest singular value — both valid matrix norms. ferray
# misroutes negative ints to a vector p-norm P(-1)/P(-2) and raises.
# ---------------------------------------------------------------------------


def test_matrix_norm_ord_neg1():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    expected = np.linalg.norm(a, ord=-1)  # oracle: 4.0 (min column abs-sum)
    got = float(fr.linalg.norm(a, ord=-1))
    assert np.isclose(got, float(expected))


def test_matrix_norm_ord_neg2():
    a = np.array([[3.0, 0.0], [0.0, 1.0]])
    expected = np.linalg.norm(a, ord=-2)  # oracle: 1.0 (smallest singular value)
    got = float(fr.linalg.norm(a, ord=-2))
    assert np.isclose(got, float(expected))


# ---------------------------------------------------------------------------
# 7. matmul of two 1-D arrays returns a 0-D scalar (the inner product).
#
# numpy.matmul / `@`: for 1-D x 1-D, the result is the scalar dot product
# (ndim 0). ferray raises ValueError "cannot multiply two 1D arrays".
# ---------------------------------------------------------------------------


def test_matmul_1d_1d_returns_scalar():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    expected = np.matmul(a, b)  # oracle: 32.0, ndim 0
    assert np.asarray(expected).ndim == 0
    got = fr.linalg.matmul(a, b)
    got_arr = np.asarray(got)
    assert got_arr.ndim == 0, "matmul(1d, 1d) must collapse to a scalar"
    assert np.isclose(float(got_arr), float(expected))


# ---------------------------------------------------------------------------
# 8. numpy.linalg.eigvals — eigenvalues of a general (non-symmetric) matrix.
#
# Public numpy.linalg.eigvals exists; ferray exposes only eigvalsh
# (symmetric). A general matrix has complex eigenvalues eigvalsh cannot give.
# ---------------------------------------------------------------------------


def test_eigvals_symbol_exists():
    assert hasattr(np.linalg, "eigvals")  # oracle
    assert hasattr(fr.linalg, "eigvals"), "fr.linalg.eigvals is missing"


# ---------------------------------------------------------------------------
# 6. ACToR critic sweep (#985 trace direct-edit): the native float-dtype
#    `trace` path returns a 0-D `ndarray`, but numpy.trace of a 2-D matrix
#    returns a TRUE numpy SCALAR (np.float64 / np.complex128), for which
#    `np.isscalar(...)` is True. ferray's OWN integer/bool `trace` path (which
#    delegates to numpy, #971) and complex path return a true scalar, so the
#    float native path is BOTH wrong vs numpy AND internally inconsistent.
#
#    numpy upstream: `numpy/_core/fromnumeric.py trace()` -> ndarray.trace,
#    whose 2-D reduction returns a numpy scalar (verified live:
#    `np.isscalar(np.trace(np.eye(2))) is True`,
#    `type(np.trace(np.eye(2))) is np.float64`).
# ---------------------------------------------------------------------------


def test_trace_float_returns_numpy_scalar_not_0d_array():
    """np.trace(2-D float) is a numpy scalar; ferray returns a 0-D ndarray."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.trace(m)  # live oracle
    assert np.isscalar(oracle)  # numpy: True (numpy scalar, not 0-D array)
    result = fr.linalg.trace(m)
    # Pin: ferray must also yield a numpy scalar, matching numpy's return type.
    assert np.isscalar(result), (
        f"fr.linalg.trace returned {type(result).__name__} "
        f"(np.isscalar={np.isscalar(result)}); numpy returns a scalar "
        f"{type(oracle).__name__}"
    )


def test_trace_complex_returns_numpy_scalar_not_0d_array():
    """np.trace(2-D complex) is np.complex128; ferray returns a 0-D ndarray."""
    m = np.array([[1 + 1j, 2 + 0j], [3 + 0j, 4 + 2j]])
    oracle = np.trace(m)  # live oracle -> np.complex128(5+3j)
    assert np.isscalar(oracle)  # numpy: True
    assert type(oracle).__name__ == "complex128"
    result = fr.linalg.trace(m)
    assert np.isscalar(result), (
        f"fr.linalg.trace(complex) returned {type(result).__name__} "
        f"(np.isscalar={np.isscalar(result)}); numpy returns np.complex128 scalar"
    )


def test_trace_float_return_type_matches_int_path():
    """ferray's int `trace` returns a scalar; the float path must too (internal
    consistency + numpy parity). numpy oracle: both ranks scalar."""
    mi = np.array([[1, 2], [3, 4]])
    mf = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Oracle: numpy returns a scalar for both int and float input.
    assert np.isscalar(np.trace(mi)) and np.isscalar(np.trace(mf))
    int_scalar = np.isscalar(fr.linalg.trace(mi))   # ferray int path -> True
    float_scalar = np.isscalar(fr.linalg.trace(mf))  # ferray float path -> False (bug)
    assert int_scalar == float_scalar, (
        "ferray.linalg.trace is inconsistent: int path scalar="
        f"{int_scalar}, float path scalar={float_scalar}; numpy: both True"
    )


# ---------------------------------------------------------------------------
# SCALAR-RETURN contract ($OUT_SCALAR): linalg functions whose FULL result is
# a single scalar must return a numpy SCALAR (np.float64 / np.complex128 /
# np.int64), NOT a 0-D ndarray. numpy's _linalg.py wrappers collapse the 0-D
# egress to a numpy scalar (matrix_rank/trace already fixed in ferray; det /
# norm (whole-array, no axis) / cond / slogdet still return a 0-D ndarray).
#
# Oracle (live): every expected TYPE/VALUE is produced by a `np.linalg.*` call
# in-test, never literal-copied from ferray (R-CHAR-3).
# Upstream: numpy/linalg/_linalg.py det:2120, slogdet:2244, norm:2599,
# cond:1846, matrix_rank:2059.
# ---------------------------------------------------------------------------


def test_det_float_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.det(2-D float) -> np.float64 scalar; ferray returns a 0-D
    ndarray (linalg.rs det: `arr0.into_pyarray` with no scalarize).
    Upstream: numpy/linalg/_linalg.py:2120 det."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.det(m)  # live -> np.float64(-2.0000000000000004)
    assert not isinstance(oracle, np.ndarray)  # numpy: scalar
    result = fr.linalg.det(m)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.det(float) returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar (not a 0-D ndarray)"
    )
    assert type(result) == type(oracle)


def test_det_complex_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.det(2-D complex) -> np.complex128 scalar; ferray returns a
    0-D ndarray. Upstream: numpy/linalg/_linalg.py:2120 det."""
    m = np.array([[1 + 1j, 2 + 0j], [1j, 1 + 0j]])
    oracle = np.linalg.det(m)  # live -> np.complex128(1-1j)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.det(m)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.det(complex) returned {type(result).__name__}; "
        "numpy returns an np.complex128 scalar"
    )
    assert type(result) == type(oracle)


def test_norm_vector_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.norm(1-D, ord=None) -> np.float64 scalar; ferray returns a
    0-D ndarray. Upstream: numpy/linalg/_linalg.py:2599 norm."""
    v = np.array([3.0, 4.0])
    oracle = np.linalg.norm(v)  # live -> np.float64(5.0)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.norm(v)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.norm(vector) returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar"
    )
    assert type(result) == type(oracle)


def test_norm_matrix_frobenius_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.norm(2-D, ord=None) Frobenius -> np.float64 scalar; ferray
    returns a 0-D ndarray. Upstream: numpy/linalg/_linalg.py:2599 norm."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.norm(m)  # live -> np.float64(5.477225575051661)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.norm(m)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.norm(matrix-fro) returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar"
    )
    assert type(result) == type(oracle)


def test_norm_matrix_nuc_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.norm(2-D, 'nuc') -> np.float64 scalar; ferray returns a 0-D
    ndarray. Upstream: numpy/linalg/_linalg.py:2599 norm."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.norm(m, "nuc")  # live -> np.float64(5.8309518948453)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.norm(m, "nuc")
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.norm(matrix-nuc) returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar"
    )
    assert type(result) == type(oracle)


def test_norm_matrix_ord1_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.norm(2-D, 1) max-col-abs-sum -> np.float64 scalar; ferray
    returns a 0-D ndarray. Upstream: numpy/linalg/_linalg.py:2599 norm."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.norm(m, 1)  # live -> np.float64(6.0)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.norm(m, 1)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.norm(matrix-ord1) returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar"
    )
    assert type(result) == type(oracle)


def test_norm_matrix_ord_inf_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.norm(2-D, inf) max-row-abs-sum -> np.float64 scalar; ferray
    returns a 0-D ndarray. Upstream: numpy/linalg/_linalg.py:2599 norm."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.norm(m, np.inf)  # live -> np.float64(7.0)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.norm(m, np.inf)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.norm(matrix-inf) returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar"
    )
    assert type(result) == type(oracle)


def test_norm_with_axis_correctly_stays_ndarray():
    """CONTROL (no divergence expected): an `axis=` norm that leaves dims
    returns an ndarray in BOTH numpy and ferray — the scalarize fix must be a
    no-op here. Upstream: numpy/linalg/_linalg.py:2599 norm (axis branch)."""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    oracle = np.linalg.norm(a, axis=0)  # live -> 1-D ndarray, NOT a scalar
    assert isinstance(oracle, np.ndarray)
    result = fr.linalg.norm(a, axis=0)
    assert isinstance(result, np.ndarray), (
        "axis-norm must stay an ndarray; scalarize must not collapse it"
    )
    assert np.allclose(result, oracle)


def test_cond_returns_numpy_scalar_not_0d_ndarray():
    """np.linalg.cond(2-D) -> np.float64 scalar; ferray returns a 0-D ndarray
    (linalg.rs cond: `arr0.into_pyarray` with no scalarize).
    Upstream: numpy/linalg/_linalg.py:1846 cond."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.cond(m)  # live -> np.float64(14.933034373659268)
    assert not isinstance(oracle, np.ndarray)
    result = fr.linalg.cond(m)
    assert np.isclose(result, oracle)
    assert not isinstance(result, np.ndarray), (
        f"fr.linalg.cond returned {type(result).__name__}; "
        "numpy returns an np.float64 scalar"
    )
    assert type(result) == type(oracle)


def test_slogdet_float_sign_element_is_numpy_scalar():
    """np.linalg.slogdet(2-D float).sign -> np.float64 scalar; ferray's tuple
    element is a 0-D ndarray. Upstream: numpy/linalg/_linalg.py:2244 slogdet."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.slogdet(m)  # live: sign=-1.0, logabsdet=0.6931...
    osign = oracle.sign
    assert not isinstance(osign, np.ndarray)
    fsign = fr.linalg.slogdet(m)[0]
    assert np.isclose(fsign, osign)
    assert not isinstance(fsign, np.ndarray), (
        f"fr.linalg.slogdet(float)[0] (sign) is {type(fsign).__name__}; "
        "numpy's sign is an np.float64 scalar"
    )
    assert type(fsign) == type(osign)


def test_slogdet_float_logabsdet_element_is_numpy_scalar():
    """np.linalg.slogdet(2-D float).logabsdet -> np.float64 scalar; ferray's
    tuple element is a 0-D ndarray. Upstream: numpy/linalg/_linalg.py:2244."""
    m = np.array([[1.0, 2.0], [3.0, 4.0]])
    oracle = np.linalg.slogdet(m)
    olog = oracle.logabsdet  # live -> np.float64(0.6931471805599455)
    assert not isinstance(olog, np.ndarray)
    flog = fr.linalg.slogdet(m)[1]
    assert np.isclose(flog, olog)
    assert not isinstance(flog, np.ndarray), (
        f"fr.linalg.slogdet(float)[1] (logabsdet) is {type(flog).__name__}; "
        "numpy's logabsdet is an np.float64 scalar"
    )
    assert type(flog) == type(olog)


def test_slogdet_complex_sign_element_is_numpy_scalar():
    """np.linalg.slogdet(2-D complex).sign -> np.complex128 scalar; ferray's
    tuple element is a 0-D ndarray. Upstream: numpy/linalg/_linalg.py:2244."""
    m = np.array([[1 + 1j, 2 + 0j], [1j, 1 + 0j]])
    oracle = np.linalg.slogdet(m)
    osign = oracle.sign  # live -> np.complex128 unit-modulus
    assert not isinstance(osign, np.ndarray)
    fsign = fr.linalg.slogdet(m)[0]
    assert np.isclose(fsign, osign)
    assert not isinstance(fsign, np.ndarray), (
        f"fr.linalg.slogdet(complex)[0] (sign) is {type(fsign).__name__}; "
        "numpy's sign is an np.complex128 scalar"
    )
    assert type(fsign) == type(osign)


def test_slogdet_complex_logabsdet_element_is_numpy_scalar():
    """np.linalg.slogdet(2-D complex).logabsdet -> np.float64 scalar (REAL,
    even for complex input); ferray's tuple element is a 0-D ndarray.
    Upstream: numpy/linalg/_linalg.py:2244 slogdet."""
    m = np.array([[1 + 1j, 2 + 0j], [1j, 1 + 0j]])
    oracle = np.linalg.slogdet(m)
    olog = oracle.logabsdet  # live -> np.float64(0.3465735902799727)
    assert not isinstance(olog, np.ndarray)
    flog = fr.linalg.slogdet(m)[1]
    assert np.isclose(flog, olog)
    assert not isinstance(flog, np.ndarray), (
        f"fr.linalg.slogdet(complex)[1] (logabsdet) is {type(flog).__name__}; "
        "numpy's logabsdet is an np.float64 scalar"
    )
    assert type(flog) == type(olog)
