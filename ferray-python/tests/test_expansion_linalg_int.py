"""#971 — integer/bool dtype acceptance for the float-only-sealed linalg
products.

Before #971 `fr.kron`/`tensordot`/`vdot`/`inner`/`dot`/`matmul`/`trace`/
`linalg.matrix_power` raised ``TypeError: unsupported dtype for floating-
point op`` on integer/bool input because their real path routes through the
``match_dtype_float!`` (f32/f64-sealed) dispatch. NumPy COMPUTES on integer
input, preserving the integer dtype for the products and upcasting via the
sum accumulator for ``trace``. The fix delegates the integer/bool case to
``numpy.<fn>`` so the exact dtype contract is owned by numpy.

Every expected value here is derived LIVE from numpy 2.4 (R-CHAR-3) — never
literal-copied from the ferray side. The assertions check BOTH the value and
the exact output dtype against numpy.
"""

import numpy as np
import pytest

import ferray as fr

# Integer/bool dtypes the sealed real path used to reject.
INT_DTYPES = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
ALL_DTYPES = INT_DTYPES + ["bool"]

IV = [1, 2, 3]
IM = [[1, 2], [3, 4]]


def _np(x):
    """Coerce a ferray result to a numpy array for comparison."""
    return np.asarray(x)


def _match(got, expected):
    """Assert ferray's result matches numpy in BOTH value and dtype."""
    got = _np(got)
    expected = np.asarray(expected)
    assert got.dtype == expected.dtype, (
        f"dtype mismatch: ferray {got.dtype} vs numpy {expected.dtype}"
    )
    assert np.array_equal(got, expected), (
        f"value mismatch: ferray {got.tolist()} vs numpy {expected.tolist()}"
    )


# ---------------------------------------------------------------------------
# Vector products — dot / vdot / inner (input int dtype preserved, no upcast).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_dot_int_dtype(dt):
    a = fr.array(np.array(IV, dtype=dt))
    n = np.array(IV, dtype=dt)
    _match(fr.dot(a, a), np.dot(n, n))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_vdot_int_dtype(dt):
    a = fr.array(np.array(IV, dtype=dt))
    n = np.array(IV, dtype=dt)
    _match(fr.vdot(a, a), np.vdot(n, n))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_inner_int_dtype(dt):
    a = fr.array(np.array(IV, dtype=dt))
    n = np.array(IV, dtype=dt)
    _match(fr.inner(a, a), np.inner(n, n))


# ---------------------------------------------------------------------------
# kron — int dtype preserved; bool uses logical-AND products.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_kron_int_dtype(dt):
    a = fr.array(np.array(IV, dtype=dt))
    n = np.array(IV, dtype=dt)
    _match(fr.kron(a, a), np.kron(n, n))


# ---------------------------------------------------------------------------
# Matrix products — matmul / tensordot / matrix_power keep the input dtype.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_matmul_int_dtype(dt):
    a = fr.array(np.array(IM, dtype=dt))
    n = np.array(IM, dtype=dt)
    _match(fr.matmul(a, a), np.matmul(n, n))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_tensordot_int_dtype(dt):
    a = fr.array(np.array(IM, dtype=dt))
    n = np.array(IM, dtype=dt)
    _match(fr.tensordot(a, a), np.tensordot(n, n))


@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_matrix_power_int_dtype(dt):
    a = fr.array(np.array(IM, dtype=dt))
    n = np.array(IM, dtype=dt)
    _match(fr.linalg.matrix_power(a, 2), np.linalg.matrix_power(n, 2))


# ---------------------------------------------------------------------------
# trace — numpy upcasts narrow ints via its sum accumulator (int8 -> int64,
# uint8 -> uint64, bool -> int64). The delegation reproduces that exactly.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dt", ALL_DTYPES)
def test_trace_int_dtype(dt):
    a = fr.array(np.array(IM, dtype=dt))
    n = np.array(IM, dtype=dt)
    _match(fr.trace(a), np.trace(n))


# ---------------------------------------------------------------------------
# float/complex paths must remain UNREGRESSED (still computed by the sealed
# real path / the complex dispatch, not delegated).
# ---------------------------------------------------------------------------
def test_float_dot_unregressed():
    a = fr.array(np.array(IV, dtype="float64"))
    n = np.array(IV, dtype="float64")
    _match(fr.dot(a, a), np.dot(n, n))


def test_complex_dot_unregressed():
    cv = np.array([1 + 2j, 3 + 4j], dtype="complex128")
    got = _np(fr.dot(fr.array(cv), fr.array(cv)))
    expected = np.dot(cv, cv)
    assert got.dtype == expected.dtype
    assert np.allclose(got, expected)
