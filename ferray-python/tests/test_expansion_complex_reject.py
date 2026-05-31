"""Real-only binary ufuncs must RAISE TypeError on a complex operand (#936).

numpy registers ONLY float (or int) loops for these ufuncs — it has no complex
loop, so a complex input on EITHER operand raises ``TypeError`` (the exact
message family ``ufunc 'X' not supported for the input types``). ferray
previously silently cast complex -> real, discarding the imaginary part and
returning a wrong real result (R-CODE-4). These tests pin the numpy contract:
complex on either side -> TypeError; real operands -> unchanged byte-identical.

Every expected value is derived LIVE from numpy (R-CHAR-3), never copied from
the ferray side.
"""

import numpy as np
import pytest

import ferray as fr

# The real-only binary ufuncs that numpy rejects on complex input. The named 7
# from the dispatch plus the two silent-cast siblings found in the same file
# (logaddexp2 via the same float-promote macro; ldexp via its hand-written
# binding).
REAL_ONLY_BINARY = [
    "heaviside",
    "copysign",
    "nextafter",
    "hypot",
    "logaddexp",
    "logaddexp2",
    "fmod",
    "arctan2",
    "ldexp",
]


def _confirm_numpy_raises(name, a, b):
    """Assert numpy itself raises TypeError — keeps the oracle honest."""
    npf = getattr(np, name)
    with pytest.raises(TypeError):
        npf(a, b)


@pytest.mark.parametrize("name", REAL_ONLY_BINARY)
def test_complex_first_operand_raises(name):
    z = np.array([1 + 2j])
    r = np.array([1.0])
    _confirm_numpy_raises(name, z, r)
    frf = getattr(fr, name)
    with pytest.raises(TypeError):
        frf(fr.array([1 + 2j]), fr.array([1.0]))


@pytest.mark.parametrize("name", REAL_ONLY_BINARY)
def test_complex_second_operand_raises(name):
    r = np.array([1.0])
    z = np.array([1 + 2j])
    _confirm_numpy_raises(name, r, z)
    frf = getattr(fr, name)
    with pytest.raises(TypeError):
        frf(fr.array([1.0]), fr.array([1 + 2j]))


def test_real_operands_unchanged():
    """The real compute path stays byte-identical to numpy (non-regression)."""
    a = np.array([0.5, -2.0, 3.25, 0.0])
    b = np.array([1.0, 0.5, -1.5, 2.0])

    cases = {
        "heaviside": (np.heaviside, fr.heaviside, a, b),
        "copysign": (np.copysign, fr.copysign, a, b),
        "nextafter": (np.nextafter, fr.nextafter, a, b),
        "hypot": (np.hypot, fr.hypot, a, b),
        "logaddexp": (np.logaddexp, fr.logaddexp, a, b),
        "logaddexp2": (np.logaddexp2, fr.logaddexp2, a, b),
        "fmod": (np.fmod, fr.fmod, a, b),
        "arctan2": (np.arctan2, fr.arctan2, a, b),
    }
    for name, (npf, frf, x, y) in cases.items():
        expected = npf(x, y)
        got = np.asarray(frf(fr.array(x), fr.array(y)))
        np.testing.assert_array_equal(got, expected, err_msg=name)
        assert got.dtype == expected.dtype, name

    # ldexp: second operand is an integer exponent.
    n = np.array([1, 2, 0, -1])
    expected = np.ldexp(a, n)
    got = np.asarray(fr.ldexp(fr.array(a), fr.array(n)))
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype
