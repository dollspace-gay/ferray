"""Complex `maximum`/`minimum`/`fmax`/`fmin` vs numpy 2.4 (divergence #929).

Pins the fix for the #929 divergence:
  - `fr.maximum`/`fr.minimum` on complex previously raised TypeError; numpy
    registers a complex ufunc loop that selects the whole operand
    LEXICOGRAPHICALLY `(real, then imag)`, NaN-PROPAGATING.
  - `fr.fmax`/`fr.fmin` on complex previously SILENTLY discarded the imaginary
    part (R-CODE-4 data corruption); numpy computes the same lexicographic
    select but NaN-SUPPRESSING.

Every expected value is derived LIVE from numpy (R-CHAR-3) — the assertions
compare ferray's output against the numpy result for the identical inputs,
never against a literal copied from the ferray side.
"""

import numpy as np
import pytest

import ferray as fr

NAN = float("nan")


def _both(name):
    return getattr(fr, name), getattr(np, name)


@pytest.mark.parametrize("op", ["maximum", "minimum", "fmax", "fmin"])
@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_complex_lexicographic_select(op, dtype):
    fop, nop = _both(op)
    a = np.array([1 + 2j, 3 + 1j, 2 + 0j, 0 + 9j], dtype=dtype)
    b = np.array([1 + 5j, 3 - 1j, 5 + 0j, 0 - 9j], dtype=dtype)
    got = np.asarray(fop(fr.array(a), fr.array(b)))
    exp = nop(a, b)
    assert got.dtype == exp.dtype, f"{op}: dtype {got.dtype} != numpy {exp.dtype}"
    assert np.array_equal(got, exp), f"{op}: {got} != numpy {exp}"


@pytest.mark.parametrize("op", ["maximum", "minimum", "fmax", "fmin"])
def test_complex_lexicographic_tie(op):
    # Equal real part, differing imag — the lexicographic tie-break.
    fop, nop = _both(op)
    a = np.array([3 + 1j, 7 - 2j], dtype=np.complex128)
    b = np.array([3 - 1j, 7 + 2j], dtype=np.complex128)
    got = np.asarray(fop(fr.array(a), fr.array(b)))
    exp = nop(a, b)
    assert np.array_equal(got, exp), f"{op}: {got} != numpy {exp}"


def _nan_equal(got, exp):
    # array_equal with NaN==NaN, comparing real and imag parts separately.
    return np.array_equal(got.real, exp.real, equal_nan=True) and np.array_equal(
        got.imag, exp.imag, equal_nan=True
    )


@pytest.mark.parametrize("op", ["maximum", "minimum", "fmax", "fmin"])
def test_complex_nan_handling(op):
    # maximum/minimum propagate NaN; fmax/fmin suppress it. Cover every NaN
    # arrangement: full-NaN, real-NaN-only, imag-NaN-only, both-NaN, and the
    # a-vs-b precedence — all derived live from numpy.
    fop, nop = _both(op)
    a = np.array(
        [
            complex(NAN, NAN),  # full nan in a
            1 + 2j,  # nan in b only
            complex(1, NAN),  # imag-nan in a
            complex(NAN, 5),  # real-nan in a
            complex(NAN, NAN),  # both nan
        ],
        dtype=np.complex128,
    )
    b = np.array(
        [
            1 + 5j,
            complex(NAN, NAN),
            3 + 5j,
            complex(1, NAN),
            complex(NAN, NAN),
        ],
        dtype=np.complex128,
    )
    got = np.asarray(fop(fr.array(a), fr.array(b)))
    exp = nop(a, b)
    assert _nan_equal(got, exp), f"{op}: {got} != numpy {exp}"


def test_fmax_fmin_no_imag_discard():
    # The R-CODE-4 corruption regression guard: fmax/fmin MUST preserve the
    # imaginary part (the prior bug returned [1+0j, 3+0j]).
    a = fr.array([1 + 2j, 3 + 1j])
    b = fr.array([1 + 5j, 3 - 1j])
    assert np.array_equal(np.asarray(fr.fmax(a, b)), np.fmax(np.array([1 + 2j, 3 + 1j]), np.array([1 + 5j, 3 - 1j])))
    assert np.array_equal(np.asarray(fr.fmin(a, b)), np.fmin(np.array([1 + 2j, 3 + 1j]), np.array([1 + 5j, 3 - 1j])))


def test_maximum_minimum_no_longer_raise():
    a = fr.array([1 + 2j, 3 + 1j])
    b = fr.array([1 + 5j, 3 - 1j])
    # Previously TypeError; now returns the lexicographic select.
    assert np.array_equal(np.asarray(fr.maximum(a, b)), np.maximum(np.array([1 + 2j, 3 + 1j]), np.array([1 + 5j, 3 - 1j])))
    assert np.array_equal(np.asarray(fr.minimum(a, b)), np.minimum(np.array([1 + 2j, 3 + 1j]), np.array([1 + 5j, 3 - 1j])))


def test_complex_real_promote():
    # complex op real: real promotes to complex; result is complex.
    a = np.array([1 + 2j, 3 + 1j], dtype=np.complex128)
    b = np.array([2.0, 3.0])
    for op in ("maximum", "minimum", "fmax", "fmin"):
        fop, nop = _both(op)
        got = np.asarray(fop(fr.array(a), fr.array(b)))
        exp = nop(a, b)
        assert got.dtype == exp.dtype
        assert np.array_equal(got, exp), f"{op}: {got} != numpy {exp}"


def test_real_path_unchanged():
    # Real arrays keep numpy's NaN-propagate (max/min) / NaN-suppress (fmax/fmin)
    # float behavior — the fix must not regress the real funnel.
    a = np.array([1.0, NAN, 3.0])
    b = np.array([2.0, 5.0, NAN])
    for op in ("maximum", "minimum", "fmax", "fmin"):
        fop, nop = _both(op)
        got = np.asarray(fop(fr.array(a), fr.array(b)))
        exp = nop(a, b)
        assert np.array_equal(got, exp, equal_nan=True), f"{op}: {got} != numpy {exp}"
