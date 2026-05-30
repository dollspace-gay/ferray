"""Adversarial divergence tests for ferray.fft vs numpy 2.4.x.

Each test pins a CONFIRMED divergence between ``import ferray as fr`` and the
live ``numpy`` oracle. Values compared with ``np.allclose`` (complex-safe);
shape/dtype/exception-type compared exactly. Every test cites the numpy
upstream source it encodes.

Written by an adversarial critic. These are EXPECTED TO FAIL against the
current ferray build — they document where ferray diverges from numpy.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# 1. fftshift / ifftshift reject an integer ``axes`` argument.
#
# numpy/fft/_helper.py:69  (fftshift)
#     elif isinstance(axes, integer_types):
#         shift = x.shape[axes] // 2
# numpy/fft/_helper.py:117 (ifftshift)
#     elif isinstance(axes, integer_types):
#         shift = -(x.shape[axes] // 2)
#
# numpy explicitly accepts a scalar int for ``axes`` (shift one axis).
# ferray's signature only accepts ``Option<Vec<isize>>`` and raises
# TypeError on a bare int.
# ---------------------------------------------------------------------------


def test_fftshift_accepts_int_axes():
    """numpy _helper.py:69 — fftshift(x, axes=<int>) shifts a single axis."""
    m = np.arange(12).reshape(3, 4)
    expected = np.fft.fftshift(m, axes=0)
    got = np.asarray(fr.fft.fftshift(m, axes=0))
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_ifftshift_accepts_int_axes():
    """numpy _helper.py:117 — ifftshift(x, axes=<int>) shifts a single axis."""
    m = np.arange(12).reshape(3, 4)
    expected = np.fft.ifftshift(m, axes=1)
    got = np.asarray(fr.fft.ifftshift(m, axes=1))
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


# ---------------------------------------------------------------------------
# 2. fft / ifft with n < 1 must raise ValueError, not OverflowError.
#
# numpy/fft/_pocketfft.py:58
#     def _raw_fft(a, n, axis, is_real, is_forward, norm, out=None):
#         if n < 1:
#             raise ValueError(f"Invalid number of FFT data points ({n})
#                               specified.")
#
# ferray binds ``n: Option<usize>``; a negative int triggers a PyO3
# OverflowError ("can't convert negative int to unsigned") at the boundary
# before the ferray length check runs. numpy raises ValueError.
# ---------------------------------------------------------------------------


def test_fft_negative_n_raises_valueerror():
    """numpy _pocketfft.py:59 — n < 1 raises ValueError, not OverflowError."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        fr.fft.fft(a, n=-1)


def test_ifft_negative_n_raises_valueerror():
    """numpy _pocketfft.py:59 — ifft n < 1 raises ValueError."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError):
        fr.fft.ifft(a, n=-1)


# ---------------------------------------------------------------------------
# 3. fftfreq / rfftfreq with negative n must raise ValueError.
#
# numpy/fft/_helper.py:168
#     if not isinstance(n, integer_types):
#         raise ValueError("n should be an integer")
#     val = 1.0 / (n * d)
#     ...
#     p2 = arange(-(n // 2), 0, dtype=int, device=device)
#
# numpy accepts a negative Python int for ``n`` and raises ValueError from
# the downstream ``empty(n, ...)`` (negative dimensions). ferray binds
# ``n: usize`` so a negative int is an OverflowError at the PyO3 boundary.
# ---------------------------------------------------------------------------


def test_fftfreq_negative_n_raises_valueerror():
    """numpy _helper.py:168-177 — fftfreq(-2) raises ValueError, not Overflow."""
    with pytest.raises(ValueError):
        fr.fft.fftfreq(-2)


def test_rfftfreq_negative_n_raises_valueerror():
    """numpy _helper.py:230-235 — rfftfreq(-2) raises ValueError, not Overflow."""
    with pytest.raises(ValueError):
        fr.fft.rfftfreq(-2)


# ---------------------------------------------------------------------------
# 4. fftn / ifftn with s containing -1 means "use whole input" (numpy 2.0).
#
# numpy/fft/_pocketfft.py:736-737  (_cook_nd_args)
#     # use the whole input array along axis `i` if `s[i] == -1`
#     s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
#
# numpy 2.0 added s[i] == -1 as a sentinel ("versionchanged:: 2.0 If it is
# -1, the whole input is used"). ferray binds ``s: Option<Vec<usize>>`` so a
# -1 entry is an OverflowError at the PyO3 boundary instead of being honored.
# ---------------------------------------------------------------------------


def test_fftn_s_minus_one_uses_whole_input():
    """numpy _pocketfft.py:737 — s[i] == -1 means use the whole input axis."""
    m = np.arange(12.0).reshape(3, 4)
    expected = np.fft.fftn(m, s=[-1, -1], axes=[0, 1])
    got = np.asarray(fr.fft.fftn(m, s=[-1, -1], axes=[0, 1]))
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# 5. rfft / rfft2 / rfftn / ihfft on COMPLEX input must raise TypeError.
#
# numpy's real-input transforms reject complex arrays:
#     >>> np.fft.rfft(np.array([1+1j, 2+0j]))
#     TypeError: ufunc 'rfft_n_even' not supported for the input types ...
# (numpy/fft/_pocketfft.py:81 dispatches to the pfu.rfft_* real ufunc, which
# has no complex loop.)
#
# ferray instead SILENTLY casts complex -> float (discarding the imaginary
# part) via coerce_dtype, producing a result for input numpy refuses. This is
# a silent lossy round-trip across the boundary (violates the real-input
# contract): ferray returns a value where numpy raises.
# ---------------------------------------------------------------------------


def test_rfft_complex_input_raises_typeerror():
    """numpy _pocketfft.py:81 — rfft has no complex ufunc loop; raises TypeError."""
    ca = np.array([1 + 1j, 2 + 2j, 3 - 1j, 4 + 0j])
    with pytest.raises(TypeError):
        fr.fft.rfft(ca)


def test_ihfft_complex_input_raises_typeerror():
    """numpy _pocketfft.py:700 — ihfft delegates to rfft (real-only); TypeError."""
    ca = np.array([1 + 1j, 2 + 2j, 3 - 1j, 4 + 0j])
    with pytest.raises(TypeError):
        fr.fft.ihfft(ca)
