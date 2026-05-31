"""Adversarial divergence tests for ferray.fft vs the live numpy 2.4.5 oracle.

Each test pins a CONFIRMED divergence between ``import ferray as fr`` and the
live ``numpy`` oracle. Values compared with ``np.allclose`` (complex-safe);
shape/dtype/exception-type compared exactly. Every test cites the numpy
upstream ``file:line`` it encodes; expected values come from a live numpy call
(R-CHAR-3 — never literal-copied from the ferray side).

Authored by acto-critic (re-audit from scratch). These are EXPECTED TO FAIL
against the current ferray build — they document where ferray diverges from
numpy. Run:

    cd ferray-python && PYTHONPATH=python python3 -m pytest tests/divergence_fft.py -q
"""

import numpy as np
import pytest

import ferray as fr


# ===========================================================================
# 1. fftshift / ifftshift must accept a SCALAR int ``axes`` argument.
#
# numpy/fft/_helper.py:69  (fftshift)
#     elif isinstance(axes, integer_types):
#         shift = x.shape[axes] // 2
# numpy/fft/_helper.py:117 (ifftshift)
#     elif isinstance(axes, integer_types):
#         shift = -(x.shape[axes] // 2)
#
# numpy explicitly accepts a scalar int for ``axes`` (shift one axis).
# ferray's signature binds ``axes: Option<Vec<isize>>`` so a bare Python int is
# rejected with TypeError ("'int' object is not an instance of 'Sequence'") at
# the PyO3 boundary before any shift runs.
# ===========================================================================


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


# ===========================================================================
# 2. fft / ifft with negative n must raise ValueError, not OverflowError.
#
# numpy/fft/_pocketfft.py:59
#     if n < 1:
#         raise ValueError(f"Invalid number of FFT data points ({n}) specified.")
#
# ferray binds ``n: Option<usize>``; a negative int triggers a PyO3
# OverflowError ("can't convert negative int to unsigned") at the boundary
# before the ferray length check runs. numpy raises ValueError.
# ===========================================================================


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


# ===========================================================================
# 3. fftfreq / rfftfreq with negative n must raise ValueError.
#
# numpy/fft/_helper.py:171  (fftfreq)
#     results = empty(n, int, device=device)
# numpy/fft/_helper.py:234  (rfftfreq, via arange(0, N, ...))
#
# numpy accepts a negative Python int for ``n`` and raises ValueError from the
# downstream ``empty(n, ...)`` ("negative dimensions are not allowed"). ferray
# binds ``n: usize`` so a negative int is an OverflowError at the PyO3 boundary.
# ===========================================================================


def test_fftfreq_negative_n_raises_valueerror():
    """numpy _helper.py:171 — fftfreq(-2) raises ValueError, not OverflowError."""
    with pytest.raises(ValueError):
        fr.fft.fftfreq(-2)


def test_rfftfreq_negative_n_raises_valueerror():
    """numpy _helper.py:234 — rfftfreq(-2) raises ValueError, not OverflowError."""
    with pytest.raises(ValueError):
        fr.fft.rfftfreq(-2)


# ===========================================================================
# 4. fftn / ifftn with s containing -1 means "use the whole input" (numpy 2.0).
#
# numpy/fft/_pocketfft.py:736-737  (_cook_nd_args)
#     # use the whole input array along axis `i` if `s[i] == -1`
#     s = [a.shape[_a] if _s == -1 else _s for _s, _a in zip(s, axes)]
#
# numpy 2.0 added s[i] == -1 as a sentinel. ferray binds ``s: Option<Vec<usize>>``
# so a -1 entry is an OverflowError at the PyO3 boundary instead of being honored.
# ===========================================================================


def test_fftn_s_minus_one_uses_whole_input():
    """numpy _pocketfft.py:737 — s[i] == -1 means use the whole input axis."""
    m = np.arange(12.0).reshape(3, 4)
    expected = np.fft.fftn(m, s=[-1, -1], axes=[0, 1])
    got = np.asarray(fr.fft.fftn(m, s=[-1, -1], axes=[0, 1]))
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=1e-10, atol=1e-10)


# ===========================================================================
# 5. rfft / ihfft on COMPLEX input must raise TypeError.
#
# numpy/fft/_pocketfft.py:81 — rfft dispatches to pfu.rfft_n_even / rfft_n_odd,
# real-only ufuncs with no complex loop:
#     >>> np.fft.rfft(np.array([1+1j, 2+0j]))
#     TypeError: ufunc 'rfft_n_even' not supported for the input types ...
# ihfft (numpy/fft/_pocketfft.py ihfft) conjugates then delegates to the same
# real-input path, so it also raises TypeError on complex input.
#
# ferray instead SILENTLY casts complex -> float (discarding the imaginary
# part) via coerce_dtype, producing a result where numpy refuses. This is a
# silent lossy round-trip across the boundary (violates R-CODE-4 / R-DEV-2).
# ===========================================================================


def test_rfft_complex_input_raises_typeerror():
    """numpy _pocketfft.py:81 — rfft has no complex ufunc loop; raises TypeError."""
    ca = np.array([1 + 1j, 2 + 2j, 3 - 1j, 4 + 0j])
    with pytest.raises(TypeError):
        fr.fft.rfft(ca)


def test_ihfft_complex_input_raises_typeerror():
    """numpy ihfft delegates to the real-input rfft path; complex -> TypeError."""
    ca = np.array([1 + 1j, 2 + 2j, 3 - 1j, 4 + 0j])
    with pytest.raises(TypeError):
        fr.fft.ihfft(ca)


def test_rfft2_complex_input_raises_typeerror():
    """numpy rfft2 -> rfftn -> rfft real-only path; complex -> TypeError."""
    ca = np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]])
    with pytest.raises(TypeError):
        fr.fft.rfft2(ca)


def test_rfftn_complex_input_raises_typeerror():
    """numpy rfftn -> rfft real-only path; complex -> TypeError."""
    ca = np.array([[1 + 1j, 2 + 0j], [3 - 1j, 4 + 2j]])
    with pytest.raises(TypeError):
        fr.fft.rfftn(ca)


# ===========================================================================
# 6. Axis out of bounds must raise IndexError, not ValueError.  [NEW]
#
# numpy/fft/_pocketfft.py:88
#     axis = normalize_axis_index(axis, a.ndim)
# normalize_axis_index raises a plain ``IndexError`` (the numpy.AxisError type
# is NOT used for fft's axis bound — confirmed live: ``type(e) is IndexError``).
# ferray maps an out-of-range axis to PyValueError at the boundary.
# ===========================================================================


def test_fft_axis_out_of_bounds_raises_indexerror():
    """numpy _pocketfft.py:88 — out-of-range axis raises IndexError."""
    m = np.arange(12.0).reshape(3, 4)
    # Confirm the oracle's exact type (IndexError, not ValueError/AxisError).
    with pytest.raises(IndexError):
        np.fft.fft(m, axis=5)
    with pytest.raises(IndexError):
        fr.fft.fft(m, axis=5)


# ===========================================================================
# 7. FFT of a 0-d / scalar input must raise IndexError.  [NEW]
#
# numpy/fft/_pocketfft.py:88 — normalize_axis_index(-1, 0) on a 0-d array has
# no axis -1, so numpy raises IndexError. ferray raises ValueError.
# ===========================================================================


def test_fft_scalar_input_raises_indexerror():
    """numpy _pocketfft.py:88 — fft of a 0-d array raises IndexError."""
    s = np.float64(3.0)
    with pytest.raises(IndexError):
        np.fft.fft(s)
    with pytest.raises(IndexError):
        fr.fft.fft(s)


# ===========================================================================
# 8. fftfreq with d=0 must raise ZeroDivisionError, not ValueError.  [NEW]
#
# numpy/fft/_helper.py:170
#     val = 1.0 / (n * d)
# With d=0 this divides by zero on a Python float -> ZeroDivisionError.
# ferray pre-checks d and raises ValueError instead.
# ===========================================================================


def test_fftfreq_d_zero_raises_zerodivisionerror():
    """numpy _helper.py:170 — fftfreq(n, d=0) raises ZeroDivisionError."""
    with pytest.raises(ZeroDivisionError):
        np.fft.fftfreq(8, d=0.0)
    with pytest.raises(ZeroDivisionError):
        fr.fft.fftfreq(8, d=0.0)


# ===========================================================================
# 9. The ``out=`` kwarg (numpy 2.0) is part of the public fft ABI.  [NEW]
#
# numpy/fft/_pocketfft.py — fft(a, n=None, axis=-1, norm=None, out=None);
# the ``out`` parameter was added in numpy 2.0 and writes the result in place,
# returning the same object:
#     >>> out = np.empty(4, np.complex128); np.fft.fft(a, out=out) is out -> True
# ferray's signature omits ``out`` entirely, so passing it raises
# TypeError ("unexpected keyword argument 'out'"). R-DEV-2: kwarg surface must
# match the numpy public signature.
# ===========================================================================


def test_fft_accepts_out_kwarg():
    """numpy fft signature carries out=None (added 2.0); result writes into out."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.fft.fft(a)
    out = np.empty(4, dtype=np.complex128)
    ret = fr.fft.fft(a, out=out)
    got = np.asarray(ret)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-10)


# ===========================================================================
# 10. The ``out=`` kwarg is part of the PUBLIC ABI of EVERY fft transform,
#     not just fft/ifft.  [NEW — acto-critic re-audit, tracking #1016]
#
# numpy/fft/_pocketfft.py — every transform signature carries out=None:
#     rfft   (a, n=None, axis=-1, norm=None, out=None)   :325
#     irfft  (a, n=None, axis=-1, norm=None, out=None)   :422
#     hfft   (a, n=None, axis=-1, norm=None, out=None)   :530
#     ihfft  (a, n=None, axis=-1, norm=None, out=None)   :633
#     fftn   (a, s=None, axes=None, norm=None, out=None) :756
#     ifftn  (a, s=None, axes=None, norm=None, out=None) :888
#     fft2   (a, s=None, axes=(-2,-1), norm=None, out=None)   :1020
#     ifft2  (a, s=None, axes=(-2,-1), norm=None, out=None)   :1145
#     rfftn  (a, s=None, axes=None, norm=None, out=None) :1267
#     rfft2  (a, s=None, axes=(-2,-1), norm=None, out=None)   :1394
#     irfftn (a, s=None, axes=None, norm=None, out=None) :1474
#     irfft2 (a, s=None, axes=(-2,-1), norm=None, out=None)   :1613
#
# ferray-python/src/fft.rs binds out= ONLY on fft (line 187) and ifft (line
# 222). The 12 nd/real/hermitian #[pyfunction]s omit it, so passing out=
# raises TypeError ("got an unexpected keyword argument 'out'") at the PyO3
# boundary, where numpy accepts it. Expected (RED) result values come from a
# live numpy call (R-CHAR-3), compared with the no-out plain call.
# ===========================================================================


def _vec_real():
    return np.array([1.0, 2.0, 3.0, 4.0])


def _vec_cplx():
    return np.array([1 + 1j, 2 + 0j, 3 - 1j, 4 + 2j])


def _mat_real():
    return np.arange(12.0).reshape(3, 4)


def test_rfft_accepts_out_kwarg():
    """numpy _pocketfft.py:325 — rfft(a, out=...) writes into & returns out."""
    a = _vec_real()
    expected = np.fft.rfft(a)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.rfft(a, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_irfft_accepts_out_kwarg():
    """numpy _pocketfft.py:422 — irfft(a, out=...) writes into & returns out."""
    ca = _vec_cplx()
    expected = np.fft.irfft(ca)  # live oracle
    out = np.empty(expected.shape, dtype=np.float64)
    got = np.asarray(fr.fft.irfft(ca, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_hfft_accepts_out_kwarg():
    """numpy _pocketfft.py:530 — hfft(a, out=...) writes into & returns out."""
    ca = _vec_cplx()
    expected = np.fft.hfft(ca)  # live oracle
    out = np.empty(expected.shape, dtype=np.float64)
    got = np.asarray(fr.fft.hfft(ca, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ihfft_accepts_out_kwarg():
    """numpy _pocketfft.py:633 — ihfft(a, out=...) writes into & returns out."""
    a = _vec_real()
    expected = np.fft.ihfft(a)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.ihfft(a, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_fftn_accepts_out_kwarg():
    """numpy _pocketfft.py:756 — fftn(a, out=...) writes into & returns out."""
    a = _vec_real()
    expected = np.fft.fftn(a)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.fftn(a, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ifftn_accepts_out_kwarg():
    """numpy _pocketfft.py:888 — ifftn(a, out=...) writes into & returns out."""
    a = _vec_real()
    expected = np.fft.ifftn(a)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.ifftn(a, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_fft2_accepts_out_kwarg():
    """numpy _pocketfft.py:1020 — fft2(a, out=...) writes into & returns out."""
    m = _mat_real()
    expected = np.fft.fft2(m)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.fft2(m, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_ifft2_accepts_out_kwarg():
    """numpy _pocketfft.py:1145 — ifft2(a, out=...) writes into & returns out."""
    m = _mat_real()
    expected = np.fft.ifft2(m)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.ifft2(m, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_rfft2_accepts_out_kwarg():
    """numpy _pocketfft.py:1394 — rfft2(a, out=...) writes into & returns out."""
    m = _mat_real()
    expected = np.fft.rfft2(m)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.rfft2(m, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_irfft2_accepts_out_kwarg():
    """numpy _pocketfft.py:1613 — irfft2(a, out=...) writes into & returns out."""
    cm = _mat_real() + 0j
    expected = np.fft.irfft2(cm)  # live oracle
    out = np.empty(expected.shape, dtype=np.float64)
    got = np.asarray(fr.fft.irfft2(cm, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_rfftn_accepts_out_kwarg():
    """numpy _pocketfft.py:1267 — rfftn(a, out=...) writes into & returns out."""
    m = _mat_real()
    expected = np.fft.rfftn(m)  # live oracle
    out = np.empty(expected.shape, dtype=np.complex128)
    got = np.asarray(fr.fft.rfftn(m, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_irfftn_accepts_out_kwarg():
    """numpy _pocketfft.py:1474 — irfftn(a, out=...) writes into & returns out."""
    cm = _mat_real() + 0j
    expected = np.fft.irfftn(cm)  # live oracle
    out = np.empty(expected.shape, dtype=np.float64)
    got = np.asarray(fr.fft.irfftn(cm, out=out))
    np.testing.assert_allclose(got, expected, atol=1e-10)
