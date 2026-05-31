"""Phase-3 parity tests for ferray.fft."""

import numpy as np
import pytest

import ferray


# ---------------------------------------------------------------------------
# Complex FFT
# ---------------------------------------------------------------------------


def test_fft_round_trip_with_ifft():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    np.testing.assert_allclose(ferray.fft.ifft(ferray.fft.fft(a)), a, atol=1e-10)


def test_fft_matches_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(ferray.fft.fft(a), np.fft.fft(a), atol=1e-10)


def test_fft_real_input_promotes_to_complex():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    out = ferray.fft.fft(a)
    assert out.dtype == np.complex128


def test_fft_int_input_promotes():
    a = np.array([1, 2, 3, 4])
    out = ferray.fft.fft(a)
    assert out.dtype == np.complex128
    np.testing.assert_allclose(out, np.fft.fft(a))


def test_fft_complex64_preserved():
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.complex64)
    out = ferray.fft.fft(a)
    assert out.dtype == np.complex64


def test_fftn_2d_matches_numpy():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 8))
    np.testing.assert_allclose(ferray.fft.fftn(a), np.fft.fftn(a), atol=1e-10)


def test_ifftn_round_trips():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    np.testing.assert_allclose(ferray.fft.ifftn(ferray.fft.fftn(a)), a, atol=1e-10)


# ---------------------------------------------------------------------------
# fft2 / ifft2 (added during phase 3 audit)
# ---------------------------------------------------------------------------


def test_fft2_matches_numpy():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 8))
    np.testing.assert_allclose(ferray.fft.fft2(a), np.fft.fft2(a), atol=1e-10)


def test_ifft2_round_trips():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    np.testing.assert_allclose(ferray.fft.ifft2(ferray.fft.fft2(a)), a, atol=1e-10)


# ---------------------------------------------------------------------------
# Real FFT
# ---------------------------------------------------------------------------


def test_rfft_matches_numpy():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    np.testing.assert_allclose(ferray.fft.rfft(a), np.fft.rfft(a), atol=1e-10)


def test_irfft_round_trips():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(16)
    np.testing.assert_allclose(
        ferray.fft.irfft(ferray.fft.rfft(a), n=16), a, atol=1e-10
    )


def test_rfft2_matches_numpy():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 8))
    np.testing.assert_allclose(ferray.fft.rfft2(a), np.fft.rfft2(a), atol=1e-10)


def test_irfft2_round_trips():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 8))
    np.testing.assert_allclose(
        ferray.fft.irfft2(ferray.fft.rfft2(a), s=(4, 8)), a, atol=1e-10
    )


def test_rfftn_3d_matches_numpy():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4, 6))
    np.testing.assert_allclose(ferray.fft.rfftn(a), np.fft.rfftn(a), atol=1e-10)


def test_irfftn_round_trips():
    rng = np.random.default_rng(42)
    a = rng.standard_normal((4, 4, 6))
    np.testing.assert_allclose(
        ferray.fft.irfftn(ferray.fft.rfftn(a), s=(4, 4, 6)), a, atol=1e-10
    )


# ---------------------------------------------------------------------------
# Frequency / shift
# ---------------------------------------------------------------------------


def test_fftfreq_matches_numpy():
    np.testing.assert_allclose(ferray.fft.fftfreq(8), np.fft.fftfreq(8))


def test_fftfreq_with_d():
    np.testing.assert_allclose(
        ferray.fft.fftfreq(8, d=0.1), np.fft.fftfreq(8, d=0.1)
    )


def test_rfftfreq_matches_numpy():
    np.testing.assert_allclose(ferray.fft.rfftfreq(8), np.fft.rfftfreq(8))


def test_fftshift_1d():
    a = np.arange(8)
    np.testing.assert_array_equal(ferray.fft.fftshift(a), np.fft.fftshift(a))


def test_ifftshift_round_trips_with_fftshift():
    a = np.arange(8)
    np.testing.assert_array_equal(
        ferray.fft.ifftshift(ferray.fft.fftshift(a)), a
    )


# ---------------------------------------------------------------------------
# Norm parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("norm", ["backward", "forward", "ortho"])
def test_fft_norm_round_trip(norm):
    a = np.array([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
    out = ferray.fft.fft(a, norm=norm)
    expected = np.fft.fft(a, norm=norm)
    np.testing.assert_allclose(out, expected, atol=1e-10)
