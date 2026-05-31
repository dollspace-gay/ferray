"""Complex unary expansion — abs/absolute, negative, sinc, rint (#923/#927/#928).

numpy registers a complex loop for each of these four unary ops; the prior
ferray-python binding routed them to `reject_complex_unary` (TypeError) or had
no complex arm at all. This suite pins the numpy-COMPUTES contract:

  - ``abs``/``absolute`` (complex) -> REAL magnitude
    (``np.abs([3+4j]) == [5.0]``; dtype float64 for c128, float32 for c64).
    (upstream: numpy/_core/code_generators/generate_umath.py `absolute`
    complex loop -> `npy_cabs`, returns the real magnitude.)
  - ``negative`` (complex) -> ``-z`` (complex; dtype preserved).
  - ``sinc`` (complex) -> ``sin(pi*z)/(pi*z)`` with ``sinc(0) == 1+0j``.
    (upstream: numpy/lib/_function_base_impl.py `def sinc`.)
  - ``rint`` (complex) -> round real AND imag independently, ties to even
    (``np.rint([2.5+3.5j]) == [2+4j]``).
    (upstream: numpy/_core/code_generators/generate_umath.py `rint`
    `TD('fdg' + cmplx)`.)

Every expected value is taken from a LIVE numpy call (R-CHAR-3) — never copied
from the ferray side. The suite also asserts the REAL path is unregressed.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# abs / absolute (complex) -> REAL magnitude
# ---------------------------------------------------------------------------


def test_abs_complex128_returns_real_magnitude_matches_numpy():
    z = np.array([3 + 4j, 1 + 2j, -5 + 12j], dtype="complex128")
    expected = np.abs(z)  # live oracle -> float64 magnitude
    result = np.asarray(fr.abs(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.float64, (
        f"abs c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_abs_complex64_returns_float32_magnitude_matches_numpy():
    z = np.array([3 + 4j, 0 + 1j], dtype="complex64")
    expected = np.abs(z)  # live oracle -> float32 magnitude
    result = np.asarray(fr.abs(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.float32, (
        f"abs c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0)


def test_absolute_complex128_matches_numpy():
    z = np.array([1 + 2j], dtype="complex128")
    expected = np.absolute(z)
    result = np.asarray(fr.absolute(fr.asarray(z)))
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# negative (complex) -> -z
# ---------------------------------------------------------------------------


def test_negative_complex128_matches_numpy():
    z = np.array([1 + 2j, 3 - 1j, -4 - 5j], dtype="complex128")
    expected = np.negative(z)  # live oracle: -z
    result = np.asarray(fr.negative(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex128, (
        f"negative c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_array_equal(result, expected)


def test_negative_complex64_preserves_width_matches_numpy():
    z = np.array([1 + 2j, 3 - 1j], dtype="complex64")
    expected = np.negative(z)
    result = np.asarray(fr.negative(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex64, (
        f"negative c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# sinc (complex) -> sin(pi*z)/(pi*z), sinc(0) == 1+0j
# ---------------------------------------------------------------------------


def test_sinc_complex128_matches_numpy():
    z = np.array([1 + 2j, 0.5 + 0j, 3 + 1j], dtype="complex128")
    expected = np.sinc(z)  # live oracle
    result = np.asarray(fr.sinc(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex128, (
        f"sinc c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_sinc_complex_at_zero_is_one_matches_numpy():
    z = np.array([0j], dtype="complex128")
    expected = np.sinc(z)  # live oracle -> [1+0j]
    result = np.asarray(fr.sinc(fr.asarray(z)))
    np.testing.assert_array_equal(result, expected)
    assert result[0] == (1 + 0j)


def test_sinc_complex64_preserves_width_matches_numpy():
    z = np.array([1 + 2j], dtype="complex64")
    expected = np.sinc(z)
    result = np.asarray(fr.sinc(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex64, (
        f"sinc c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=0)


# ---------------------------------------------------------------------------
# rint (complex) -> per-component round-half-to-even
# ---------------------------------------------------------------------------


def test_rint_complex128_matches_numpy():
    z = np.array([1.4 + 2.6j, -1.4 - 2.6j, 0.5 + 1.5j], dtype="complex128")
    expected = np.rint(z)  # live oracle
    result = np.asarray(fr.rint(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex128, (
        f"rint c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_array_equal(result, expected)


def test_rint_complex_half_to_even_matches_numpy():
    # Ties: 2.5 -> 2, 3.5 -> 4, -2.5 -> -2, -3.5 -> -4 (round-half-to-even).
    z = np.array([2.5 + 3.5j, -2.5 - 3.5j], dtype="complex128")
    expected = np.rint(z)  # live oracle
    result = np.asarray(fr.rint(fr.asarray(z)))
    np.testing.assert_array_equal(result, expected)


def test_rint_complex64_preserves_width_matches_numpy():
    z = np.array([1.4 + 2.6j], dtype="complex64")
    expected = np.rint(z)
    result = np.asarray(fr.rint(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex64, (
        f"rint c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# REAL path unregressed (the complex arms must not touch real dispatch)
# ---------------------------------------------------------------------------


def test_abs_real_unchanged_matches_numpy():
    x = np.array([-3.0, 4.0, -5.0])
    expected = np.abs(x)
    result = np.asarray(fr.abs(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_abs_int_unchanged_matches_numpy():
    x = np.array([-3, 4, -5], dtype="int32")
    expected = np.abs(x)  # int32 preserved
    result = np.asarray(fr.abs(fr.asarray(x)))
    assert result.dtype == expected.dtype == np.int32
    np.testing.assert_array_equal(result, expected)


def test_negative_real_unchanged_matches_numpy():
    x = np.array([1.0, -2.0])
    expected = np.negative(x)
    result = np.asarray(fr.negative(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_rint_real_unchanged_matches_numpy():
    x = np.array([1.5, 2.5, -0.5])
    expected = np.rint(x)  # half-to-even
    result = np.asarray(fr.rint(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_sinc_real_unchanged_matches_numpy():
    x = np.array([0.0, 0.5, 1.0])
    expected = np.sinc(x)
    result = np.asarray(fr.sinc(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Genuinely-real-only ops still RAISE on complex (no over-correction)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cbrt", "i0", "degrees", "radians",
                                  "deg2rad", "rad2deg", "fabs", "spacing"])
def test_real_only_unary_still_raises_on_complex(name):
    z = fr.array([1 + 2j])
    with pytest.raises(TypeError):
        getattr(fr, name)(z)
