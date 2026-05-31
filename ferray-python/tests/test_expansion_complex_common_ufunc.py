"""Complex common-ufunc expansion — square / reciprocal / sign / isnan /
isinf / isfinite (#934).

numpy registers a complex loop for each of these six common ufuncs; the prior
ferray-python binding rejected complex input (``square``/``reciprocal`` funnelled
a complex dtype into their integer fallback, ``sign`` through the real
float/int split, the predicates through the real ``match_dtype_float!`` path —
all raising ``TypeError``). This suite pins the numpy-COMPUTES contract:

  - ``square`` (complex) -> ``z*z`` (complex; width preserved).
    ``np.square([1+2j]) == [-3+4j]``.
  - ``reciprocal`` (complex) -> ``1/z`` (complex; width preserved);
    ``np.reciprocal([0j]) == [nan+nanj]`` (no panic, RuntimeWarning).
  - ``sign`` (complex) -> unit phasor ``z/|z|`` for ``z != 0``, ``0+0j`` for
    ``z == 0`` (``np.sign([3+4j]) == [0.6+0.8j]``, ``abs(sign) == 1``).
  - ``isnan`` (complex) -> bool, True if NaN in EITHER part.
  - ``isinf`` (complex) -> bool, True if ±Inf in EITHER part.
  - ``isfinite`` (complex) -> bool, True iff BOTH parts finite.

Every expected value is taken from a LIVE numpy call (R-CHAR-3) — never copied
from the ferray side. The suite also asserts the REAL path is unregressed.
"""

import numpy as np
import pytest

import ferray as fr


# ---------------------------------------------------------------------------
# square (complex) -> z*z
# ---------------------------------------------------------------------------


def test_square_complex128_matches_numpy():
    z = np.array([1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j], dtype="complex128")
    expected = np.square(z)  # live oracle: [-3+4j, -7-24j, 0, 3-4j]
    result = np.asarray(fr.square(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex128, (
        f"square c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_square_complex64_preserves_width():
    z = np.array([1 + 2j, -3 + 4j], dtype="complex64")
    expected = np.square(z)  # live oracle, complex64
    result = np.asarray(fr.square(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex64, (
        f"square c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# reciprocal (complex) -> 1/z, with 1/0 -> nan+nanj
# ---------------------------------------------------------------------------


def test_reciprocal_complex128_matches_numpy():
    z = np.array([1 + 2j, -3 + 4j, 2 - 1j, 1 + 1j], dtype="complex128")
    expected = np.reciprocal(z)  # live oracle
    result = np.asarray(fr.reciprocal(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex128, (
        f"reciprocal c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_reciprocal_complex64_preserves_width():
    z = np.array([1 + 2j, 2 - 1j], dtype="complex64")
    expected = np.reciprocal(z)  # live oracle, complex64
    result = np.asarray(fr.reciprocal(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex64, (
        f"reciprocal c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0)


def test_reciprocal_complex_zero_is_nan_nanj():
    # numpy: 1/(0+0j) == nan+nanj (RuntimeWarning), NOT a panic / inf.
    z = np.array([0 + 0j, 1 + 0j], dtype="complex128")
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.reciprocal(z)  # live oracle: [nan+nanj, 1+0j]
    result = np.asarray(fr.reciprocal(fr.asarray(z)))
    assert np.isnan(result[0].real) and np.isnan(result[0].imag), (
        f"reciprocal(0j): numpy {expected[0]!r}, ferray {result[0]!r}"
    )
    np.testing.assert_allclose(result[1], expected[1], rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# sign (complex) -> z/|z| (0+0j for z==0)
# ---------------------------------------------------------------------------


def test_sign_complex128_is_unit_phasor_matches_numpy():
    z = np.array([3 + 4j, 1 + 2j, -3 + 4j, 2 - 1j], dtype="complex128")
    expected = np.sign(z)  # live oracle: z/|z| -> [0.6+0.8j, ...]
    result = np.asarray(fr.sign(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex128, (
        f"sign c128 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    # unit modulus (live numpy: abs(sign(z)) == 1 for z != 0)
    np.testing.assert_allclose(np.abs(result), np.abs(expected), rtol=1e-12)


def test_sign_complex_zero_is_zero():
    # numpy: sign(0+0j) == 0+0j (NOT nan from z/|z| with |z|==0).
    z = np.array([0 + 0j, 3 + 4j], dtype="complex128")
    expected = np.sign(z)  # live oracle: [0+0j, 0.6+0.8j]
    result = np.asarray(fr.sign(fr.asarray(z)))
    assert result[0] == expected[0] == 0 + 0j, (
        f"sign(0j): numpy {expected[0]!r}, ferray {result[0]!r}"
    )
    np.testing.assert_allclose(result[1], expected[1], rtol=1e-12, atol=0)


def test_sign_complex64_preserves_width():
    z = np.array([3 + 4j, 0 + 0j], dtype="complex64")
    expected = np.sign(z)  # live oracle, complex64
    result = np.asarray(fr.sign(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.complex64, (
        f"sign c64 dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=0)


# ---------------------------------------------------------------------------
# isnan / isinf / isfinite (complex) -> bool (per-part rule)
# ---------------------------------------------------------------------------


def test_isnan_complex_either_part_matches_numpy():
    z = np.array(
        [1 + 2j, complex(np.nan, 0), complex(0, np.nan), complex(np.nan, np.nan)],
        dtype="complex128",
    )
    expected = np.isnan(z)  # live oracle: [F, T, T, T]
    result = np.asarray(fr.isnan(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.bool_, (
        f"isnan dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.array_equal(result, expected), (
        f"isnan: numpy {expected!r}, ferray {result!r}"
    )


def test_isinf_complex_either_part_matches_numpy():
    z = np.array(
        [1 + 2j, complex(np.inf, 0), complex(0, np.inf), complex(-np.inf, 5)],
        dtype="complex128",
    )
    expected = np.isinf(z)  # live oracle: [F, T, T, T]
    result = np.asarray(fr.isinf(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.bool_, (
        f"isinf dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.array_equal(result, expected), (
        f"isinf: numpy {expected!r}, ferray {result!r}"
    )


def test_isfinite_complex_both_parts_matches_numpy():
    z = np.array(
        [1 + 2j, complex(np.inf, 0), complex(0, np.nan), complex(3, 4)],
        dtype="complex128",
    )
    expected = np.isfinite(z)  # live oracle: [T, F, F, T]
    result = np.asarray(fr.isfinite(fr.asarray(z)))
    assert result.dtype == expected.dtype == np.bool_, (
        f"isfinite dtype: numpy {expected.dtype}, ferray {result.dtype}"
    )
    assert np.array_equal(result, expected), (
        f"isfinite: numpy {expected!r}, ferray {result!r}"
    )


def test_predicates_complex64_match_numpy():
    z = np.array(
        [1 + 2j, complex(np.nan, 0), complex(0, np.inf)], dtype="complex64"
    )
    for name, frf, npf in [
        ("isnan", fr.isnan, np.isnan),
        ("isinf", fr.isinf, np.isinf),
        ("isfinite", fr.isfinite, np.isfinite),
    ]:
        expected = npf(z)  # live oracle
        result = np.asarray(frf(fr.asarray(z)))
        assert np.array_equal(result, expected), (
            f"{name} c64: numpy {expected!r}, ferray {result!r}"
        )


# ---------------------------------------------------------------------------
# REAL path unregressed (byte-identical to numpy) — guards against the
# complex arm leaking into the real dispatch.
# ---------------------------------------------------------------------------


def test_real_square_unregressed():
    x = np.array([1.0, -2.0, 3.5], dtype="float64")
    expected = np.square(x)  # live oracle
    result = np.asarray(fr.square(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_real_int_square_unregressed():
    x = np.array([2, -3, 4], dtype="int64")
    expected = np.square(x)  # live oracle: int64 [4, 9, 16]
    result = np.asarray(fr.square(fr.asarray(x)))
    assert result.dtype == expected.dtype == np.int64
    np.testing.assert_array_equal(result, expected)


def test_real_sign_unregressed():
    x = np.array([-2.0, 0.0, 5.0], dtype="float64")
    expected = np.sign(x)  # live oracle: [-1, 0, 1]
    result = np.asarray(fr.sign(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_real_isnan_unregressed():
    x = np.array([1.0, np.nan, np.inf, -np.inf], dtype="float64")
    expected = np.isnan(x)  # live oracle: [F, T, F, F]
    result = np.asarray(fr.isnan(fr.asarray(x)))
    assert result.dtype == expected.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)


def test_real_reciprocal_unregressed():
    x = np.array([2.0, 4.0, -0.5], dtype="float64")
    expected = np.reciprocal(x)  # live oracle: [0.5, 0.25, -2.0]
    result = np.asarray(fr.reciprocal(fr.asarray(x)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
