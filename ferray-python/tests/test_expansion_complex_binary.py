"""Complex BINARY expansion — arithmetic, comparison, and raise guards (#924).

numpy registers a complex loop for the binary ops below; the prior
ferray-python binding funnelled every binary op through `match_dtype_numeric!`
(real-only) and so rejected complex inputs with `TypeError`, even though the
ferray-ufunc library ops already accept complex (`add_broadcast`/… `T:
WrappingArith`/`TrueDivide`, `equal_broadcast` `T: PartialEq`, the array-array
`power_complex`; #869). This suite pins the numpy contract:

  - ``add``/``subtract``/``multiply``/``divide``/``power`` (complex) -> COMPUTE
    the complex result (numpy: ``np.add([1+2j],[3+4j]) == [4+6j]``; division is
    true complex division; power is the principal complex power).
  - ``equal``/``not_equal`` (complex) -> bool, elementwise.
  - ``less``/``greater``/``less_equal``/``greater_equal`` (complex) -> COMPUTE
    LEXICOGRAPHICALLY ``(real, then imag)`` — numpy does NOT raise
    (``np.less([1+2j,3+2j],[1+5j,3+1j]) == [True False]``, verified live).
    (upstream contract mirrors ferray-ma #874's ``cmp_complex_arm!``.)
  - ``floor_divide``/``remainder``/``mod``/``bitwise_and``/``bitwise_or``/
    ``bitwise_xor``/``left_shift``/``right_shift`` (complex) -> RAISE
    ``TypeError`` (numpy registers NO complex loop for these).

Promotion: ``complex+real``, ``complex+python-scalar``, and reflected operands
all promote via numpy's own ``result_type`` (complex+real -> complex). c64 keeps
c64 width; c128 keeps c128.

Every expected value is taken from a LIVE numpy call (R-CHAR-3) — never copied
from the ferray side. The suite also asserts the REAL path is unregressed.
"""

import numpy as np
import pytest

import ferray as fr


def _fr(z):
    """Round-trip a numpy array through ferray's complex coercion."""
    return fr.asarray(z)


# ---------------------------------------------------------------------------
# Arithmetic: add / subtract / multiply / divide / power (complex compute)
# ---------------------------------------------------------------------------

C128_A = np.array([1 + 2j, 3 - 1j, -2 + 0.5j, 0 + 0j], dtype="complex128")
C128_B = np.array([3 + 4j, 1 + 1j, 2 - 3j, 1 + 0j], dtype="complex128")


@pytest.mark.parametrize(
    "frfn,npfn",
    [
        (fr.add, np.add),
        (fr.subtract, np.subtract),
        (fr.multiply, np.multiply),
        (fr.divide, np.divide),
    ],
)
def test_arith_complex128_matches_numpy(frfn, npfn):
    expected = npfn(C128_A, C128_B)  # live oracle
    result = np.asarray(frfn(_fr(C128_A), _fr(C128_B)))
    assert result.dtype == expected.dtype == np.complex128
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "frfn,npfn",
    [
        (fr.add, np.add),
        (fr.subtract, np.subtract),
        (fr.multiply, np.multiply),
        (fr.divide, np.divide),
    ],
)
def test_arith_complex64_preserves_width_matches_numpy(frfn, npfn):
    a = C128_A.astype("complex64")
    b = C128_B.astype("complex64")
    expected = npfn(a, b)  # live oracle -> complex64
    result = np.asarray(frfn(_fr(a), _fr(b)))
    assert result.dtype == expected.dtype == np.complex64
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=0)


def test_power_complex128_integer_exponent_matches_numpy():
    base = np.array([1 + 2j, 2 + 0j, 0 + 1j], dtype="complex128")
    exp = np.array([2 + 0j, 3 + 0j, 2 + 0j], dtype="complex128")
    expected = np.power(base, exp)  # live oracle: [-3+4j, 8+0j, -1+0j]
    result = np.asarray(fr.power(_fr(base), _fr(exp)))
    assert result.dtype == expected.dtype == np.complex128
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_power_complex128_general_exponent_matches_numpy():
    base = np.array([1 + 2j, 3 - 1j], dtype="complex128")
    exp = np.array([0.5 + 0.5j, 1 + 1j], dtype="complex128")
    expected = np.power(base, exp)  # live oracle (general exp(w*log z))
    result = np.asarray(fr.power(_fr(base), _fr(exp)))
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


def test_divide_by_zero_complex_matches_numpy():
    a = np.array([1 + 2j], dtype="complex128")
    b = np.array([0 + 0j], dtype="complex128")
    expected = np.divide(a, b)  # live: complex inf/nan, no panic
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.asarray(fr.divide(_fr(a), _fr(b)))
    # nan/inf positions must match numpy exactly.
    assert np.isnan(result.real) == np.isnan(expected.real)
    assert np.isinf(result.real) == np.isinf(expected.real)


# ---------------------------------------------------------------------------
# Promotion: complex + real, complex + python scalar, reflected
# ---------------------------------------------------------------------------


def test_add_complex_plus_real_array_promotes_to_complex():
    z = np.array([1 + 2j, 3 - 1j], dtype="complex128")
    r = np.array([1, 2], dtype="int64")
    expected = np.add(z, r)  # live -> complex128
    result = np.asarray(fr.add(_fr(z), fr.asarray(r)))
    assert result.dtype == expected.dtype == np.complex128
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_add_complex_plus_python_scalar_promotes():
    z = np.array([1 + 2j], dtype="complex128")
    expected = np.add(z, 1)  # live -> complex128 [2+2j]
    result = np.asarray(fr.add(_fr(z), 1))
    assert result.dtype == expected.dtype == np.complex128
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_add_python_scalar_plus_complex_reflected():
    z = np.array([1 + 2j], dtype="complex128")
    expected = np.add(1, z)  # live -> complex128
    result = np.asarray(fr.add(1, _fr(z)))
    assert result.dtype == expected.dtype == np.complex128
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)


def test_multiply_complex_plus_real_promote_matches_numpy():
    z = np.array([1 + 2j, 0 + 1j], dtype="complex128")
    r = np.array([2.0, 3.0], dtype="float64")
    expected = np.multiply(z, r)  # live -> complex128
    result = np.asarray(fr.multiply(_fr(z), fr.asarray(r)))
    assert result.dtype == expected.dtype == np.complex128
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# equal / not_equal (bool, elementwise)
# ---------------------------------------------------------------------------


def test_equal_complex_matches_numpy():
    a = np.array([1 + 2j, 3 - 1j, 0 + 0j], dtype="complex128")
    b = np.array([1 + 2j, 3 + 1j, 0 + 0j], dtype="complex128")
    expected = np.equal(a, b)  # live -> [True False True]
    result = np.asarray(fr.equal(_fr(a), _fr(b)))
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)


def test_not_equal_complex_matches_numpy():
    a = np.array([1 + 2j, 3 - 1j, 0 + 0j], dtype="complex128")
    b = np.array([1 + 2j, 3 + 1j, 0 + 0j], dtype="complex128")
    expected = np.not_equal(a, b)  # live -> [False True False]
    result = np.asarray(fr.not_equal(_fr(a), _fr(b)))
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)


def test_equal_complex64_matches_numpy():
    a = np.array([1 + 2j, 3 - 1j], dtype="complex64")
    b = np.array([1 + 2j, 9 + 9j], dtype="complex64")
    expected = np.equal(a, b)  # live
    result = np.asarray(fr.equal(_fr(a), _fr(b)))
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# less / greater / less_equal / greater_equal — LEXICOGRAPHIC (real, then imag)
# ---------------------------------------------------------------------------

LEX_A = np.array([1 + 2j, 3 + 2j, 1 + 2j, 2 + 0j], dtype="complex128")
LEX_B = np.array([1 + 5j, 3 + 1j, 1 + 2j, 1 + 9j], dtype="complex128")


@pytest.mark.parametrize(
    "frfn,npfn",
    [
        (fr.less, np.less),
        (fr.greater, np.greater),
        (fr.less_equal, np.less_equal),
        (fr.greater_equal, np.greater_equal),
    ],
)
def test_ordering_complex_lexicographic_matches_numpy(frfn, npfn):
    expected = npfn(LEX_A, LEX_B)  # live LEXICOGRAPHIC oracle
    result = np.asarray(frfn(_fr(LEX_A), _fr(LEX_B)))
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)


def test_less_complex_canonical_brief_example():
    a = np.array([1 + 2j, 3 + 2j], dtype="complex128")
    b = np.array([1 + 5j, 3 + 1j], dtype="complex128")
    expected = np.less(a, b)  # live -> [True False]
    result = np.asarray(fr.less(_fr(a), _fr(b)))
    np.testing.assert_array_equal(result, expected)
    # Independent assertion of the lexicographic contract value.
    assert list(result) == [True, False]


def test_ordering_complex_tie_real_part_compares_imag():
    # equal real parts -> the imag part decides (the lexicographic tie path).
    a = np.array([5 + 1j, 5 + 3j, 5 + 2j], dtype="complex128")
    b = np.array([5 + 2j, 5 + 2j, 5 + 2j], dtype="complex128")
    for frfn, npfn in [
        (fr.less, np.less),
        (fr.greater, np.greater),
        (fr.less_equal, np.less_equal),
        (fr.greater_equal, np.greater_equal),
    ]:
        expected = npfn(a, b)  # live
        result = np.asarray(frfn(_fr(a), _fr(b)))
        np.testing.assert_array_equal(result, expected)


def test_ordering_complex64_lexicographic_matches_numpy():
    a = LEX_A.astype("complex64")
    b = LEX_B.astype("complex64")
    expected = np.less(a, b)  # live
    result = np.asarray(fr.less(_fr(a), _fr(b)))
    np.testing.assert_array_equal(result, expected)


def test_ordering_complex_nan_part_matches_numpy():
    # A NaN part makes every ordering compare False (numpy's invalid->False).
    a = np.array([complex(np.nan, 1.0), 1 + 2j], dtype="complex128")
    b = np.array([1 + 1j, complex(np.nan, 0.0)], dtype="complex128")
    with np.errstate(invalid="ignore"):
        expected = np.less(a, b)  # live -> [False False]
        result = np.asarray(fr.less(_fr(a), _fr(b)))
    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Raise guards: floor_divide / remainder / mod / bitwise_* / shifts -> TypeError
# ---------------------------------------------------------------------------

Z = np.array([1 + 2j, 3 - 1j], dtype="complex128")


@pytest.mark.parametrize(
    "frfn,npfn",
    [
        (fr.floor_divide, np.floor_divide),
        (fr.remainder, np.remainder),
        (fr.mod, np.mod),
        (fr.bitwise_and, np.bitwise_and),
        (fr.bitwise_or, np.bitwise_or),
        (fr.bitwise_xor, np.bitwise_xor),
        (fr.left_shift, np.left_shift),
        (fr.right_shift, np.right_shift),
    ],
)
def test_unsupported_complex_binary_raises_typeerror_matching_numpy(frfn, npfn):
    # numpy raises TypeError for these on complex; ferray must match.
    with pytest.raises(TypeError):
        npfn(Z, Z)  # confirm numpy raises (live oracle for the exception type)
    with pytest.raises(TypeError):
        frfn(_fr(Z), _fr(Z))


def test_complex64_unsupported_binary_raises():
    z = Z.astype("complex64")
    with pytest.raises(TypeError):
        fr.floor_divide(_fr(z), _fr(z))


# ---------------------------------------------------------------------------
# Mask / nan in complex arithmetic
# ---------------------------------------------------------------------------


def test_arith_complex_with_nan_part_matches_numpy():
    a = np.array([complex(np.nan, 1.0), 1 + 2j], dtype="complex128")
    b = np.array([1 + 1j, complex(2.0, np.nan)], dtype="complex128")
    expected = np.add(a, b)  # live
    result = np.asarray(fr.add(_fr(a), _fr(b)))
    # NaN positions must coincide.
    np.testing.assert_array_equal(np.isnan(result.real), np.isnan(expected.real))
    np.testing.assert_array_equal(np.isnan(result.imag), np.isnan(expected.imag))


# ---------------------------------------------------------------------------
# REAL path UNREGRESSED — arithmetic / divide / compare byte-identical
# ---------------------------------------------------------------------------


def test_real_add_unregressed():
    a = np.array([1, 2, 3], dtype="int64")
    b = np.array([10, 20, 30], dtype="int64")
    expected = np.add(a, b)
    result = np.asarray(fr.add(fr.asarray(a), fr.asarray(b)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_real_divide_true_division_unregressed():
    a = np.array([1, 2, 3], dtype="int64")
    b = np.array([2, 2, 2], dtype="int64")
    expected = np.divide(a, b)  # int/int -> float64
    result = np.asarray(fr.divide(fr.asarray(a), fr.asarray(b)))
    assert result.dtype == expected.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


def test_real_less_unregressed():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([2.0, 2.0, 3.0])
    expected = np.less(a, b)
    result = np.asarray(fr.less(fr.asarray(a), fr.asarray(b)))
    np.testing.assert_array_equal(result, expected)


def test_real_floor_divide_unregressed():
    a = np.array([7, 8, 9], dtype="int64")
    b = np.array([2, 3, 4], dtype="int64")
    expected = np.floor_divide(a, b)
    result = np.asarray(fr.floor_divide(fr.asarray(a), fr.asarray(b)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


def test_real_bitwise_and_unregressed():
    a = np.array([0b1100, 0b1010], dtype="int64")
    b = np.array([0b1010, 0b0110], dtype="int64")
    expected = np.bitwise_and(a, b)
    result = np.asarray(fr.bitwise_and(fr.asarray(a), fr.asarray(b)))
    np.testing.assert_array_equal(result, expected)


def test_real_power_unregressed():
    a = np.array([2, 3, 4], dtype="int64")
    b = np.array([3, 2, 1], dtype="int64")
    expected = np.power(a, b)
    result = np.asarray(fr.power(fr.asarray(a), fr.asarray(b)))
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)
