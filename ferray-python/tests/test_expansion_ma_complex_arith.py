"""Complex masked-array arithmetic +,-,*,/,** (#869, R-3).

Pins ferray.ma's complex64/complex128 masked arithmetic against the live
numpy.ma oracle (numpy 2.4.5). Every expected value is DERIVED from a live
numpy.ma call (R-CHAR-3), never literal-copied from the ferray side.

Scope of #869 (this build, the ferray-ufunc complex-arithmetic prereq +
binding wiring):
  - `+`/`-`/`*`/`/`/`**` COMPUTE over the complex variant (data + mask).
  - mask = operand union for `+`/`-`/`*`/`**`; `/` is a DOMAINED op that
    materializes a real mask and OR's in divisor==0 (complex zero re==0 &&
    im==0) / non-finite quotient.
  - real-operand promotion to complex (`(1+2j)+1` → `2+2j`) via NEP-50
    result_type routing both operands through the complex arm.
  - reflected ops (`2/a`, `1+a`, …) compute `other OP self`.
  - complex64 + complex64 stays complex64.
  - `+`/`-`/`*`/`**` keep `nomask` when both operands are nomask; `/`
    MATERIALIZES a real mask (numpy `_DomainedBinaryOperation`).
  - `//`/`%` over complex RAISE `TypeError` (numpy has no floor_divide /
    remainder loop for complex — permanent).

A helper asserts data + mask elementwise against the numpy.ma result so the
test is exactly numpy.ma-faithful.
"""

import numpy as np
import pytest

import ferray as fr


def _assert_ma_equal(f, n):
    """ferray masked array f matches numpy.ma masked array n (data + mask)."""
    assert str(f.dtype) == str(n.dtype), f"dtype {f.dtype} != {n.dtype}"
    # getmaskarray materializes an all-False mask for a nomask result, so it
    # compares cleanly against numpy's getmaskarray regardless of the nomask
    # sentinel identity.
    fm = np.asarray(fr.ma.getmaskarray(f), dtype=bool)
    nm = np.ma.getmaskarray(n)
    np.testing.assert_array_equal(fm, nm)
    # Compare unmasked data only (masked slots may carry arbitrary fill data).
    fd = np.asarray(f.data)
    nd = np.asarray(n.data)
    keep = ~nm
    np.testing.assert_allclose(fd[keep], nd[keep], rtol=1e-6, atol=1e-7)


def _a():
    return np.ma.array([1 + 2j, 3 - 1j], mask=[0, 1])


def _b():
    return np.ma.array([2 + 0j, 1 + 1j])


# --- core +,-,* (complex128) ----------------------------------------------

def test_complex_add():
    n = _a() + _b()
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) + fr.ma.array([2 + 0j, 1 + 1j])
    _assert_ma_equal(f, n)


def test_complex_sub():
    n = _a() - _b()
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) - fr.ma.array([2 + 0j, 1 + 1j])
    _assert_ma_equal(f, n)


def test_complex_mul():
    n = _a() * _b()
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) * fr.ma.array([2 + 0j, 1 + 1j])
    _assert_ma_equal(f, n)


def test_complex_mul_cross_term():
    # exercises the (ac-bd, ad+bc) cross-product, not a component-wise op.
    n = np.ma.array([1 + 2j]) * np.ma.array([3 + 4j])
    f = fr.ma.array([1 + 2j]) * fr.ma.array([3 + 4j])
    _assert_ma_equal(f, n)


# --- true division (domained) ---------------------------------------------

def test_complex_truediv():
    n = _a() / _b()
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) / fr.ma.array([2 + 0j, 1 + 1j])
    _assert_ma_equal(f, n)


def test_complex_truediv_by_complex_zero_domain_masks():
    # position 0 divisor is complex zero -> domain-masked; position 1 already
    # masked on the left operand.
    n = _a() / np.ma.array([0j, 1 + 0j])
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) / fr.ma.array([0j, 1 + 0j])
    np.testing.assert_array_equal(np.asarray(f.mask, dtype=bool), np.ma.getmaskarray(n))


def test_complex_truediv_scalar():
    n = np.ma.array([1 + 2j]) / 2
    f = fr.ma.array([1 + 2j]) / 2
    _assert_ma_equal(f, n)


# --- power (int / float / complex exponent) -------------------------------

def test_complex_pow_int_exact():
    # (1+2j)**2 == -3+4j EXACTLY (npy_cpow integer fast path, no float drift).
    n = _a() ** 2
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) ** 2
    _assert_ma_equal(f, n)
    # exactness on the unmasked slot
    assert complex(np.asarray(f.data)[0]) == complex(np.asarray(n.data)[0]) == (-3 + 4j)


def test_complex_pow_float_exponent():
    n = np.ma.array([1 + 2j]) ** 0.5
    f = fr.ma.array([1 + 2j]) ** 0.5
    _assert_ma_equal(f, n)


def test_complex_pow_complex_exponent():
    n = np.ma.array([1 + 2j]) ** (1 + 1j)
    f = fr.ma.array([1 + 2j]) ** (1 + 1j)
    _assert_ma_equal(f, n)


# --- real-operand promotion to complex (NEP-50 result_type) ----------------

def test_complex_plus_python_scalar_promotes():
    n = _a() + 1
    f = fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1]) + 1
    _assert_ma_equal(f, n)


def test_complex_plus_real_array_promotes():
    n = np.ma.array([1 + 2j, 3 + 0j]) + np.array([1.0, 2.0])
    f = fr.ma.array([1 + 2j, 3 + 0j]) + np.array([1.0, 2.0])
    _assert_ma_equal(f, n)


# --- reflected ops ---------------------------------------------------------

def test_complex_reflected_div():
    n = 2 / np.ma.array([1 + 1j])
    f = 2 / fr.ma.array([1 + 1j])
    _assert_ma_equal(f, n)


def test_complex_reflected_add():
    n = 1 + _a()
    f = 1 + fr.ma.array([1 + 2j, 3 - 1j], mask=[0, 1])
    _assert_ma_equal(f, n)


def test_complex_reflected_sub():
    n = 1 - np.ma.array([1 + 2j])
    f = 1 - fr.ma.array([1 + 2j])
    _assert_ma_equal(f, n)


# --- complex64 stays complex64 --------------------------------------------

def test_complex64_add_keeps_width():
    na = np.ma.array([1 + 2j], dtype=np.complex64)
    nb = np.ma.array([1 + 1j], dtype=np.complex64)
    n = na + nb
    f = fr.ma.array(np.array([1 + 2j], dtype=np.complex64)) + fr.ma.array(
        np.array([1 + 1j], dtype=np.complex64)
    )
    _assert_ma_equal(f, n)


def test_complex64_div_keeps_width():
    na = np.ma.array([1 + 2j], dtype=np.complex64)
    n = na / np.ma.array(np.array([2 + 0j], dtype=np.complex64))
    f = fr.ma.array(np.array([1 + 2j], dtype=np.complex64)) / fr.ma.array(
        np.array([2 + 0j], dtype=np.complex64)
    )
    _assert_ma_equal(f, n)


# --- nomask materialization rules -----------------------------------------

def test_complex_add_keeps_nomask():
    # both operands nomask -> result nomask (numpy keeps `mask is nomask`).
    n = np.ma.array([1 + 2j]) + np.ma.array([1 + 0j])
    assert np.ma.getmask(n) is np.ma.nomask
    f = fr.ma.array([1 + 2j]) + fr.ma.array([1 + 0j])
    assert f.mask is np.ma.nomask


def test_complex_mul_keeps_nomask():
    f = fr.ma.array([1 + 2j]) * fr.ma.array([2 + 0j])
    assert f.mask is np.ma.nomask


def test_complex_pow_keeps_nomask():
    f = fr.ma.array([1 + 2j]) ** 2
    assert f.mask is np.ma.nomask


def test_complex_truediv_materializes_mask():
    # `/` ALWAYS materializes a real (all-False here) mask, even nomask in.
    n = np.ma.array([1 + 2j]) / np.ma.array([1 + 0j])
    assert np.ma.getmaskarray(n).shape == (1,)
    f = fr.ma.array([1 + 2j]) / fr.ma.array([1 + 0j])
    # a real mask exists (shape-(1,) all-False), not the nomask sentinel.
    assert f.mask is not np.ma.nomask
    np.testing.assert_array_equal(np.asarray(f.mask, dtype=bool), np.array([False]))


# --- mask union -----------------------------------------------------------

def test_complex_mask_union():
    na = np.ma.array([1 + 2j, 3 - 1j, 5 + 0j], mask=[1, 0, 0])
    nb = np.ma.array([2 + 0j, 1 + 1j, 0 + 2j], mask=[0, 1, 0])
    n = na * nb
    fa = fr.ma.array([1 + 2j, 3 - 1j, 5 + 0j], mask=[1, 0, 0])
    fb = fr.ma.array([2 + 0j, 1 + 1j, 0 + 2j], mask=[0, 1, 0])
    f = fa * fb
    _assert_ma_equal(f, n)


# --- //, % raise TypeError (permanent — no complex loop in numpy) ----------

def test_complex_floordiv_raises_typeerror():
    a = np.ma.array([1 + 2j])
    with pytest.raises(TypeError):
        a // 2  # confirm numpy raises (oracle)
    with pytest.raises(TypeError):
        fr.ma.array([1 + 2j]) // 2


def test_complex_mod_raises_typeerror():
    a = np.ma.array([1 + 2j])
    with pytest.raises(TypeError):
        a % 2  # confirm numpy raises (oracle)
    with pytest.raises(TypeError):
        fr.ma.array([1 + 2j]) % 2
