"""Complex masked-array construction + round-trip (#868, R-1).

Pins ferray.ma's complex64/complex128 masked-array surface against the live
numpy.ma oracle (numpy 2.4.5). Every expected value is DERIVED from a live
numpy.ma call (R-CHAR-3), never literal-copied from the ferray side.

Scope of #868 (this build): construction, `.dtype`, `.filled()` (default
`1e20+0j` + custom), `.data`/`__array__`, `.mask`, `__repr__`, `__getitem__`
(masked-scalar return), `__setitem__`, `==`/`!=` (compute), `sum`/`prod`
(compute via the complex `ReduceAcc`). Ordering (`<`/`>`/`<=`/`>=`) COMPUTES
LEXICOGRAPHICALLY (numpy.ma does NOT raise — `np.less`/etc on complex compare
`(real, then imag)`; #874). Bitwise (`&`/`|`/`^`) RAISE `TypeError` (numpy
raises too — complex is not integer). Complex arithmetic (`+`/`-`/`*`/`/`/`**`)
and the real-collapsing
reductions (`mean`/`var`/`std`/`min`/`max`) RAISE a tracked `TypeError` for now
— numpy.ma SUPPORTS these, so the asserts below assert the CURRENT
(intentionally-incomplete) TypeError; they FLIP to value-equality assertions
when the ferray-ufunc complex-arithmetic prereq (#869) and complex reductions
(#873) land.
"""

import numpy as np
import pytest

import ferray as fr


def _data():
    return [1 + 2j, 3 - 1j, 5 + 0j]


def _mask():
    return [0, 1, 0]


# --- construction + dtype --------------------------------------------------

def test_complex128_construct_dtype():
    expected = str(np.ma.array(_data(), mask=_mask()).dtype)  # complex128
    f = fr.ma.array(_data(), mask=_mask())
    assert f.dtype == expected


def test_complex64_construct_dtype():
    expected = str(np.ma.array([1 + 2j], dtype=np.complex64).dtype)  # complex64
    f = fr.ma.array(np.array([1 + 2j], dtype=np.complex64))
    assert f.dtype == expected


def test_complex_construct_no_mask():
    expected = str(np.ma.array(_data()).dtype)
    f = fr.ma.array(_data())
    assert f.dtype == expected


# --- __array__ / .data (verbatim, masked slots keep stored value) ----------

def test_complex_array_egress_verbatim():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    # numpy's __array__ keeps the stored value at masked positions.
    np.testing.assert_array_equal(np.asarray(f), np.asarray(n))


def test_complex_data_property():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    np.testing.assert_array_equal(np.asarray(f.data), np.asarray(n.data))


# --- mask ------------------------------------------------------------------

def test_complex_mask():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    np.testing.assert_array_equal(np.asarray(f.mask), np.ma.getmaskarray(n))


# --- filled (default 1e20+0j + custom) -------------------------------------

def test_complex_filled_default():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    np.testing.assert_array_equal(f.filled(), n.filled())


def test_complex_fill_value_default():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(f.fill_value) == complex(n.fill_value)  # (1e+20+0j)


def test_complex_filled_custom():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    np.testing.assert_array_equal(f.filled(0 + 0j), n.filled(0 + 0j))


# --- __getitem__ (masked-scalar return) ------------------------------------

def test_complex_getitem_unmasked():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(f[0]) == complex(n[0])  # (1+2j)


def test_complex_getitem_masked_returns_masked():
    f = fr.ma.array(_data(), mask=_mask())
    # numpy returns the `masked` singleton at a masked scalar position.
    assert f[1] is fr.ma.masked


def test_complex_getitem_slice():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    sub_f = f[0:2]
    sub_n = n[0:2]
    np.testing.assert_array_equal(np.asarray(sub_f), np.asarray(sub_n))
    np.testing.assert_array_equal(np.asarray(sub_f.mask), np.ma.getmaskarray(sub_n))


# --- __setitem__ -----------------------------------------------------------

def test_complex_setitem_value():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    n[0] = 7 + 8j
    f[0] = 7 + 8j
    np.testing.assert_array_equal(f.filled(), n.filled())


def test_complex_setitem_masked():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    n[2] = np.ma.masked
    f[2] = fr.ma.masked
    np.testing.assert_array_equal(np.asarray(f.mask), np.ma.getmaskarray(n))


# --- compressed ------------------------------------------------------------

def test_complex_compressed():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    np.testing.assert_array_equal(f.compressed(), n.compressed())


# --- == / != (compute, masked-position override) ---------------------------

def test_complex_eq_compute():
    n_a = np.ma.array(_data(), mask=_mask())
    n_b = np.ma.array([1 + 2j, 9, 5], mask=[0, 0, 0])
    f_a = fr.ma.array(_data(), mask=_mask())
    f_b = fr.ma.array([1 + 2j, 9, 5], mask=[0, 0, 0])
    res_n = n_a == n_b
    res_f = f_a == f_b
    # data + mask must match numpy.ma's `_comparison` (masked-position override).
    np.testing.assert_array_equal(
        np.ma.getdata(res_n), np.asarray(res_f.data)
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(res_n), np.asarray(res_f.mask)
    )


def test_complex_ne_compute():
    n_a = np.ma.array(_data(), mask=_mask())
    n_b = np.ma.array([1 + 2j, 9, 5], mask=[0, 0, 0])
    f_a = fr.ma.array(_data(), mask=_mask())
    f_b = fr.ma.array([1 + 2j, 9, 5], mask=[0, 0, 0])
    res_n = n_a != n_b
    res_f = f_a != f_b
    np.testing.assert_array_equal(
        np.ma.getdata(res_n), np.asarray(res_f.data)
    )


# --- sum / prod (compute via complex ReduceAcc) ----------------------------

def test_complex_sum():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(f.sum()) == complex(n.sum())  # (6+2j)


def test_complex_prod():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(fr.ma.prod(f)) == complex(n.prod())  # (5+10j)


# --- ordering COMPUTES lexicographically (numpy.ma does NOT raise) ----------
# numpy.ma compares complex LEXICOGRAPHICALLY `(real, then imag)` via `np.less`/
# etc; it does NOT raise (verified live, numpy 2.4.5:
# `np.ma.array([1+2j]) < np.ma.array([1+3j])` -> `[True]`). The four ordering
# ops are fully covered by `test_divergence_ma_complex_audit.py` (#874); these
# two assert the value (data + mask union) against the live numpy.ma oracle.

def test_complex_lt_computes_lexicographically():
    n = np.ma.array(_data(), mask=_mask())
    nr = n < n
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    f = fr.ma.array(_data(), mask=_mask())
    fr_res = f < f
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask


def test_complex_gt_computes_lexicographically():
    n = np.ma.array(_data(), mask=_mask())
    nr = n > n
    exp_data = np.asarray(np.ma.getdata(nr)).tolist()
    exp_mask = np.asarray(np.ma.getmaskarray(nr)).tolist()

    f = fr.ma.array(_data(), mask=_mask())
    fr_res = f > f
    assert np.asarray(fr_res.data).tolist() == exp_data
    assert np.asarray(fr_res.mask).tolist() == exp_mask


# --- bitwise RAISES TypeError (numpy raises: complex is not integer) --------

def test_complex_bitwise_and_raises():
    f = fr.ma.array([1 + 2j, 3 + 4j])
    with pytest.raises(TypeError):
        _ = f & 1


# --- arithmetic COMPUTES (#869 landed) -------------------------------------
# numpy.ma computes complex `+`/`-`/`*`/`/`/`**`; #869 wired the ferray-ufunc
# complex-arithmetic prereq + the binding. These flipped from the pending
# TypeError pins to value+mask equality vs numpy.ma (full coverage lives in
# test_expansion_ma_complex_arith.py). Unary `-` (negative) is still tracked
# separately and stays a TypeError below.

def _eq_ma(f, n):
    fm = np.asarray(fr.ma.getmaskarray(f), dtype=bool)
    nm = np.ma.getmaskarray(n)
    np.testing.assert_array_equal(fm, nm)
    keep = ~nm
    np.testing.assert_allclose(
        np.asarray(f.data)[keep], np.asarray(n.data)[keep], rtol=1e-6, atol=1e-7
    )


def test_complex_add_computes_869():
    n = np.ma.array(_data(), mask=_mask()) + np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask()) + fr.ma.array(_data(), mask=_mask())
    _eq_ma(f, n)


def test_complex_mul_computes_869():
    n = np.ma.array(_data(), mask=_mask()) * 2
    f = fr.ma.array(_data(), mask=_mask()) * 2
    _eq_ma(f, n)


def test_complex_truediv_computes_869():
    n = np.ma.array(_data(), mask=_mask()) / 2
    f = fr.ma.array(_data(), mask=_mask()) / 2
    _eq_ma(f, n)


def test_complex_pow_computes_869():
    n = np.ma.array(_data(), mask=_mask()) ** 2
    f = fr.ma.array(_data(), mask=_mask()) ** 2
    _eq_ma(f, n)


def test_complex_neg_computes_869():
    # numpy.ma computes complex unary negate, mask-preserving
    # (numpy/ma/core.py:955); ferray matches after the Smith-division /
    # complex-negate fix (#876). Was: asserted a TypeError (the old divergence).
    n = -np.ma.array(_data(), mask=_mask())
    f = -fr.ma.array(_data(), mask=_mask())
    _eq_ma(f, n)


# --- mean / min / max COMPUTE (#873 landed) --------------------------------
# numpy.ma COMPUTES a complex mean (keeps the imaginary part, promotes to
# complex128) and lexicographic min/max (real, then imag). #873 wired these
# over the complex DynMa variants. Was: asserted a TypeError (the old gap).
# Full coverage lives in test_expansion_ma_complex_reduce.py.

def test_complex_mean_computes_873():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(f.mean()) == pytest.approx(complex(n.mean()))


def test_complex_min_computes_873():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(f.min()) == complex(n.min())


def test_complex_max_computes_873():
    n = np.ma.array(_data(), mask=_mask())
    f = fr.ma.array(_data(), mask=_mask())
    assert complex(f.max()) == complex(n.max())
