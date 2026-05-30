"""Complex masked-array `.real`/`.imag`/`.conjugate()`/`.conj()`/`abs()` (#870, R-2).

Pins ferray.ma's complex (and real-dtype) masked-array attribute/method surface
against the live numpy.ma oracle (numpy 2.4.x). Every expected value is DERIVED
from a live numpy.ma call (R-CHAR-3) — never literal-copied from the ferray
side.

Scope of #870 (this build):
  - `.real` / `.imag` (getter properties — attribute VIEWS, so they PRESERVE the
    mask state verbatim: nomask stays nomask). complex -> real-dtype magnitude-
    width result (complex128 -> float64, complex64 -> float32); real-dtype
    `.real` -> the array (same dtype), `.imag` -> zeros (same dtype).
  - `.conjugate()` / `.conj()` (methods; conj is an alias — ufunc calls, so they
    MATERIALIZE a real mask even from a nomask operand). complex -> same complex
    width; real-dtype -> identity (same dtype).
  - `abs()` / `np.abs()` (the `__abs__` dunder — a ufunc call). complex -> REAL
    magnitude (complex128 -> float64, complex64 -> float32). The pending-#870
    raise the #868 build installed in the complex arm is REPLACED by this
    compute path.

Mask semantics confirmed live (numpy 2.4.x):
  - `.real`/`.imag` are property views: `np.ma.array([1+2j]).real.mask is
    np.ma.nomask` -> True (no materialization).
  - `.conjugate()`/`abs()` are ufunc calls:
    `np.ma.array([1+2j]).conjugate().mask is np.ma.nomask` -> False, and
    `abs(np.ma.array([1+2j])).mask is np.ma.nomask` -> False (materialized).
"""

import numpy as np
import pytest

import ferray as fr


def _cdata():
    return [1 + 2j, 3 - 4j, 0 + 1j]


def _mask():
    return [0, 1, 0]


def _assert_ma_equal(fr_ma, np_ma):
    """Assert a ferray masked array equals a numpy.ma one in dtype, data
    (under the mask too), and mask — all derived from numpy (R-CHAR-3)."""
    assert str(fr_ma.dtype) == str(np_ma.dtype), (fr_ma.dtype, np_ma.dtype)
    fr_data = np.asarray(fr_ma.data)
    np_data = np.ma.getdata(np_ma)
    np.testing.assert_allclose(fr_data, np_data, rtol=1e-6, atol=0)
    fr_mask = fr_ma.mask
    np_mask = np.ma.getmaskarray(np_ma)
    if fr_mask is np.ma.nomask:
        # ferray nomask <-> numpy all-False (numpy.ma.getmaskarray fills nomask
        # to all-False); accept either as "no real mask".
        assert not np_mask.any() or (np_ma.mask is np.ma.nomask)
    else:
        np.testing.assert_array_equal(np.asarray(fr_mask), np_mask)


# --- complex128 .real / .imag ----------------------------------------------

def test_complex128_real_dtype_and_values():
    f = fr.ma.array(_cdata(), mask=_mask()).real
    n = np.ma.array(_cdata(), mask=_mask()).real
    assert str(f.dtype) == "float64" == str(n.dtype)
    _assert_ma_equal(f, n)


def test_complex128_imag_dtype_and_values():
    f = fr.ma.array(_cdata(), mask=_mask()).imag
    n = np.ma.array(_cdata(), mask=_mask()).imag
    assert str(f.dtype) == "float64" == str(n.dtype)
    _assert_ma_equal(f, n)


# --- complex64 .real / .imag (float32 result) ------------------------------

def test_complex64_real_is_float32():
    src = [1 + 2j, 3 - 4j]
    f = fr.ma.array(src, mask=[0, 1], dtype="complex64").real
    n = np.ma.array(src, mask=[0, 1], dtype=np.complex64).real
    assert str(f.dtype) == "float32" == str(n.dtype)
    _assert_ma_equal(f, n)


def test_complex64_imag_is_float32():
    src = [1 + 2j, 3 - 4j]
    f = fr.ma.array(src, mask=[0, 1], dtype="complex64").imag
    n = np.ma.array(src, mask=[0, 1], dtype=np.complex64).imag
    assert str(f.dtype) == "float32" == str(n.dtype)
    _assert_ma_equal(f, n)


# --- .conjugate() / .conj() (complex, same width) --------------------------

def test_complex128_conjugate():
    f = fr.ma.array(_cdata(), mask=_mask()).conjugate()
    n = np.ma.array(_cdata(), mask=_mask()).conjugate()
    assert str(f.dtype) == "complex128" == str(n.dtype)
    _assert_ma_equal(f, n)


def test_complex128_conj_alias_matches_conjugate():
    a = fr.ma.array(_cdata(), mask=_mask())
    _assert_ma_equal(a.conj(), np.ma.array(_cdata(), mask=_mask()).conj())
    # conj is an alias for conjugate (same result).
    np.testing.assert_array_equal(
        np.asarray(a.conj().data), np.asarray(a.conjugate().data)
    )


def test_complex64_conjugate_is_complex64():
    src = [1 + 2j, 3 - 4j]
    f = fr.ma.array(src, mask=[0, 1], dtype="complex64").conjugate()
    n = np.ma.array(src, mask=[0, 1], dtype=np.complex64).conjugate()
    assert str(f.dtype) == "complex64" == str(n.dtype)
    _assert_ma_equal(f, n)


# --- abs() / np.abs() — REAL magnitude (replaces the pending-#870 raise) ----

def test_complex128_abs_is_real_magnitude_float64():
    f = abs(fr.ma.array(_cdata(), mask=_mask()))
    n = abs(np.ma.array(_cdata(), mask=_mask()))
    assert str(f.dtype) == "float64" == str(n.dtype)
    _assert_ma_equal(f, n)


def test_complex64_abs_is_float32():
    src = [3 + 4j, 1 + 0j]
    f = abs(fr.ma.array(src, mask=[0, 1], dtype="complex64"))
    n = abs(np.ma.array(src, mask=[0, 1], dtype=np.complex64))
    assert str(f.dtype) == "float32" == str(n.dtype)
    _assert_ma_equal(f, n)


def test_complex_np_abs_magnitude_values():
    # `np.abs(fr_ma)` egresses via numpy's array protocol (the ferray pyclass
    # is not a numpy.ma subclass, so np.abs returns a bare ndarray of the
    # magnitudes — the masked positions still carry the under-mask magnitude,
    # matching numpy.ma.getdata). The builtin `abs()` -> `__abs__` is the
    # mask-preserving path tested above; here we only confirm the magnitudes.
    f = np.abs(fr.ma.array(_cdata(), mask=_mask()))
    n = np.ma.getdata(np.abs(np.ma.array(_cdata(), mask=_mask())))
    assert str(f.dtype) == "float64"
    np.testing.assert_allclose(np.asarray(f), n, rtol=1e-6)


def test_complex_abs_known_magnitude_value():
    # abs(3+4j) == 5.0 (Pythagorean), derived live from numpy.
    f = abs(fr.ma.array([3 + 4j], mask=[0]))
    n = abs(np.ma.array([3 + 4j], mask=[0]))
    assert float(np.asarray(f.data)[0]) == float(np.ma.getdata(n)[0]) == 5.0


# --- real-dtype .real / .imag / .conjugate() (identity / zeros) -------------

@pytest.mark.parametrize("dtype", ["int64", "int32", "float64", "float32", "uint8"])
def test_real_dtype_real_is_identity(dtype):
    src = [1, 2, 3]
    f = fr.ma.array(src, mask=[0, 1, 0], dtype=dtype).real
    n = np.ma.array(src, mask=[0, 1, 0], dtype=getattr(np, dtype)).real
    assert str(f.dtype) == dtype == str(n.dtype)
    _assert_ma_equal(f, n)


@pytest.mark.parametrize("dtype", ["int64", "int32", "float64", "float32"])
def test_real_dtype_imag_is_zeros_same_dtype(dtype):
    src = [1, 2, 3]
    f = fr.ma.array(src, mask=[0, 1, 0], dtype=dtype).imag
    n = np.ma.array(src, mask=[0, 1, 0], dtype=getattr(np, dtype)).imag
    assert str(f.dtype) == dtype == str(n.dtype)
    # numpy .imag on a real array -> all zeros, same dtype.
    np.testing.assert_array_equal(np.asarray(f.data), np.ma.getdata(n))
    _assert_ma_equal(f, n)


@pytest.mark.parametrize("dtype", ["int64", "float64"])
def test_real_dtype_conjugate_is_identity(dtype):
    src = [1, 2, 3]
    f = fr.ma.array(src, mask=[0, 1, 0], dtype=dtype).conjugate()
    n = np.ma.array(src, mask=[0, 1, 0], dtype=getattr(np, dtype)).conjugate()
    assert str(f.dtype) == dtype == str(n.dtype)
    _assert_ma_equal(f, n)


# --- mask preservation: .real/.imag are VIEWS (preserve nomask) -------------

def test_complex_real_preserves_nomask():
    f = fr.ma.array(_cdata()).real  # no mask= -> nomask
    assert f.mask is np.ma.nomask
    assert np.ma.array(_cdata()).real.mask is np.ma.nomask


def test_complex_imag_preserves_nomask():
    f = fr.ma.array(_cdata()).imag
    assert f.mask is np.ma.nomask
    assert np.ma.array(_cdata()).imag.mask is np.ma.nomask


def test_real_dtype_real_preserves_nomask():
    f = fr.ma.array([1, 2, 3]).real
    assert f.mask is np.ma.nomask
    assert np.ma.array([1, 2, 3]).real.mask is np.ma.nomask


# --- mask materialization: conjugate()/abs() are ufunc calls ----------------

def test_complex_conjugate_materializes_mask_from_nomask():
    # numpy.ma.conjugate is a ufunc -> materializes a real all-False mask.
    f = fr.ma.array(_cdata()).conjugate()
    n = np.ma.array(_cdata()).conjugate()
    assert (n.mask is np.ma.nomask) is False
    assert f.mask is not np.ma.nomask
    np.testing.assert_array_equal(np.asarray(f.mask), np.ma.getmaskarray(n))


def test_complex_abs_materializes_mask_from_nomask():
    f = abs(fr.ma.array(_cdata()))
    n = abs(np.ma.array(_cdata()))
    assert (n.mask is np.ma.nomask) is False
    assert f.mask is not np.ma.nomask
    np.testing.assert_array_equal(np.asarray(f.mask), np.ma.getmaskarray(n))


# --- mask carried with values across a real mask ----------------------------

def test_complex_real_carries_real_mask():
    f = fr.ma.array(_cdata(), mask=_mask()).real
    n = np.ma.array(_cdata(), mask=_mask()).real
    np.testing.assert_array_equal(np.asarray(f.mask), np.ma.getmaskarray(n))


def test_complex_abs_carries_real_mask():
    f = abs(fr.ma.array(_cdata(), mask=_mask()))
    n = abs(np.ma.array(_cdata(), mask=_mask()))
    np.testing.assert_array_equal(np.asarray(f.mask), np.ma.getmaskarray(n))


# --- getitem-then-.real (compose with the gather path) ----------------------

def test_getitem_slice_then_real():
    a = fr.ma.array(_cdata(), mask=_mask())
    na = np.ma.array(_cdata(), mask=_mask())
    f = a[0:2].real
    n = na[0:2].real
    assert str(f.dtype) == "float64" == str(n.dtype)
    _assert_ma_equal(f, n)


def test_getitem_slice_then_abs():
    a = fr.ma.array(_cdata(), mask=_mask())
    na = np.ma.array(_cdata(), mask=_mask())
    f = abs(a[0:2])
    n = abs(na[0:2])
    _assert_ma_equal(f, n)


# --- conjugate round-trip: conj(conj(z)) == z -------------------------------

def test_complex_conjugate_involution():
    a = fr.ma.array(_cdata(), mask=_mask())
    n = np.ma.array(_cdata(), mask=_mask())
    _assert_ma_equal(a.conjugate().conjugate(), n.conjugate().conjugate())
