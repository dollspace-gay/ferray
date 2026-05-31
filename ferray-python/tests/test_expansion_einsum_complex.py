"""einsum complex / non-real dtype operands compute via numpy-delegate (R-CODE-4 #965).

Before the fix the einsum binding coerced every operand to float32/float64,
silently DROPPING the imaginary part of complex operands. The fix routes any
non-real-dtype einsum (complex64/128, float16, mixed real+complex) to
numpy.einsum, which performs the full complex contraction (no conjugation) +
NEP-50 promotion. The real path stays byte-identical.

Oracle = live numpy (R-CHAR-3): expected values come from numpy.einsum, never
literal-copied from ferray.
"""

import numpy as np
import pytest

import ferray as fr


def _check(subscripts, *operands):
    fr_out = np.asarray(fr.einsum(subscripts, *(fr.array(o) for o in operands)))
    np_out = np.einsum(subscripts, *operands)
    assert fr_out.dtype == np_out.dtype, (subscripts, fr_out.dtype, np_out.dtype)
    assert fr_out.shape == np_out.shape, (subscripts, fr_out.shape, np_out.shape)
    assert np.allclose(fr_out, np_out), (subscripts, fr_out, np_out)


# --- the exact #965 corruption case ---------------------------------------
def test_matmul_complex128_keeps_imag():
    a = np.array([[1 + 1j, 0], [0, 1]])
    _check("ij,jk->ik", a, a)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_various_subscripts(dtype):
    a = np.array([[1 + 1j, 2 - 3j], [4j, 5]], dtype=dtype)
    b = np.array([[2 + 1j, 0], [1 - 1j, 3j]], dtype=dtype)
    v = np.array([1 + 1j, 2j, 3 - 1j], dtype=dtype)
    _check("ij,jk->ik", a, b)
    _check("ii->i", a)
    _check("ij->ji", a)
    _check("i,i->", v, v)
    _check("ij,ij->", a, b)


def test_mixed_real_complex_promotes_to_complex():
    c = np.array([[1 + 1j, 0], [0, 1]])
    r = np.array([[2, 0], [0, 2]], dtype=np.float64)
    _check("ij,jk->ik", c, r)
    _check("ij,jk->ik", r, c)


def test_float16_computes_as_float16():
    a = np.array([[1, 2], [3, 4]], dtype=np.float16)
    b = np.eye(2, dtype=np.float16)
    _check("ij,jk->ik", a, b)


def test_optimize_kwarg_via_numpy_path():
    # The current binding signature is (subscripts, *operands); optimize is not
    # surfaced, but numpy's own default contraction must still match.
    a = np.array([[1 + 1j, 2], [3, 4 - 1j]])
    _check("ij,jk,kl->il", a, a, a)


# --- the real einsum path must stay UNCHANGED (byte-identical) --------------
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_real_einsum_unchanged(dtype):
    a = np.array([[1, 2], [3, 4]], dtype=dtype)
    b = np.array([[5, 6], [7, 8]], dtype=dtype)
    fr_out = np.asarray(fr.einsum("ij,jk->ik", fr.array(a), fr.array(b)))
    np_out = np.einsum("ij,jk->ik", a, b)
    assert fr_out.dtype == np_out.dtype
    assert np.array_equal(fr_out, np_out)
