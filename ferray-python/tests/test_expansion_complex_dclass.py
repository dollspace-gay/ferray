"""Expansion suite for #938 (D-class): top-level ops that numpy COMPUTES on
complex input where ferray previously raised `TypeError`. Every expected value
is derived LIVE from numpy 2.4 (R-CHAR-3 — never copied from the ferray side).

Grouped by the seam each op reuses:
  * dtype-passthrough via the #933 `DynMarshal` seam (take/nonzero/pad/
    broadcast_to/count_nonzero/eye/identity),
  * compute composed from complex arithmetic (#924) (trace/kron/tensordot/
    vander/geomspace),
  * sorted/set composed from the #932 `cplx_sort_cmp` lexicographic order
    (searchsorted/partition/intersect1d/union1d/setdiff1d/setxor1d/isin).

Real-dtype paths must be UNREGRESSED — the final block re-checks the real
contract for every op.

Run:
    cd ferray-python && PYTHONPATH=python python3 -m pytest \
        tests/test_expansion_complex_dclass.py -q
"""
import numpy as np
import ferray as fr
import pytest

# Fixtures (complex128 and complex64 widths).
Z = [3 + 4j, 1 - 2j, -5 + 12j]
M = [[1 + 1j, 2 - 1j], [3 + 0j, 4 + 2j]]
Z32 = np.asarray(Z, dtype=np.complex64)


def _np(x):
    return np.asarray(x)


# ---------------------------------------------------------------------------
# dtype-passthrough (DynMarshal seam #933) — accept complex, preserve dtype
# ---------------------------------------------------------------------------

def test_take_complex_preserves_dtype():
    exp = np.take(_np(Z), [2, 0, 1])
    got = _np(fr.take(fr.array(Z), [2, 0, 1]))
    assert np.iscomplexobj(got)
    np.testing.assert_array_equal(got, exp)


def test_take_complex64_width_preserved():
    exp = np.take(Z32, [0, 2])
    got = _np(fr.take(fr.array(Z32), [0, 2]))
    assert got.dtype == np.complex64
    np.testing.assert_array_equal(got, exp)


def test_nonzero_complex_re_or_im():
    src = [0j, 0 + 5j, 2 + 0j, 0j, 1 + 1j]
    exp = np.nonzero(_np(src))[0]
    got = _np(fr.nonzero(fr.array(src))[0])
    np.testing.assert_array_equal(got, exp)


def test_count_nonzero_complex():
    src = [0j, 0 + 5j, 2 + 0j, 0j, 1 + 1j]
    exp = np.count_nonzero(_np(src))
    assert int(fr.count_nonzero(fr.array(src))) == int(exp)


def test_pad_complex_constant_zero_fill():
    exp = np.pad(_np(Z), (2, 1))
    got = _np(fr.pad(fr.array(Z), (2, 1)))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, exp)


def test_pad_complex_edge_mode():
    exp = np.pad(_np(Z), (1, 1), mode="edge")
    got = _np(fr.pad(fr.array(Z), (1, 1), mode="edge"))
    np.testing.assert_allclose(got, exp)


def test_broadcast_to_complex():
    exp = np.broadcast_to(_np(Z), (4, 3))
    got = _np(fr.broadcast_to(fr.array(Z), (4, 3)))
    assert np.iscomplexobj(got)
    np.testing.assert_allclose(got, exp)


def test_eye_complex_dtype():
    exp = np.eye(3, dtype=complex)
    np.testing.assert_allclose(_np(fr.eye(3, dtype=complex)), exp)


def test_eye_complex_offset_diag():
    exp = np.eye(3, k=1, dtype=complex)
    np.testing.assert_allclose(_np(fr.eye(3, k=1, dtype=complex)), exp)


def test_identity_complex_dtype():
    exp = np.identity(3, dtype=complex)
    np.testing.assert_allclose(_np(fr.identity(3, dtype=complex)), exp)


# ---------------------------------------------------------------------------
# compute (composed complex arithmetic #924)
# ---------------------------------------------------------------------------

def test_trace_complex_sum_of_diagonal():
    exp = np.trace(_np(M))
    assert _np(fr.trace(fr.array(M))).item() == exp


def test_trace_complex_nonsquare():
    A = [[1 + 1j, 2 + 0j, 3 - 1j], [4 + 0j, 5 + 2j, 6 + 0j]]
    exp = np.trace(_np(A))
    assert _np(fr.trace(fr.array(A))).item() == exp


def test_kron_complex_matrix():
    exp = np.kron(_np(M), _np(M))
    np.testing.assert_allclose(_np(fr.kron(fr.array(M), fr.array(M))), exp)


def test_kron_complex_rectangular():
    A = [[1 + 1j, 2 - 1j]]
    B = [[1j, 2 + 0j], [3 + 0j, 4 - 1j]]
    exp = np.kron(_np(A), _np(B))
    np.testing.assert_allclose(_np(fr.kron(fr.array(A), fr.array(B))), exp)


def test_tensordot_complex_full_contraction():
    exp = np.tensordot(_np(M), _np(M))
    assert _np(fr.tensordot(fr.array(M), fr.array(M))).item() == exp


def test_tensordot_complex_axes_one():
    exp = np.tensordot(_np(M), _np(M), axes=1)
    np.testing.assert_allclose(_np(fr.tensordot(fr.array(M), fr.array(M), axes=1)), exp)


def test_vander_complex_powers():
    exp = np.vander(_np(Z))
    np.testing.assert_allclose(_np(fr.vander(fr.array(Z))), exp)


def test_vander_complex_increasing_n():
    exp = np.vander(_np(Z), N=4, increasing=True)
    np.testing.assert_allclose(_np(fr.vander(fr.array(Z), n=4, increasing=True)), exp)


def test_geomspace_complex_endpoints():
    exp = np.geomspace(1 + 1j, 8 + 8j, 3)
    np.testing.assert_allclose(_np(fr.geomspace(1 + 1j, 8 + 8j, 3)), exp)


def test_geomspace_complex_num5():
    exp = np.geomspace(2 + 0j, 32 + 0j, 5)
    np.testing.assert_allclose(_np(fr.geomspace(2 + 0j, 32 + 0j, 5)), exp)


# ---------------------------------------------------------------------------
# sorted / set (composed #932 lexicographic order)
# ---------------------------------------------------------------------------

def test_searchsorted_complex_scalar_is_0d():
    base = [1 + 0j, 2 + 0j, 3 + 0j]
    exp = np.searchsorted(_np(base), 2 + 1j)
    got = fr.searchsorted(fr.array(base), 2 + 1j)
    assert int(got) == int(exp)


def test_searchsorted_complex_side_right():
    base = [1 + 0j, 2 + 0j, 2 + 0j, 3 + 0j]
    exp = np.searchsorted(_np(base), [2 + 0j], side="right")
    got = _np(fr.searchsorted(fr.array(base), fr.array([2 + 0j]), side="right"))
    np.testing.assert_array_equal(got, exp)


def test_partition_complex_kth_in_place():
    exp = np.partition(_np(Z), 1)
    got = _np(fr.partition(fr.array(Z), 1))
    np.testing.assert_array_equal(got, exp)


def test_intersect1d_complex():
    exp = np.intersect1d(_np(Z), [1 - 2j, 3 + 4j, 99 + 0j])
    got = _np(fr.intersect1d(fr.array(Z), [1 - 2j, 3 + 4j, 99 + 0j]))
    np.testing.assert_array_equal(got, exp)


def test_union1d_complex():
    exp = np.union1d(_np(Z), [1 - 2j, 7 + 7j])
    got = _np(fr.union1d(fr.array(Z), [1 - 2j, 7 + 7j]))
    np.testing.assert_array_equal(got, exp)


def test_setdiff1d_complex():
    exp = np.setdiff1d(_np(Z), [1 - 2j])
    got = _np(fr.setdiff1d(fr.array(Z), [1 - 2j]))
    np.testing.assert_array_equal(got, exp)


def test_setxor1d_complex():
    exp = np.setxor1d(_np(Z), [1 - 2j, 100 + 0j])
    got = _np(fr.setxor1d(fr.array(Z), [1 - 2j, 100 + 0j]))
    np.testing.assert_array_equal(got, exp)


def test_isin_complex():
    exp = np.isin(_np(Z), [1 - 2j, -5 + 12j])
    got = _np(fr.isin(fr.array(Z), [1 - 2j, -5 + 12j]))
    np.testing.assert_array_equal(got, exp)


# ---------------------------------------------------------------------------
# real-dtype paths UNREGRESSED — same ops on real input
# ---------------------------------------------------------------------------

def test_real_paths_unregressed():
    R = [3.0, 1.0, -5.0, 2.0]
    Mr = [[1.0, 2.0], [3.0, 4.0]]
    np.testing.assert_array_equal(_np(fr.take(fr.array(R), [0, 2])), np.take(_np(R), [0, 2]))
    np.testing.assert_array_equal(_np(fr.nonzero(fr.array(R))[0]), np.nonzero(_np(R))[0])
    assert int(fr.count_nonzero(fr.array(R))) == int(np.count_nonzero(_np(R)))
    np.testing.assert_allclose(_np(fr.pad(fr.array(R), (1, 1))), np.pad(_np(R), (1, 1)))
    np.testing.assert_allclose(_np(fr.broadcast_to(fr.array(R), (2, 4))),
                               np.broadcast_to(_np(R), (2, 4)))
    np.testing.assert_allclose(_np(fr.eye(3)), np.eye(3))
    np.testing.assert_allclose(_np(fr.identity(3)), np.identity(3))
    assert _np(fr.trace(fr.array(Mr))).item() == np.trace(_np(Mr))
    np.testing.assert_allclose(_np(fr.kron(fr.array(Mr), fr.array(Mr))),
                               np.kron(_np(Mr), _np(Mr)))
    assert _np(fr.tensordot(fr.array(Mr), fr.array(Mr))).item() == \
        np.tensordot(_np(Mr), _np(Mr)).item()
    np.testing.assert_allclose(_np(fr.vander(fr.array(R))), np.vander(_np(R)))
    np.testing.assert_allclose(_np(fr.geomspace(1.0, 1000.0, 4)), np.geomspace(1.0, 1000.0, 4))
    np.testing.assert_array_equal(_np(fr.partition(fr.array(R), 1)), np.partition(_np(R), 1))
    np.testing.assert_array_equal(_np(fr.intersect1d(fr.array(R), [1.0, 2.0])),
                                  np.intersect1d(_np(R), [1.0, 2.0]))
    np.testing.assert_array_equal(_np(fr.union1d(fr.array(R), [9.0])),
                                  np.union1d(_np(R), [9.0]))
    np.testing.assert_array_equal(_np(fr.setdiff1d(fr.array(R), [1.0])),
                                  np.setdiff1d(_np(R), [1.0]))
    np.testing.assert_array_equal(_np(fr.isin(fr.array(R), [1.0, 2.0])),
                                  np.isin(_np(R), [1.0, 2.0]))
    base = [1.0, 2.0, 3.0]
    np.testing.assert_array_equal(
        _np(fr.searchsorted(fr.array(base), fr.array([2.5]))),
        np.searchsorted(_np(base), [2.5]),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
