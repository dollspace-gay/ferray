"""Round-trip parity tests for the ferray file-IO surface (refs #832).

Every expected value is constructed from a *live* numpy call (R-CHAR-3):
each test writes with one library and reads with the other (or asserts the
on-disk bytes numpy itself produced), never a literal copied from ferray.

Covers save/load (.npy), savez/savez_compressed/load (.npz), and
savetxt/loadtxt/genfromtxt (delimited text), wired over ferray-io.
"""

import os
import tempfile

import numpy as np
import pytest

import ferray as fr


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# .npy save / load
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_float64(tmpdir):
    p = os.path.join(tmpdir, "f64.npy")
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    fr.save(p, a)
    back = fr.load(p)
    assert back.dtype == a.dtype
    assert back.shape == a.shape
    np.testing.assert_array_equal(back, a)


def test_save_load_roundtrip_int64(tmpdir):
    p = os.path.join(tmpdir, "i64.npy")
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    fr.save(p, a)
    back = fr.load(p)
    assert back.dtype == np.int64
    np.testing.assert_array_equal(back, a)


def test_save_load_roundtrip_bool(tmpdir):
    p = os.path.join(tmpdir, "b.npy")
    a = np.array([True, False, True, True])
    fr.save(p, a)
    back = fr.load(p)
    assert back.dtype == np.bool_
    np.testing.assert_array_equal(back, a)


@pytest.mark.parametrize(
    "dt", [np.float32, np.int32, np.int16, np.int8, np.uint8, np.uint16, np.uint32, np.uint64]
)
def test_save_load_roundtrip_dtypes(tmpdir, dt):
    p = os.path.join(tmpdir, f"{np.dtype(dt).name}.npy")
    a = np.arange(6).astype(dt).reshape(2, 3)
    fr.save(p, a)
    back = fr.load(p)
    assert back.dtype == np.dtype(dt)
    np.testing.assert_array_equal(back, a)


def test_save_appends_npy_extension(tmpdir):
    # numpy appends .npy when no extension is given; ferray must match.
    base = os.path.join(tmpdir, "noext")
    a = np.array([1.0, 2.0, 3.0])
    fr.save(base, a)
    assert os.path.exists(base + ".npy")
    np.testing.assert_array_equal(fr.load(base + ".npy"), a)


def test_ferray_save_read_by_numpy(tmpdir):
    # AC-2: a file ferray writes is loadable by numpy with identical content.
    p = os.path.join(tmpdir, "fr_to_np.npy")
    a = np.array([[1.5, -2.5], [3.0, 4.25]])
    fr.save(p, a)
    np.testing.assert_array_equal(np.load(p), a)


def test_numpy_save_read_by_ferray(tmpdir):
    # AC-2 (reverse): a file numpy writes is loadable by ferray.
    p = os.path.join(tmpdir, "np_to_fr.npy")
    a = np.array([10, 20, 30], dtype=np.int64)
    np.save(p, a)
    np.testing.assert_array_equal(fr.load(p), a)


# ---------------------------------------------------------------------------
# .npz savez / savez_compressed / load
# ---------------------------------------------------------------------------


def test_savez_keyword_names(tmpdir):
    p = os.path.join(tmpdir, "kw.npz")
    x = np.arange(3)
    y = np.ones(2)
    fr.savez(p, x=x, y=y)
    z = fr.load(p)
    assert set(z.keys()) == {"x", "y"}
    np.testing.assert_array_equal(z["x"], x)
    np.testing.assert_array_equal(z["y"], y)


def test_savez_positional_names(tmpdir):
    # numpy names positional args arr_0, arr_1, ...
    p = os.path.join(tmpdir, "pos.npz")
    a = np.array([1.0, 2.0])
    b = np.array([3, 4, 5], dtype=np.int64)
    fr.savez(p, a, b)
    z = fr.load(p)
    np.testing.assert_array_equal(z["arr_0"], a)
    np.testing.assert_array_equal(z["arr_1"], b)


def test_savez_compressed_roundtrip(tmpdir):
    p = os.path.join(tmpdir, "comp.npz")
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    fr.savez_compressed(p, data=a)
    z = fr.load(p)
    np.testing.assert_array_equal(z["data"], a)


def test_savez_appends_npz_extension(tmpdir):
    base = os.path.join(tmpdir, "arch")
    fr.savez(base, q=np.array([7.0, 8.0]))
    assert os.path.exists(base + ".npz")


def test_ferray_npz_read_by_numpy(tmpdir):
    # AC-3: ferray .npz is a valid archive numpy can open.
    p = os.path.join(tmpdir, "fr_arch.npz")
    fr.savez(p, x=np.arange(4), y=np.array([1.0, 2.0]))
    with np.load(p) as z:
        np.testing.assert_array_equal(z["x"], np.arange(4))
        np.testing.assert_array_equal(z["y"], np.array([1.0, 2.0]))


def test_numpy_npz_read_by_ferray(tmpdir):
    p = os.path.join(tmpdir, "np_arch.npz")
    np.savez(p, a=np.arange(3), b=np.ones(2))
    z = fr.load(p)
    np.testing.assert_array_equal(z["a"], np.arange(3))
    np.testing.assert_array_equal(z["b"], np.ones(2))


# ---------------------------------------------------------------------------
# delimited text: savetxt / loadtxt
# ---------------------------------------------------------------------------


def test_savetxt_loadtxt_2d_roundtrip(tmpdir):
    p = os.path.join(tmpdir, "grid.txt")
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fr.savetxt(p, a)
    back = fr.loadtxt(p)
    assert back.shape == (2, 3)
    np.testing.assert_allclose(back, a)


def test_savetxt_loadtxt_1d_roundtrip(tmpdir):
    p = os.path.join(tmpdir, "vec.txt")
    a = np.array([1.5, -2.25, 3.75])
    fr.savetxt(p, a)
    back = fr.loadtxt(p)
    assert back.shape == (3,)
    np.testing.assert_allclose(back, a)


def test_ferray_savetxt_read_by_numpy(tmpdir):
    # AC-5 (reverse): numpy.loadtxt parses what ferray.savetxt wrote.
    p = os.path.join(tmpdir, "fr_txt.txt")
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    fr.savetxt(p, a)
    np.testing.assert_allclose(np.loadtxt(p), a)


def test_numpy_savetxt_read_by_ferray(tmpdir):
    p = os.path.join(tmpdir, "np_txt.txt")
    a = np.array([[10.0, 20.0], [30.0, 40.0]])
    np.savetxt(p, a)
    np.testing.assert_allclose(fr.loadtxt(p), a)


def test_loadtxt_skiprows(tmpdir):
    p = os.path.join(tmpdir, "skip.txt")
    # A non-comment header row so `skiprows=1` is the only thing dropping it
    # (a leading '#' would also be stripped as a comment, double-counting).
    with open(p, "w") as f:
        f.write("col_a col_b\n")
        f.write("1.0 2.0\n")
        f.write("3.0 4.0\n")
    back = fr.loadtxt(p, skiprows=1)
    np.testing.assert_allclose(back, np.loadtxt(p, skiprows=1))


def test_loadtxt_custom_delimiter(tmpdir):
    p = os.path.join(tmpdir, "csv.txt")
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.savetxt(p, a, delimiter=",")
    back = fr.loadtxt(p, delimiter=",")
    np.testing.assert_allclose(back, a)


# ---------------------------------------------------------------------------
# genfromtxt — missing -> nan
# ---------------------------------------------------------------------------


def test_genfromtxt_missing_becomes_nan(tmpdir):
    p = os.path.join(tmpdir, "missing.txt")
    with open(p, "w") as f:
        f.write("1.0,2.0,3.0\n")
        f.write("4.0,,6.0\n")
    back = fr.genfromtxt(p, delimiter=",")
    expected = np.genfromtxt(p, delimiter=",")
    np.testing.assert_array_equal(np.isnan(back), np.isnan(expected))
    np.testing.assert_allclose(back[~np.isnan(back)], expected[~np.isnan(expected)])
