"""Expansion suite (#948): datetime64 / timedelta64 dtype-passthrough through the
manipulation surface (reshape / transpose / flip / roll / repeat / tile / stack /
concatenate and siblings).

These ops are PURE DATA MOVES — no element arithmetic — so they must preserve the
datetime64[unit] / timedelta64[unit] dtype, unit, NaT and values exactly the way
numpy does. ferray delegates each op to numpy on the coerced time array (numpy
owns the dtype-passthrough semantics) and marshals the result back through the
int64-view transport (#941).

Every expected value is DERIVED LIVE from numpy 2.4.x (R-CHAR-3): nothing is a
literal copied from the ferray side. Each test asserts dtype + values match the
numpy oracle, plus NaT is carried through, plus real/complex paths are unchanged.
"""
import numpy as np
import pytest
import ferray as fr


# --- live-oracle constructors -------------------------------------------------

def _dt(vals, unit="D"):
    return fr.array(vals, dtype=f"datetime64[{unit}]")


def _ndt(vals, unit="D"):
    return np.array(vals, dtype=f"datetime64[{unit}]")


def _td(vals, unit="D"):
    return fr.array(vals, dtype=f"timedelta64[{unit}]")


def _ntd(vals, unit="D"):
    return np.array(vals, dtype=f"timedelta64[{unit}]")


def _same(got, exp):
    """Assert ferray `got` matches numpy oracle `exp` in dtype + values."""
    g = np.asarray(got)
    assert str(g.dtype) == str(exp.dtype), f"dtype: numpy {exp.dtype}, ferray {g.dtype}"
    assert g.shape == exp.shape, f"shape: numpy {exp.shape}, ferray {g.shape}"
    # NaT != NaT, so compare on the int64 view (NaT == i64::MIN) for equality.
    assert np.array_equal(g.view("int64"), exp.view("int64")), (
        f"values: numpy {exp!r}, ferray {g!r}")


_DAYS = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
_DT_NaT = ["2020-01-01", "NaT", "2020-01-03"]
_TD_NaT = [2, 4, 6]


# --- reshape (incl -1 and order) ----------------------------------------------

@pytest.mark.parametrize("unit", ["D", "M", "s"])
def test_reshape_datetime(unit):
    exp = np.reshape(_ndt(_DAYS, unit), (2, 2))
    _same(fr.reshape(_dt(_DAYS, unit), (2, 2)), exp)


def test_reshape_minus_one():
    exp = np.reshape(_ndt(_DAYS), (-1, 2))
    _same(fr.reshape(_dt(_DAYS), (-1, 2)), exp)


def test_reshape_order_f():
    exp = np.reshape(_ndt(_DAYS), (2, 2), order="F")
    _same(fr.reshape(_dt(_DAYS), (2, 2), order="F"), exp)


def test_reshape_timedelta():
    exp = np.reshape(_ntd([1, 2, 3, 4]), (2, 2))
    _same(fr.reshape(_td([1, 2, 3, 4]), (2, 2)), exp)


# --- transpose / swapaxes / moveaxis ------------------------------------------

def test_transpose_default():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.transpose(fa), np.transpose(a))


def test_transpose_axes():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.transpose(fa, [1, 0]), np.transpose(a, [1, 0]))


def test_swapaxes_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.swapaxes(fa, 0, 1), np.swapaxes(a, 0, 1))


def test_moveaxis_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.moveaxis(fa, 0, 1), np.moveaxis(a, 0, 1))


# --- flip / fliplr / flipud / rot90 -------------------------------------------

def test_flip_datetime():
    _same(fr.flip(_dt(_DAYS)), np.flip(_ndt(_DAYS)))


def test_flip_timedelta():
    _same(fr.flip(_td([1, 2, 3])), np.flip(_ntd([1, 2, 3])))


def test_fliplr_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.fliplr(fa), np.fliplr(a))


def test_flipud_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.flipud(fa), np.flipud(a))


def test_rot90_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.rot90(fa), np.rot90(a))


# --- roll ---------------------------------------------------------------------

@pytest.mark.parametrize("shift", [1, -1, 3])
def test_roll_datetime(shift):
    _same(fr.roll(_dt(_DAYS), shift), np.roll(_ndt(_DAYS), shift))


def test_roll_axis():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.roll(fa, 1, axis=0), np.roll(a, 1, axis=0))


# --- repeat -------------------------------------------------------------------

def test_repeat_scalar():
    _same(fr.repeat(_dt(_DAYS), 2), np.repeat(_ndt(_DAYS), 2))


def test_repeat_per_element():
    _same(fr.repeat(_dt(_DAYS), [1, 2, 1, 3]), np.repeat(_ndt(_DAYS), [1, 2, 1, 3]))


def test_repeat_axis():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.repeat(fa, 2, axis=1), np.repeat(a, 2, axis=1))


def test_repeat_timedelta():
    _same(fr.repeat(_td([1, 2, 3]), 2), np.repeat(_ntd([1, 2, 3]), 2))


# --- tile ---------------------------------------------------------------------

def test_tile_scalar():
    _same(fr.tile(_dt(_DAYS), 2), np.tile(_ndt(_DAYS), 2))


def test_tile_tuple():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.tile(fa, (2, 1)), np.tile(a, (2, 1)))


# --- stack / vstack / hstack / column_stack / dstack --------------------------

def test_stack_datetime():
    exp = np.stack([_ndt(_DAYS), _ndt(_DAYS)])
    _same(fr.stack([_dt(_DAYS), _dt(_DAYS)]), exp)


def test_stack_axis1():
    exp = np.stack([_ndt(_DAYS), _ndt(_DAYS)], axis=1)
    _same(fr.stack([_dt(_DAYS), _dt(_DAYS)], axis=1), exp)


def test_vstack_datetime():
    exp = np.vstack([_ndt(_DAYS), _ndt(_DAYS)])
    _same(fr.vstack([_dt(_DAYS), _dt(_DAYS)]), exp)


def test_hstack_datetime():
    exp = np.hstack([_ndt(_DAYS), _ndt(_DAYS)])
    _same(fr.hstack([_dt(_DAYS), _dt(_DAYS)]), exp)


def test_column_stack_datetime():
    exp = np.column_stack([_ndt(_DAYS), _ndt(_DAYS)])
    _same(fr.column_stack([_dt(_DAYS), _dt(_DAYS)]), exp)


def test_stack_timedelta():
    exp = np.stack([_ntd([1, 2, 3]), _ntd([4, 5, 6])])
    _same(fr.stack([_td([1, 2, 3]), _td([4, 5, 6])]), exp)


# --- concatenate (same unit, cross-unit -> finer, axis=None) ------------------

def test_concatenate_same_unit():
    exp = np.concatenate([_ndt(["2020-01-01"]), _ndt(["2020-01-02"])])
    _same(fr.concatenate([_dt(["2020-01-01"]), _dt(["2020-01-02"])]), exp)


def test_concatenate_cross_unit_promotes_finer():
    # datetime64[D] + datetime64[s] -> datetime64[s] (the finer unit), live.
    a, b = _ndt(_DAYS, "D"), _ndt(_DAYS, "s")
    exp = np.concatenate([a, b])
    assert str(exp.dtype) == "datetime64[s]"
    got = fr.concatenate([_dt(_DAYS, "D"), _dt(_DAYS, "s")])
    _same(got, exp)


def test_concatenate_axis_none():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.concatenate([fa, fa], axis=None), np.concatenate([a, a], axis=None))


def test_concatenate_timedelta():
    exp = np.concatenate([_ntd([1, 2]), _ntd([3, 4])])
    _same(fr.concatenate([_td([1, 2]), _td([3, 4])]), exp)


# --- siblings: squeeze / expand_dims / broadcast_to / ravel / flatten ---------

def test_ravel_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.ravel(fa), np.ravel(a))


def test_flatten_datetime():
    a = _ndt(_DAYS).reshape(2, 2)
    fa = fr.reshape(_dt(_DAYS), (2, 2))
    _same(fr.flatten(fa), a.flatten())


def test_expand_dims_datetime():
    _same(fr.expand_dims(_dt(_DAYS), 0), np.expand_dims(_ndt(_DAYS), 0))


def test_squeeze_datetime():
    a = _ndt(_DAYS).reshape(1, 4)
    fa = fr.reshape(_dt(_DAYS), (1, 4))
    _same(fr.squeeze(fa), np.squeeze(a))


def test_atleast_2d_datetime():
    _same(fr.atleast_2d(_dt(_DAYS)), np.atleast_2d(_ndt(_DAYS)))


# --- delete / insert / append / resize ----------------------------------------

def test_delete_datetime():
    _same(fr.delete(_dt(_DAYS), 1, 0), np.delete(_ndt(_DAYS), 1, 0))


def test_insert_datetime():
    exp = np.insert(_ndt(_DAYS), 1, np.datetime64("2019-01-01"))
    _same(fr.insert(_dt(_DAYS), 1, fr.datetime64("2019-01-01"), 0), exp)


def test_append_datetime():
    exp = np.append(_ndt(_DAYS), np.datetime64("2019-01-01"))
    _same(fr.append(_dt(_DAYS), fr.datetime64("2019-01-01")), exp)


def test_resize_datetime():
    exp = np.resize(_ndt(_DAYS), (6,))
    _same(fr.resize(_dt(_DAYS), (6,)), exp)


# --- split family (returns a list of datetime arrays) -------------------------

def test_split_datetime():
    exp = np.split(_ndt(_DAYS), 2)
    got = fr.split(_dt(_DAYS), 2)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        _same(g, e)


def test_array_split_datetime():
    exp = np.array_split(_ndt(_DAYS), 3)
    got = fr.array_split(_dt(_DAYS), 3)
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        _same(g, e)


# --- pad / block --------------------------------------------------------------

def test_pad_datetime_epoch_fill():
    exp = np.pad(_ndt(_DAYS), (1, 1))   # constant fill 0 -> epoch 1970-01-01
    _same(fr.pad(_dt(_DAYS), (1, 1)), exp)


def test_block_datetime():
    # ferray.block requires a nested-list grid for every dtype (pre-existing
    # input contract), so use the [[...]] form numpy also accepts.
    exp = np.block([[_ndt(["2020-01-01"]), _ndt(["2020-01-02"])]])
    got = fr.block([[_dt(["2020-01-01"]), _dt(["2020-01-02"])]])
    _same(got, exp)


# --- NaT preservation ---------------------------------------------------------

def test_reshape_preserves_nat():
    a = _ndt(_DT_NaT)
    exp = np.flip(a)
    got = fr.flip(_dt(_DT_NaT))
    _same(got, exp)
    # NaT really survived (not coerced to a finite tick).
    assert np.isnat(np.asarray(got)).sum() == np.isnat(exp).sum() == 1


def test_concatenate_preserves_nat():
    exp = np.concatenate([_ndt(["NaT", "2020-01-01"]), _ndt(["2020-01-02"])])
    got = fr.concatenate([_dt(["NaT", "2020-01-01"]), _dt(["2020-01-02"])])
    _same(got, exp)
    assert np.isnat(np.asarray(got))[0]


def test_repeat_preserves_nat_timedelta():
    exp = np.repeat(_ntd([2, np.iinfo(np.int64).min, 6]), 2)
    a = fr.array([2, "NaT", 6], dtype="timedelta64[D]")
    got = fr.repeat(a, 2)
    _same(got, exp)


# --- real / complex paths UNCHANGED (no regression) ---------------------------

def test_real_path_unchanged():
    a = np.arange(4.0)
    _same_real = lambda g, e: (
        np.asarray(g).dtype == e.dtype and np.array_equal(np.asarray(g), e))
    assert _same_real(fr.reshape(fr.array(a), (2, 2)), np.reshape(a, (2, 2)))
    assert _same_real(fr.flip(fr.array(a)), np.flip(a))
    assert _same_real(fr.roll(fr.array(a), 1), np.roll(a, 1))
    assert _same_real(fr.repeat(fr.array(a), 2), np.repeat(a, 2))
    got = fr.concatenate([fr.array(a), fr.array(a)])
    assert _same_real(got, np.concatenate([a, a]))


def test_int_path_unchanged():
    a = np.arange(4, dtype="int32")
    got = fr.reshape(fr.array(a), (2, 2))
    g = np.asarray(got)
    assert g.dtype == np.int32 and np.array_equal(g, np.reshape(a, (2, 2)))


def test_complex_path_unchanged():
    a = np.array([1 + 2j, 3 + 4j], dtype="complex128")
    got = fr.flip(fr.array(a))
    g = np.asarray(got)
    assert g.dtype == np.complex128 and np.array_equal(g, np.flip(a))
