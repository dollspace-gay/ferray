"""Expansion suite for blocker #933 (epic #919): complex dtype-passthrough
manipulation ops.

reshape / transpose / concatenate / stack / vstack / hstack /
column_stack / dstack / where / flip / fliplr / flipud / roll / repeat /
tile / squeeze / expand_dims / moveaxis / swapaxes / atleast_* /
broadcast_to / split / rot90 / diag / delete / insert / append / resize
do **no arithmetic** — they are pure data moves / views. numpy preserves
the input dtype (incl. complex64 / complex128) and the exact element
values through every one of them. Before this build ferray raised
``TypeError: unsupported dtype: complex128`` because the manipulation
dispatch macro only carried the 11 real dtypes.

Every expected value here is derived **live** from numpy 2.x (the oracle)
— never literal-copied from ferray (R-CHAR-3). Each case asserts the
ferray result matches numpy in **both dtype and values**, and a final
group asserts the real-dtype path is unchanged.
"""

import numpy as np
import pytest

import ferray as fr

# A non-trivial complex sample reused across cases; values chosen so the
# imaginary parts are all distinct and non-zero (a silent ->real cast
# would visibly corrupt them).
Z1 = [1 + 2j, -3 + 4j, 0 - 5j, 2 - 1j, 6 + 0j, -7 - 8j]
Z2D = [[1 + 1j, 2 - 2j, 3 + 3j], [4 - 4j, 5 + 5j, 6 - 6j]]


def _eq(got, expected):
    """Assert ferray result matches numpy in dtype AND values."""
    g = np.asarray(got)
    e = np.asarray(expected)
    assert g.dtype == e.dtype, f"dtype: ferray {g.dtype} vs numpy {e.dtype}"
    assert g.shape == e.shape, f"shape: ferray {g.shape} vs numpy {e.shape}"
    assert np.array_equal(g, e), f"values: ferray {g!r} vs numpy {e!r}"


@pytest.mark.parametrize("cdtype", [np.complex128, np.complex64])
class TestComplexPassthrough:
    """Each op preserves dtype+values for both complex128 and complex64."""

    def test_reshape(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.reshape(fr.array(z), (2, 3)), np.reshape(z, (2, 3)))

    def test_reshape_neg1_inference(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.reshape(fr.array(z), (-1, 2)), np.reshape(z, (-1, 2)))

    def test_reshape_fortran_order(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(
            fr.reshape(fr.array(z), (2, 3), order="F"),
            np.reshape(z, (2, 3), order="F"),
        )

    def test_ravel(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.ravel(fr.array(z)), np.ravel(z))

    def test_flatten(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.flatten(fr.array(z)), np.array(z).flatten())

    def test_transpose(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.transpose(fr.array(z)), np.transpose(z))

    def test_transpose_axes_3d(self, cdtype):
        z = (np.arange(24).reshape(2, 3, 4) + 1j * np.arange(24).reshape(2, 3, 4)).astype(
            cdtype
        )
        _eq(fr.transpose(fr.array(z), (2, 0, 1)), np.transpose(z, (2, 0, 1)))

    def test_swapaxes(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.swapaxes(fr.array(z), 0, 1), np.swapaxes(z, 0, 1))

    def test_moveaxis(self, cdtype):
        z = (np.arange(24).reshape(2, 3, 4) + 1j).astype(cdtype)
        _eq(fr.moveaxis(fr.array(z), 0, 2), np.moveaxis(z, 0, 2))

    def test_squeeze(self, cdtype):
        z = np.array(Z1, dtype=cdtype).reshape(1, 6, 1)
        _eq(fr.squeeze(fr.array(z)), np.squeeze(z))

    def test_expand_dims(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.expand_dims(fr.array(z), 0), np.expand_dims(z, 0))

    def test_atleast_1d(self, cdtype):
        z = np.array(1 + 2j, dtype=cdtype)
        _eq(fr.atleast_1d(fr.array(z)), np.atleast_1d(z))

    def test_atleast_2d(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.atleast_2d(fr.array(z)), np.atleast_2d(z))

    def test_atleast_3d(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.atleast_3d(fr.array(z)), np.atleast_3d(z))

    def test_flip(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.flip(fr.array(z)), np.flip(z))

    def test_flip_axis(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.flip(fr.array(z), 1), np.flip(z, 1))

    def test_fliplr(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.fliplr(fr.array(z)), np.fliplr(z))

    def test_flipud(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.flipud(fr.array(z)), np.flipud(z))

    def test_rot90(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.rot90(fr.array(z)), np.rot90(z))

    def test_roll(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.roll(fr.array(z), 2), np.roll(z, 2))

    def test_roll_axis(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.roll(fr.array(z), 1, axis=1), np.roll(z, 1, axis=1))

    def test_repeat_scalar(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.repeat(fr.array(z), 3), np.repeat(z, 3))

    def test_repeat_per_element(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        counts = [1, 2, 3, 0, 1, 2]
        _eq(fr.repeat(fr.array(z), counts), np.repeat(z, counts))

    def test_repeat_axis(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)
        _eq(fr.repeat(fr.array(z), 2, axis=0), np.repeat(z, 2, axis=0))

    def test_tile_scalar(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.tile(fr.array(z), 2), np.tile(z, 2))

    def test_tile_reps_tuple(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.tile(fr.array(z), (2, 2)), np.tile(z, (2, 2)))

    def test_concatenate(self, cdtype):
        a = np.array([1 + 2j, 3 + 4j], dtype=cdtype)
        b = np.array([5 - 6j, 7 - 8j], dtype=cdtype)
        _eq(
            fr.concatenate([fr.array(a), fr.array(b)]),
            np.concatenate([a, b]),
        )

    def test_concatenate_axis1(self, cdtype):
        a = np.array(Z2D, dtype=cdtype)
        b = np.array(Z2D, dtype=cdtype)
        _eq(
            fr.concatenate([fr.array(a), fr.array(b)], axis=1),
            np.concatenate([a, b], axis=1),
        )

    def test_stack(self, cdtype):
        a = np.array([1 + 2j, 3 + 4j], dtype=cdtype)
        b = np.array([5 - 6j, 7 - 8j], dtype=cdtype)
        _eq(fr.stack([fr.array(a), fr.array(b)]), np.stack([a, b]))

    def test_vstack(self, cdtype):
        a = np.array([1 + 2j, 3 + 4j], dtype=cdtype)
        b = np.array([5 - 6j, 7 - 8j], dtype=cdtype)
        _eq(fr.vstack([fr.array(a), fr.array(b)]), np.vstack([a, b]))

    def test_hstack(self, cdtype):
        a = np.array([1 + 2j, 3 + 4j], dtype=cdtype)
        b = np.array([5 - 6j, 7 - 8j], dtype=cdtype)
        _eq(fr.hstack([fr.array(a), fr.array(b)]), np.hstack([a, b]))

    def test_column_stack(self, cdtype):
        a = np.array([1 + 2j, 3 + 4j], dtype=cdtype)
        b = np.array([5 - 6j, 7 - 8j], dtype=cdtype)
        _eq(fr.column_stack([fr.array(a), fr.array(b)]), np.column_stack([a, b]))

    def test_dstack(self, cdtype):
        a = np.array([1 + 2j, 3 + 4j], dtype=cdtype)
        b = np.array([5 - 6j, 7 - 8j], dtype=cdtype)
        _eq(fr.dstack([fr.array(a), fr.array(b)]), np.dstack([a, b]))

    def test_split(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        got = fr.split(fr.array(z), 3)
        exp = np.split(z, 3)
        assert len(got) == len(exp)
        for g, e in zip(got, exp):
            _eq(g, e)

    def test_delete(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.delete(fr.array(z), [1, 3], 0), np.delete(z, [1, 3], 0))

    def test_insert(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        v = np.array([9 + 9j], dtype=cdtype)
        _eq(fr.insert(fr.array(z), 2, fr.array(v), 0), np.insert(z, 2, v, 0))

    def test_append(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        v = np.array([9 + 9j, 8 - 8j], dtype=cdtype)
        _eq(fr.append(fr.array(z), fr.array(v)), np.append(z, v))

    def test_resize(self, cdtype):
        z = np.array(Z1, dtype=cdtype)
        _eq(fr.resize(fr.array(z), (3, 4)), np.resize(z, (3, 4)))

    def test_diag_extract(self, cdtype):
        z = np.array(Z2D, dtype=cdtype)[:, :2]
        _eq(fr.diag(fr.array(z)), np.diag(z))

    def test_where_same_complex(self, cdtype):
        a = np.array([1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j], dtype=cdtype)
        b = np.array([2 + 0j, 1 - 1j, 3 + 3j, 1 + 0j], dtype=cdtype)
        mask = np.array([True, False, True, False])
        _eq(
            fr.where(fr.array(mask), fr.array(a), fr.array(b)),
            np.where(mask, a, b),
        )


# ---------------------------------------------------------------------------
# Cross-dtype promotion: complex + real -> complex (numpy result_type).
# ---------------------------------------------------------------------------


def test_concatenate_complex_plus_real_promotes():
    a = np.array([1 + 2j, 3 + 4j], dtype=np.complex128)
    r = np.array([5.0, 6.0], dtype=np.float64)
    got = np.asarray(fr.concatenate([fr.array(a), fr.array(r)]))
    exp = np.concatenate([a, r])
    assert exp.dtype == np.complex128  # numpy oracle: promotion target
    _eq(got, exp)


def test_concatenate_complex64_plus_float32_promotes():
    a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    r = np.array([5.0, 6.0], dtype=np.float32)
    got = np.asarray(fr.concatenate([fr.array(a), fr.array(r)]))
    exp = np.concatenate([a, r])
    assert exp.dtype == np.complex64
    _eq(got, exp)


def test_where_complex_x_real_y_promotes():
    mask = np.array([True, False, True, False])
    x = np.array([1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j], dtype=np.complex128)
    y = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    got = np.asarray(fr.where(fr.array(mask), fr.array(x), fr.array(y)))
    exp = np.where(mask, x, y)
    assert exp.dtype == np.complex128
    _eq(got, exp)


def test_where_real_x_complex_y_promotes():
    # The complex operand is `y` (second position); numpy still promotes
    # the output to complex. A naive "take x's dtype" binding would drop
    # the imaginary part.
    mask = np.array([True, False, True, False])
    x = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    y = np.array([1 + 2j, -3 + 4j, 0 + 0j, 2 - 1j], dtype=np.complex128)
    got = np.asarray(fr.where(fr.array(mask), fr.array(x), fr.array(y)))
    exp = np.where(mask, x, y)
    assert exp.dtype == np.complex128
    _eq(got, exp)


def test_stack_complex64_plus_complex128_promotes():
    a = np.array([1 + 2j, 3 + 4j], dtype=np.complex64)
    b = np.array([5 - 6j, 7 - 8j], dtype=np.complex128)
    got = np.asarray(fr.stack([fr.array(a), fr.array(b)]))
    exp = np.stack([a, b])
    assert exp.dtype == np.complex128
    _eq(got, exp)


# ---------------------------------------------------------------------------
# Real-dtype path is unchanged (no regression from the complex arms).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rdtype", [np.float64, np.int64, np.int32, np.uint8, np.bool_])
def test_real_path_reshape_unchanged(rdtype):
    a = np.arange(6).astype(rdtype)
    _eq(fr.reshape(fr.array(a), (2, 3)), np.reshape(a, (2, 3)))


@pytest.mark.parametrize("rdtype", [np.float64, np.int64])
def test_real_path_concatenate_unchanged(rdtype):
    a = np.arange(3).astype(rdtype)
    b = np.arange(3, 6).astype(rdtype)
    _eq(fr.concatenate([fr.array(a), fr.array(b)]), np.concatenate([a, b]))


def test_real_path_where_unchanged():
    mask = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    _eq(fr.where(fr.array(mask), fr.array(x), fr.array(y)), np.where(mask, x, y))
