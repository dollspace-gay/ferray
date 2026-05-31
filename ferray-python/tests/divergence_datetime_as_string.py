"""ACToR critic pins — datetime64 0-d scalar shape preservation.

Audit context: `ferray-python/src/datetime.rs::datetime_as_string` (route design
`.design/ferray-core.md`; upstream numpy
`numpy/_core/src/multiarray/datetime_strings.c datetime_as_string`).

VERDICT for `datetime_as_string` itself: NO DIVERGENCE. The binding delegates to
`numpy.datetime_as_string` on the asarray'd input and matches numpy exactly across
scalar input, 0-d array, NaT, `timezone='UTC'`, casting `unit=`, timedelta
rejection (TypeError), and 2-D — including the `<U10` (0-d day scalar) vs `<U28`
(>=1-D day array) width split. Verified live numpy 2.4.x.

The divergence pinned here is ADJACENT, in the datetime *creation/coercion* path
(`fr.array` / `fr.asarray` over a datetime64 0-d scalar). numpy keeps a 0-d
datetime64 scalar as a 0-d array (shape `()`); ferray's datetime round-trip
coercion promotes it to a 1-element 1-D array (shape `(1,)`). A plain non-time
scalar (`fr.array(5)`) is NOT promoted (stays shape `()`), so this is specific to
the datetime64/timedelta64 coercion seam. This is the exact failure mode the
`datetime_as_string` doc-comment warns about ("promoted a 0-d scalar to a
1-element 1-D array"); the binding sidesteps it by NOT routing through the
round-trip, but `fr.array`/`fr.asarray` still exhibit it.

All expected values are taken LIVE from numpy in-test (R-CHAR-3); no target value
is literal-copied. Written to FAIL against the current ferray implementation.

Run: pytest tests/divergence_datetime_as_string.py -q
"""
import numpy as np
import ferray as fr


def test_array_datetime_scalar_preserves_0d_shape():
    """fr.array(datetime64 scalar) must stay 0-d (shape ()), like numpy.

    numpy: np.array(np.datetime64('2024-01-01')) -> shape (), dtype datetime64[D].
    ferray promotes to shape (1,).
    """
    exp = np.array(np.datetime64('2024-01-01'))
    got = fr.array(np.datetime64('2024-01-01'))
    assert np.asarray(got).shape == exp.shape, (
        f"array(datetime64 scalar) shape: numpy {exp.shape}, ferray "
        f"{np.asarray(got).shape}")


def test_asarray_datetime_scalar_preserves_0d_shape():
    """fr.asarray(datetime64 scalar) must stay 0-d (shape ()), like numpy."""
    exp = np.asarray(np.datetime64('2024-01-01'))
    got = fr.asarray(np.datetime64('2024-01-01'))
    assert np.asarray(got).shape == exp.shape, (
        f"asarray(datetime64 scalar) shape: numpy {exp.shape}, ferray "
        f"{np.asarray(got).shape}")


def test_array_datetime_0d_ndarray_preserves_0d_shape():
    """fr.array(0-d datetime64 ndarray) must stay 0-d (shape ()), like numpy."""
    src = np.array(np.datetime64('2024-01-01'))   # genuine 0-d M8[D]
    exp = np.array(src)
    got = fr.array(src)
    assert np.asarray(got).shape == exp.shape, (
        f"array(0-d datetime64 ndarray) shape: numpy {exp.shape}, ferray "
        f"{np.asarray(got).shape}")


def test_datetime_as_string_on_fr_array_scalar_is_0d():
    """End-to-end: numpy renders a 0-d datetime64 scalar to a 0-d <U10 string.

    fr.datetime_as_string(fr.array(scalar)) diverges because fr.array first
    promotes the scalar to shape (1,), so the rendered output is shape (1,)
    dtype <U28 instead of numpy's shape () dtype <U10.
    """
    exp = np.datetime_as_string(np.array(np.datetime64('2024-01-01')))
    got = fr.datetime_as_string(fr.array(np.datetime64('2024-01-01')))
    got = np.asarray(got)
    assert got.shape == exp.shape, (
        f"datetime_as_string(fr.array scalar) shape: numpy {exp.shape}, "
        f"ferray {got.shape}")
    assert str(got.dtype) == str(exp.dtype), (
        f"datetime_as_string(fr.array scalar) dtype: numpy {exp.dtype}, "
        f"ferray {got.dtype}")
