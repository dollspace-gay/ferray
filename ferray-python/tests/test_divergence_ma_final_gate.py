"""ACToR critic — fr.ma final-gate divergence pins.

Each test derives its EXPECTED value LIVE from numpy.ma (R-CHAR-3) and
asserts the ferray.ma result matches. A FAILING test pins a real divergence.

Divergence found at HEAD 15b6788e:
  Module-level reduction wrappers (numpy.ma.sum/prod/mean/max/min/ptp/std/
  var/any/all/count) accept an `axis` kwarg in numpy but ferray.ma's free
  functions take only the array (full reduction). The METHOD form
  (arr.sum(axis=...)) works in ferray; the MODULE form does not.

  numpy/ma/core.py: `sum = _frommethod('sum')` etc. — _frommethod forwards
  *args/**kwargs (incl. axis) to the bound method. ferray binds these as
  `pub fn sum(py, a: &PyMaskedArray)` (src/ma.rs:6913+ region) with no axis.

  Impact: HIGH. `np.ma.sum(a, axis=0)` is a ubiquitous idiom.
"""

import numpy as np
import numpy.ma as nma
import ferray.ma as fma
import pytest


def _mk():
    data = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    mask = [[0, 1, 0], [0, 0, 1]]
    return nma.array(data, mask=mask), fma.array(data, mask=mask)


_AXIS_REDUCERS = ["sum", "prod", "mean", "max", "min", "ptp", "std", "var", "any", "all"]


@pytest.mark.parametrize("fn", _AXIS_REDUCERS)
@pytest.mark.parametrize("axis", [0, 1])
def test_module_reduction_axis_kwarg(fn, axis):
    """numpy.ma.<fn>(a, axis=k) must work and match; ferray rejects axis."""
    a_np, a_fr = _mk()
    expected = getattr(nma, fn)(a_np, axis=axis)
    exp_vals = np.ma.filled(np.ma.asarray(expected), 0.0)

    got = getattr(fma, fn)(a_fr, axis=axis)  # diverges: TypeError (no axis param)
    got_vals = np.asarray(got.filled(0.0) if hasattr(got, "filled") else got)

    assert np.allclose(np.asarray(exp_vals, dtype=float),
                       np.asarray(got_vals, dtype=float)), \
        f"{fn} axis={axis}: numpy {exp_vals} vs ferray {got_vals}"


def test_module_count_axis_kwarg():
    """numpy.ma.count(a, axis=k) returns per-axis unmasked counts."""
    a_np, a_fr = _mk()
    expected = nma.count(a_np, axis=0)
    got = fma.count(a_fr, axis=0)  # diverges: TypeError (no axis param)
    assert np.array_equal(np.asarray(expected), np.asarray(got)), \
        f"count axis=0: numpy {expected} vs ferray {got}"


def test_module_sum_axis_positional():
    """numpy.ma.sum(a, 0) accepts axis positionally; ferray rejects it."""
    a_np, a_fr = _mk()
    expected = np.ma.filled(nma.sum(a_np, 0), 0.0)
    got = fma.sum(a_fr, 0)  # diverges: takes 1 positional arg
    got_vals = np.asarray(got.filled(0.0) if hasattr(got, "filled") else got)
    assert np.allclose(np.asarray(expected, dtype=float),
                       np.asarray(got_vals, dtype=float))
